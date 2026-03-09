#!/usr/bin/env python3
"""
Category C: Convert SWC skeleton files to meshes, then to Neuroglancer
precomputed mesh format.

For neurons that have SWC tracings but no proper mesh (only a point-cloud OBJ),
this script inflates the skeleton into a tubular mesh and writes it as
precomputed format.

Two approaches are provided:
  1. navis-based (preferred): Uses navis.conversion.tree2meshneuron() with CGAL
  2. trimesh-based (fallback): Manually creates truncated cones along each edge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import requests
import trimesh
from cloudvolume import CloudVolume
from cloudvolume.mesh import Mesh


def vfb_image_url(vfb_id: str, template_id: str, filename: str) -> str:
    prefix = vfb_id.replace("VFB_", "")
    first4, last4 = prefix[:4], prefix[4:]
    return f"https://www.virtualflybrain.org/data/VFB/i/{first4}/{last4}/{template_id}/{filename}"


def download_file(url: str, dest: str) -> str:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(1024 * 1024):
            f.write(chunk)
    return dest


# ---------------------------------------------------------------------------
# SWC parsing
# ---------------------------------------------------------------------------

def parse_swc(filepath: str) -> dict:
    """Parse an SWC file into a dict of nodes.

    Returns:
        {node_id: {"x": float, "y": float, "z": float,
                    "radius": float, "parent": int, "type": int}}
    """
    nodes = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            node_id = int(parts[0])
            nodes[node_id] = {
                "type": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "radius": float(parts[5]),
                "parent": int(parts[6]),
            }
    return nodes


# ---------------------------------------------------------------------------
# Approach 1: navis-based conversion (preferred)
# ---------------------------------------------------------------------------

def swc_to_mesh_navis(swc_path: str, verbose: bool = True) -> trimesh.Trimesh:
    """Convert SWC to mesh using navis (requires navis + CGAL or scipy)."""
    try:
        import navis
    except ImportError:
        raise ImportError("navis required for this method. Install with: pip install navis")

    if verbose:
        print("  Using navis for SWC → mesh conversion")

    neuron = navis.read_swc(swc_path)

    if verbose:
        print(f"  Loaded neuron: {neuron.n_nodes} nodes, {neuron.n_branches} branches")

    # Convert TreeNeuron to MeshNeuron
    mesh_neuron = navis.conversion.tree2meshneuron(neuron)

    if verbose:
        print(f"  Generated mesh: {len(mesh_neuron.vertices)} vertices, {len(mesh_neuron.faces)} faces")

    return trimesh.Trimesh(
        vertices=mesh_neuron.vertices,
        faces=mesh_neuron.faces,
    )


# ---------------------------------------------------------------------------
# Approach 2: trimesh-based tube generation (fallback)
# ---------------------------------------------------------------------------

def create_tube_segment(p1: np.ndarray, p2: np.ndarray, r1: float, r2: float,
                        n_sides: int = 8) -> trimesh.Trimesh:
    """Create a truncated cone (tube segment) between two points."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return None

    # Create a unit cylinder and transform it
    # Use trimesh's creation utilities
    cylinder = trimesh.creation.cylinder(
        radius=1.0, height=length, sections=n_sides
    )

    # Scale radii: cylinder goes from -height/2 to +height/2 along Z
    verts = cylinder.vertices.copy()
    # Scale top vs bottom radius
    top_mask = verts[:, 2] > 0
    bottom_mask = ~top_mask
    verts[top_mask, 0] *= r2
    verts[top_mask, 1] *= r2
    verts[bottom_mask, 0] *= r1
    verts[bottom_mask, 1] *= r1

    cylinder.vertices = verts

    # Align cylinder to the direction vector
    direction_norm = direction / length
    z_axis = np.array([0, 0, 1.0])

    if np.allclose(direction_norm, z_axis):
        rotation = np.eye(4)
    elif np.allclose(direction_norm, -z_axis):
        rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    else:
        axis = np.cross(z_axis, direction_norm)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1, 1))
        rotation = trimesh.transformations.rotation_matrix(angle, axis)

    cylinder.apply_transform(rotation)

    # Translate to midpoint
    midpoint = (p1 + p2) / 2.0
    cylinder.apply_translation(midpoint)

    return cylinder


def swc_to_mesh_tubes(swc_path: str, tube_sides: int = 8,
                      min_radius: float = 0.05, verbose: bool = True) -> trimesh.Trimesh:
    """Convert SWC to mesh by creating tube segments along each edge."""
    nodes = parse_swc(swc_path)

    if verbose:
        print(f"  Using tube generation: {len(nodes)} nodes, {tube_sides} sides per tube")

    meshes = []
    for node_id, node in nodes.items():
        parent_id = node["parent"]
        if parent_id < 0 or parent_id not in nodes:
            continue

        parent = nodes[parent_id]
        p1 = np.array([parent["x"], parent["y"], parent["z"]])
        p2 = np.array([node["x"], node["y"], node["z"]])
        r1 = max(parent["radius"], min_radius)
        r2 = max(node["radius"], min_radius)

        tube = create_tube_segment(p1, p2, r1, r2, n_sides=tube_sides)
        if tube is not None:
            meshes.append(tube)

    # Also add spheres at branch points for smoother junctions
    branch_points = set()
    child_count = {}
    for node in nodes.values():
        pid = node["parent"]
        if pid > 0:
            child_count[pid] = child_count.get(pid, 0) + 1
    for nid, count in child_count.items():
        if count > 1:
            branch_points.add(nid)

    for nid in branch_points:
        node = nodes[nid]
        r = max(node["radius"], min_radius)
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=r)
        sphere.apply_translation([node["x"], node["y"], node["z"]])
        meshes.append(sphere)

    if not meshes:
        raise ValueError("No tube segments could be generated from SWC")

    if verbose:
        print(f"  Created {len(meshes)} tube/sphere primitives, merging...")

    combined = trimesh.util.concatenate(meshes)

    if verbose:
        print(f"  Merged mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    return combined


# ---------------------------------------------------------------------------
# Write precomputed
# ---------------------------------------------------------------------------

def write_precomputed_mesh(mesh: trimesh.Trimesh, output_dir: str, vfb_id: str,
                           resolution: list[float], segment_id: int = 1,
                           label: str | None = None, verbose: bool = True):
    """Write a trimesh to Neuroglancer precomputed mesh format."""

    dest_local = os.path.join(output_dir, vfb_id)
    os.makedirs(dest_local, exist_ok=True)
    dest = f"file://{dest_local}"

    # Compute size from mesh bounds
    mesh_max = mesh.vertices.max(axis=0)
    size = [int(np.ceil(mesh_max[i] / resolution[i])) + 2 for i in range(3)]

    info = {
        "data_type": "uint32",
        "num_channels": 1,
        "scales": [{
            "chunk_sizes": [[64, 64, 64]],
            "encoding": "raw",
            "key": "0",
            "resolution": resolution,
            "size": size,
            "voxel_offset": [0, 0, 0],
        }],
        "type": "segmentation",
        "mesh": "mesh",
        "segment_properties": "segment_properties",
    }

    vol = CloudVolume(dest, mip=0, info=info, compress=False)
    vol.commit_info()

    # Mesh directory
    mesh_dir = os.path.join(dest_local, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    with open(os.path.join(mesh_dir, "info"), "w") as f:
        json.dump({"@type": "neuroglancer_legacy_mesh"}, f, indent=2)

    # Write mesh
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    mesh_obj = Mesh(vertices, faces, segid=segment_id)
    vol.mesh.put(mesh_obj, compress=True)

    # Segment properties
    seg_dir = os.path.join(dest_local, "segment_properties")
    os.makedirs(seg_dir, exist_ok=True)
    display_label = label or vfb_id
    seg_info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [str(segment_id)],
            "properties": [
                {"id": "label", "type": "label", "values": [display_label]},
                {"id": "description", "type": "description", "values": [vfb_id]},
            ],
        },
    }
    with open(os.path.join(seg_dir, "info"), "w") as f:
        json.dump(seg_info, f, indent=2)

    if verbose:
        print(f"  Wrote precomputed mesh to {dest_local}")

    return dest_local


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert SWC skeleton files to Neuroglancer precomputed mesh format"
    )
    parser.add_argument("--input-swc", default=None,
                        help="Path to local SWC file")
    parser.add_argument("--vfb-id", default=None,
                        help="VFB image ID to download SWC from server")
    parser.add_argument("--template-id", default="VFB_00101567",
                        help="Template ID for URL construction (default: JRC2018Unisex)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for precomputed datasets")
    parser.add_argument("--resolution", type=float, nargs=3,
                        default=[518.9161, 518.9161, 1000.0],
                        help="Voxel resolution in nm [x y z] (default: JRC2018U)")
    parser.add_argument("--method", choices=["navis", "tubes"], default="navis",
                        help="Mesh generation method (default: navis)")
    parser.add_argument("--tube-sides", type=int, default=8,
                        help="Number of sides per tube segment (tubes method only, default: 8)")
    parser.add_argument("--label", default=None,
                        help="Display label for the mesh segment")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.input_swc and not args.vfb_id:
        parser.error("Must provide either --input-swc or --vfb-id")

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    swc_path = args.input_swc
    vfb_id = args.vfb_id or os.path.splitext(os.path.basename(args.input_swc))[0]
    tmp_swc = None

    if not swc_path:
        url = vfb_image_url(args.vfb_id, args.template_id, "volume.swc")
        if args.verbose:
            print(f"Downloading SWC: {url}")
        tmp_swc = tempfile.NamedTemporaryFile(suffix=".swc", delete=False)
        swc_path = tmp_swc.name
        tmp_swc.close()
        download_file(url, swc_path)

    try:
        if args.verbose:
            print(f"Converting SWC: {swc_path}")

        if args.method == "navis":
            try:
                mesh = swc_to_mesh_navis(swc_path, verbose=args.verbose)
            except ImportError:
                print("  navis not available, falling back to tube method")
                mesh = swc_to_mesh_tubes(swc_path, tube_sides=args.tube_sides,
                                         verbose=args.verbose)
        else:
            mesh = swc_to_mesh_tubes(swc_path, tube_sides=args.tube_sides,
                                     verbose=args.verbose)

        write_precomputed_mesh(
            mesh, output_dir, vfb_id,
            resolution=args.resolution, label=args.label, verbose=args.verbose,
        )
    finally:
        if tmp_swc:
            os.unlink(swc_path)

    print(f"Done. Output at: {output_dir}/{vfb_id}")


if __name__ == "__main__":
    main()
