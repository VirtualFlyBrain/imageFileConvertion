#!/usr/bin/env python3
"""
VFB Jenkins Pipeline: Generate missing volume_man.obj files from SWC skeletons
and create Neuroglancer precomputed mesh data.

Designed to run on the VFB Jenkins server where
https://www.virtualflybrain.org/data/ is mounted at /IMAGE_WRITE/

Workflow:
  1. Discover image directories (filesystem scan or KB query)
  2. For each image with volume.swc but no volume_man.obj → generate OBJ
  3. For each image with volume_man.obj but no neuroglancer/ → generate precomputed

Discovery modes:
  - Default: scan /IMAGE_WRITE/VFB/i/ filesystem
  - --use-kb: query kb.virtualflybrain.org for live images from production
    datasets only (requires vfb-connect package)

Folder structure per image:
  /IMAGE_WRITE/VFB/i/{first4}/{last4}/{template_id}/
  ├── volume.swc            (existing)
  ├── volume.obj            (existing, auto-generated point cloud)
  ├── volume_man.obj        ← generated from SWC if missing
  ├── volume.nrrd           (existing)
  ├── volume.wlz            (existing)
  └── neuroglancer/         ← NEW precomputed mesh
      ├── info
      ├── mesh/
      │   ├── info
      │   ├── 1:0
      │   └── 1:0:1
      └── segment_properties/
          └── info

Neuroglancer URL:
  precomputed://https://www.virtualflybrain.org/data/VFB/i/{first4}/{last4}/{template_id}/neuroglancer

Usage:
  # Dry run — show what would be done
  python vfb_pipeline.py --dry-run

  # Process everything
  python vfb_pipeline.py

  # Process specific IDs
  python vfb_pipeline.py --ids VFB_00000001 VFB_00000002

  # Use KB to discover only live images from production datasets
  python vfb_pipeline.py --use-kb

  # Process from a file
  python vfb_pipeline.py --ids-file missing_objs.txt

  # Force regeneration even if outputs exist
  python vfb_pipeline.py --force
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import trimesh
from cloudvolume import CloudVolume
from cloudvolume.mesh import Mesh

# Import NRRD converter (same package)
try:
    from convert_nrrd import convert_nrrd as _convert_nrrd
except ImportError:
    _convert_nrrd = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_ROOT = "/IMAGE_WRITE"
VFB_DATA_DIR = os.path.join(IMAGE_ROOT, "VFB", "i")

# Default resolution for JRC2018Unisex template (nm)
DEFAULT_RESOLUTION = [518.9161, 518.9161, 1000.0]

# KB (Knowledge Base) endpoint for VFBconnect
KB_ENDPOINT = "http://kb.virtualflybrain.org"
KB_USER = "neo4j"
KB_PASSWORD = "vfb"

# Folder URL prefix in KB → local path mapping
KB_FOLDER_URL_PREFIX = "http://www.virtualflybrain.org/data/"

log = logging.getLogger("vfb_pipeline")


# ---------------------------------------------------------------------------
# KB Discovery: query kb.virtualflybrain.org for live image directories
# ---------------------------------------------------------------------------

def _get_neo4j_connect():
    """Import Neo4jConnect without triggering vfb_connect's slow __init__.

    The vfb_connect package's __init__.py instantiates a global VfbConnect
    object that connects to PDB and caches all terms, which blocks for a very
    long time.  We bypass this by stubbing the top-level package and importing
    the Neo4j module directly.
    """
    # If already imported (e.g. caller already set up the bypass), just use it
    if "vfb_connect.neo.neo4j_tools" in sys.modules:
        mod = sys.modules["vfb_connect.neo.neo4j_tools"]
        return mod.Neo4jConnect, mod.dict_cursor

    # Remove any existing stub so find_spec works on the real package
    saved = sys.modules.pop("vfb_connect", None)
    try:
        spec = importlib.util.find_spec("vfb_connect")
    finally:
        # Restore or re-stub immediately
        if saved is not None:
            sys.modules["vfb_connect"] = saved

    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("vfb_connect is not installed — pip install vfb-connect")

    pkg = types.ModuleType("vfb_connect")
    pkg.__path__ = list(spec.submodule_search_locations)
    sys.modules["vfb_connect"] = pkg
    from vfb_connect.neo.neo4j_tools import Neo4jConnect, dict_cursor  # noqa: E402
    return Neo4jConnect, dict_cursor


def iter_kb_image_dirs(image_root: str = IMAGE_ROOT):
    """Query KB for live image folders from production datasets.

    Yields (image_dir, vfb_id, template_id) tuples, matching the
    interface of iter_image_dirs().
    """
    Neo4jConnect, dict_cursor = _get_neo4j_connect()
    nc = Neo4jConnect(endpoint=KB_ENDPOINT, usr=KB_USER, pwd=KB_PASSWORD)

    query = """
        MATCH (c:Individual)-[:depicts]->(i:Individual)-[:has_source]->(ds:DataSet)
        MATCH (c)-[r:in_register_with]->(t:Template)
        WHERE ds.production[0] = true
          AND r.folder IS NOT NULL
          AND (r.block IS NULL OR NOT r.block[0] = 'Missing Image')
        RETURN DISTINCT i.short_form AS id, r.folder[0] AS folder
    """

    log.info("Querying KB at %s for live image directories...", KB_ENDPOINT)
    results = nc.commit_list([query])
    rows = dict_cursor(results)

    count = 0
    for row in rows:
        folder_url = row["folder"]
        vfb_id = row["id"]

        # Map URL to local path: replace URL prefix with IMAGE_ROOT
        if not folder_url.startswith(KB_FOLDER_URL_PREFIX):
            log.debug("  Skipping unexpected folder URL: %s", folder_url)
            continue

        rel_path = folder_url[len(KB_FOLDER_URL_PREFIX):]
        # Strip trailing slash
        rel_path = rel_path.rstrip("/")
        image_dir = os.path.join(image_root, rel_path)

        # Template ID is the last component of the path
        template_id = os.path.basename(image_dir)

        count += 1
        yield image_dir, vfb_id, template_id

    log.info("KB returned %d live image directories", count)


# ---------------------------------------------------------------------------
# Discovery: find image directories and classify them
# ---------------------------------------------------------------------------

def iter_image_dirs(vfb_data_dir: str = VFB_DATA_DIR):
    """Yield (image_dir, vfb_id) for every image directory under VFB/i/."""
    vfb_data = Path(vfb_data_dir)
    if not vfb_data.is_dir():
        log.error("VFB data directory not found: %s", vfb_data)
        return

    for first4 in sorted(vfb_data.iterdir()):
        if not first4.is_dir():
            continue
        for last4 in sorted(first4.iterdir()):
            if not last4.is_dir():
                continue
            # Each subdirectory under last4/ is a template alignment
            for template_dir in sorted(last4.iterdir()):
                if not template_dir.is_dir():
                    continue
                vfb_id = "VFB_" + first4.name + last4.name
                yield str(template_dir), vfb_id, template_dir.name


def find_image_dir(vfb_id: str, vfb_data_dir: str = VFB_DATA_DIR) -> list[str]:
    """Find all image directories for a given VFB ID (may be aligned to multiple templates)."""
    prefix = vfb_id.replace("VFB_", "")
    first4, last4 = prefix[:4], prefix[4:]
    parent = Path(vfb_data_dir) / first4 / last4
    if not parent.is_dir():
        return []
    return [str(d) for d in sorted(parent.iterdir()) if d.is_dir()]


def has_faces(obj_path: str) -> bool:
    """Check if an OBJ file has face definitions (real mesh vs point cloud)."""
    with open(obj_path) as f:
        for line in f:
            if line.startswith("f "):
                return True
    return False


def classify_dir(image_dir: str) -> dict:
    """Classify what files exist and what needs to be done for an image directory."""
    d = Path(image_dir)
    return {
        "has_swc": (d / "volume.swc").is_file(),
        "has_nrrd": (d / "volume.nrrd").is_file(),
        "has_obj_man": (d / "volume_man.obj").is_file(),
        "has_obj_man_faces": (
            has_faces(str(d / "volume_man.obj"))
            if (d / "volume_man.obj").is_file()
            else False
        ),
        "has_neuroglancer": (d / "neuroglancer" / "info").is_file(),
    }


# ---------------------------------------------------------------------------
# SWC → OBJ conversion (imported from convert_swc_to_mesh)
# ---------------------------------------------------------------------------

def parse_swc(filepath: str) -> dict:
    """Parse an SWC file into a dict of nodes."""
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
            try:
                radius = float(parts[5])
            except ValueError:
                radius = float("nan")
            nodes[node_id] = {
                "type": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "radius": radius,
                "parent": int(parts[6]),
            }
    return nodes


def create_tube_segment(p1, p2, r1, r2, n_sides=20):
    """Create a truncated cone (tube segment) between two points."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return None

    cylinder = trimesh.creation.cylinder(radius=1.0, height=length, sections=n_sides)
    verts = cylinder.vertices.copy()
    top_mask = verts[:, 2] > 0
    bottom_mask = ~top_mask
    verts[top_mask, 0] *= r2
    verts[top_mask, 1] *= r2
    verts[bottom_mask, 0] *= r1
    verts[bottom_mask, 1] *= r1
    cylinder.vertices = verts

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
    cylinder.apply_translation((p1 + p2) / 2.0)
    return cylinder


def swc_to_mesh(swc_path: str, min_radius: float = 0.2,
                tube_sides: int = 20) -> trimesh.Trimesh:
    """Convert SWC skeleton to a tubular trimesh."""
    nodes = parse_swc(swc_path)
    if not nodes:
        raise ValueError("No nodes found in SWC file: " + swc_path)

    meshes = []
    for node_id, node in nodes.items():
        parent_id = node["parent"]
        if parent_id < 0 or parent_id not in nodes:
            continue
        parent = nodes[parent_id]
        p1 = np.array([parent["x"], parent["y"], parent["z"]])
        p2 = np.array([node["x"], node["y"], node["z"]])
        r1 = parent["radius"] if not np.isnan(parent["radius"]) else min_radius
        r1 = max(r1, min_radius)
        r2 = node["radius"] if not np.isnan(node["radius"]) else min_radius
        r2 = max(r2, min_radius)
        tube = create_tube_segment(p1, p2, r1, r2, n_sides=tube_sides)
        if tube is not None:
            meshes.append(tube)

    # Spheres at branch points
    child_count = {}
    for node in nodes.values():
        pid = node["parent"]
        if pid > 0:
            child_count[pid] = child_count.get(pid, 0) + 1

    for nid, count in child_count.items():
        if count > 1 and nid in nodes:
            node = nodes[nid]
            r = node["radius"] if not np.isnan(node["radius"]) else min_radius
            r = max(r, min_radius)
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=r)
            sphere.apply_translation([node["x"], node["y"], node["z"]])
            meshes.append(sphere)

    if not meshes:
        raise ValueError("No tube segments could be generated from: " + swc_path)

    return trimesh.util.concatenate(meshes)


def generate_obj_from_swc(image_dir: str) -> str:
    """Generate volume_man.obj from volume.swc in the given directory.

    Returns the path to the created OBJ file.
    """
    swc_path = os.path.join(image_dir, "volume.swc")
    obj_path = os.path.join(image_dir, "volume_man.obj")

    log.info("  Generating OBJ from SWC: %s", swc_path)
    mesh = swc_to_mesh(swc_path)
    mesh.export(obj_path, file_type="obj")
    log.info("  Saved: %s (%d vertices, %d faces)",
             obj_path, len(mesh.vertices), len(mesh.faces))
    return obj_path


# ---------------------------------------------------------------------------
# OBJ → Neuroglancer precomputed
# ---------------------------------------------------------------------------

def write_precomputed(obj_path: str, output_dir: str,
                      resolution: list[float] = DEFAULT_RESOLUTION,
                      segment_id: int = 1, label: str | None = None):
    """Convert OBJ mesh to Neuroglancer precomputed format (uncompressed)."""

    mesh = trimesh.load(obj_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not load as triangle mesh: " + obj_path)

    log.info("  Writing precomputed: %s (%d verts, %d faces)",
             output_dir, len(mesh.vertices), len(mesh.faces))

    os.makedirs(output_dir, exist_ok=True)
    dest = "file://" + output_dir

    mesh_max = mesh.vertices.max(axis=0)
    mesh_min = mesh.vertices.min(axis=0)
    size = [
        int(np.ceil((mesh_max[i] - mesh_min[i]) / resolution[i])) + 2
        for i in range(3)
    ]

    info = {
        "data_type": "uint32",
        "num_channels": 1,
        "type": "segmentation",
        "mesh": "mesh",
        "segment_properties": "segment_properties",
        "scales": [{
            "chunk_sizes": [[64, 64, 64]],
            "encoding": "raw",
            "key": "0",
            "resolution": resolution,
            "size": size,
            "voxel_offset": [0, 0, 0],
        }],
    }

    vol = CloudVolume(dest, mip=0, info=info, compress=False)
    vol.commit_info()

    # Mesh directory
    mesh_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    with open(os.path.join(mesh_dir, "info"), "w") as f:
        json.dump({"@type": "neuroglancer_legacy_mesh"}, f)

    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    mesh_obj = Mesh(vertices, faces, segid=segment_id)
    vol.mesh.put(mesh_obj, compress=False)

    # Segment properties
    seg_dir = os.path.join(output_dir, "segment_properties")
    os.makedirs(seg_dir, exist_ok=True)
    display_label = label or "mesh"
    seg_info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [str(segment_id)],
            "properties": [
                {"id": "label", "type": "label", "values": [display_label]},
            ],
        },
    }
    with open(os.path.join(seg_dir, "info"), "w") as f:
        json.dump(seg_info, f, indent=2)

    log.info("  Precomputed written: %s", output_dir)


# ---------------------------------------------------------------------------
# Pipeline logic
# ---------------------------------------------------------------------------

def process_image(image_dir: str, vfb_id: str, template_id: str,
                  force: bool = False, dry_run: bool = False,
                  resolution: list[float] = DEFAULT_RESOLUTION) -> dict:
    """Process a single image directory. Returns a status dict."""

    status = classify_dir(image_dir)
    result = {
        "vfb_id": vfb_id,
        "template_id": template_id,
        "image_dir": image_dir,
        "obj_generated": False,
        "precomputed_generated": False,
        "skipped": False,
        "error": None,
    }

    needs_obj = status["has_swc"] and (not status["has_obj_man"] or
                                        (status["has_obj_man"] and not status["has_obj_man_faces"]))
    needs_precomputed = not status["has_neuroglancer"]
    has_usable_obj = status["has_obj_man"] and status["has_obj_man_faces"]

    if force:
        if status["has_swc"]:
            needs_obj = True
        needs_precomputed = True

    if not needs_obj and not needs_precomputed:
        result["skipped"] = True
        log.debug("  [%s] Already complete, skipping", vfb_id)
        return result

    obj_status = ("mesh" if status["has_obj_man_faces"]
                  else "no-faces" if status["has_obj_man"]
                  else "missing")
    log.info("[%s] %s (swc=%s, nrrd=%s, obj_man=%s, ng=%s)",
             vfb_id, image_dir,
             status["has_swc"], status["has_nrrd"], obj_status,
             status["has_neuroglancer"])

    if dry_run:
        if needs_obj:
            log.info("  Would generate: volume_man.obj from volume.swc")
        if needs_precomputed and (has_usable_obj or needs_obj):
            log.info("  Would generate: neuroglancer/ from volume_man.obj")
        if needs_precomputed and status["has_nrrd"] and not has_usable_obj and not needs_obj:
            log.info("  Would generate: neuroglancer/ (including 0/ chunks) from volume.nrrd")
        return result

    # Step 1: Generate OBJ from SWC if needed
    if needs_obj:
        try:
            generate_obj_from_swc(image_dir)
            result["obj_generated"] = True
            has_usable_obj = True
        except Exception as e:
            result["error"] = f"OBJ generation failed: {e}"
            log.error("  ERROR generating OBJ: %s", e)
            return result

    # Step 2a: Generate precomputed from OBJ if available
    if needs_precomputed and has_usable_obj:
        try:
            obj_path = os.path.join(image_dir, "volume_man.obj")
            ng_dir = os.path.join(image_dir, "neuroglancer")
            write_precomputed(obj_path, ng_dir, resolution=resolution)
            result["precomputed_generated"] = True
        except Exception as e:
            result["error"] = f"Precomputed generation failed: {e}"
            log.error("  ERROR generating precomputed: %s", e)

    # Step 2b: Generate precomputed (with 0/ volume chunks) from NRRD when no OBJ is available
    elif needs_precomputed and status["has_nrrd"]:
        if _convert_nrrd is None:
            result["error"] = "convert_nrrd module not available"
            log.error("  ERROR: convert_nrrd module could not be imported")
        else:
            try:
                nrrd_path = os.path.join(image_dir, "volume.nrrd")
                log.info("  Generating neuroglancer/ (with 0/ chunks) from NRRD: %s", nrrd_path)
                # convert_nrrd writes to {output_dir}/{dataset_name}/, so passing
                # image_dir + "neuroglancer" produces image_dir/neuroglancer/ with
                # the 0/ chunk directory, mesh/, segment_properties/ all inside it.
                _convert_nrrd(
                    nrrd_path=nrrd_path,
                    output_dir=image_dir,
                    dataset_name="neuroglancer",
                    verbose=log.isEnabledFor(logging.DEBUG),
                )
                result["precomputed_generated"] = True
            except Exception as e:
                result["error"] = f"NRRD precomputed generation failed: {e}"
                log.error("  ERROR generating precomputed from NRRD: %s", e)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VFB Jenkins Pipeline: generate volume_man.obj and neuroglancer precomputed data"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--ids", nargs="+",
                             help="Specific VFB IDs to process")
    input_group.add_argument("--ids-file",
                             help="File with VFB IDs (one per line)")

    parser.add_argument("--image-root", default=IMAGE_ROOT,
                        help="Root of VFB image data (default: /IMAGE_WRITE)")
    parser.add_argument("--use-kb", action="store_true",
                        help="Query kb.virtualflybrain.org for live images instead of scanning the filesystem")
    parser.add_argument("--resolution", type=float, nargs=3,
                        default=DEFAULT_RESOLUTION,
                        help="Voxel resolution in nm [x y z]")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if output already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    vfb_data_dir = os.path.join(args.image_root, "VFB", "i")

    if not os.path.isdir(vfb_data_dir):
        log.error("VFB data directory not found: %s", vfb_data_dir)
        log.error("Set --image-root to the directory that contains VFB/i/")
        sys.exit(1)

    # Collect targets
    if args.ids:
        targets = []
        for vfb_id in args.ids:
            for d in find_image_dir(vfb_id, vfb_data_dir):
                template_id = os.path.basename(d)
                targets.append((d, vfb_id, template_id))
            if not find_image_dir(vfb_id, vfb_data_dir):
                log.warning("No directories found for %s", vfb_id)
    elif args.ids_file:
        targets = []
        with open(args.ids_file) as f:
            for line in f:
                vfb_id = line.strip()
                if not vfb_id or vfb_id.startswith("#"):
                    continue
                for d in find_image_dir(vfb_id, vfb_data_dir):
                    template_id = os.path.basename(d)
                    targets.append((d, vfb_id, template_id))
    elif args.use_kb:
        log.info("Using KB to discover live image directories...")
        targets = list(iter_kb_image_dirs(args.image_root))
        log.info("Found %d live image directories from KB", len(targets))
    else:
        # Scan all
        log.info("Scanning %s for image directories...", vfb_data_dir)
        targets = list(iter_image_dirs(vfb_data_dir))
        log.info("Found %d image directories", len(targets))

    if not targets:
        log.info("No image directories to process")
        return

    # Process
    start = time.time()
    stats = {
        "total": len(targets),
        "obj_generated": 0,
        "precomputed_generated": 0,
        "skipped": 0,
        "errors": 0,
    }

    for i, (image_dir, vfb_id, template_id) in enumerate(targets, 1):
        if i % 500 == 0 or args.verbose:
            log.info("Progress: %d / %d", i, stats["total"])

        result = process_image(
            image_dir, vfb_id, template_id,
            force=args.force, dry_run=args.dry_run,
            resolution=args.resolution,
        )

        if result["obj_generated"]:
            stats["obj_generated"] += 1
        if result["precomputed_generated"]:
            stats["precomputed_generated"] += 1
        if result["skipped"]:
            stats["skipped"] += 1
        if result["error"]:
            stats["errors"] += 1

    elapsed = time.time() - start

    # Summary
    log.info("=" * 60)
    log.info("Pipeline Summary%s", " (DRY RUN)" if args.dry_run else "")
    log.info("=" * 60)
    log.info("  Total image dirs:      %d", stats["total"])
    log.info("  OBJ generated:         %d", stats["obj_generated"])
    log.info("  Precomputed generated: %d", stats["precomputed_generated"])
    log.info("  Skipped (up to date):  %d", stats["skipped"])
    log.info("  Errors:                %d", stats["errors"])
    log.info("  Elapsed:               %.1fs", elapsed)

    if stats["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
