#!/usr/bin/env python3
"""
Convert NRRD volume files to Neuroglancer precomputed format
with a mesh generated from the external volume boundary.

This script:
  1. Reads the NRRD file
  2. Converts to precomputed format (segmentation or image layer)
  3. For segmentation data, generates a single mesh from the outer boundary
     of all non-zero voxels via marching cubes

Volume conversion approach matches MetaCell/virtual-fly-brain converter.py:
  - Detects voxel spacing and origin from NRRD header
  - Writes data as-is (no thresholding/relabeling for integer data)
  - Supports gzip compression
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import nrrd
import requests
from cloudvolume import CloudVolume
from cloudvolume.mesh import Mesh

try:
    from skimage import measure
except ImportError:
    print("ERROR: scikit-image required. Install with: pip install scikit-image")
    sys.exit(1)


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


def detect_spacing(header: dict) -> list[float]:
    if "space directions" in header and header["space directions"] is not None:
        try:
            dirs = header["space directions"]
            return [float(np.linalg.norm(d)) if d is not None else 1.0 for d in dirs][::-1]
        except Exception:
            pass
    if "spacings" in header:
        try:
            return list(map(float, header["spacings"]))[::-1]
        except Exception:
            pass
    return [1.0, 1.0, 1.0]


def detect_origin(header: dict) -> list[float]:
    """Extract space origin from NRRD header and convert to XYZ order."""
    if "space origin" in header and header["space origin"] is not None:
        try:
            origin = header["space origin"]
            return [float(x) for x in origin[::-1]]
        except Exception:
            pass
    return [0.0, 0.0, 0.0]


def convert_nrrd(nrrd_path: str, output_dir: str, dataset_name: str,
                 threshold: float | None = None, dust_threshold: int = 100,
                 merge_segments: bool = False,
                 min_intensity: int | None = None,
                 max_intensity: int | None = None,
                 compress: bool = False,
                 verbose: bool = True):
    """Convert an NRRD volume to precomputed format with external-boundary mesh.

    The volume is written as-is (matching converter.py behaviour).
    A single mesh is generated from the outer boundary of all non-zero voxels
    rather than one mesh per segmented region.
    """

    if verbose:
        print(f"Reading NRRD: {nrrd_path}")

    data, header = nrrd.read(nrrd_path)
    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D volume, got ndim={data.ndim}")

    # Transpose from ZYX (NRRD) to XYZ (Neuroglancer)
    arr = np.transpose(data, (2, 1, 0)).copy()
    voxel_size = detect_spacing(header)
    voxel_offset = detect_origin(header)

    if verbose:
        print(f"  Shape (XYZ): {arr.shape}")
        print(f"  Voxel size:  {voxel_size}")
        print(f"  Voxel offset: {voxel_offset}")
        print(f"  Dtype:       {arr.dtype}")
        print(f"  Value range: [{arr.min()}, {arr.max()}]")

    # Apply intensity filtering if specified (for segmentation data)
    if np.issubdtype(arr.dtype, np.integer):
        if min_intensity is not None or max_intensity is not None:
            original_segments = len(np.unique(arr[arr > 0]))
            if min_intensity is not None:
                arr[arr < min_intensity] = 0
            if max_intensity is not None:
                arr[arr > max_intensity] = 0
            filtered_segments = len(np.unique(arr[arr > 0]))
            if verbose:
                print(f"  Intensity filter: {original_segments} segments -> {filtered_segments} segments")
                print(f"  Range: [{min_intensity or 'any'}, {max_intensity or 'any'}]")

    # Determine layer type
    is_segmentation = np.issubdtype(arr.dtype, np.integer)
    layer_type = "segmentation" if is_segmentation else "image"
    dtype_str = str(np.dtype(arr.dtype).name)

    dest_local = os.path.join(output_dir, dataset_name)
    os.makedirs(dest_local, exist_ok=True)
    dest = f"file://{dest_local}"

    # Write precomputed volume
    info = {
        "data_type": dtype_str,
        "num_channels": 1,
        "scales": [{
            "chunk_sizes": [[64, 64, 64]],
            "encoding": "raw",
            "key": "0",
            "resolution": voxel_size,
            "size": list(arr.shape),
            "voxel_offset": voxel_offset,
        }],
        "type": layer_type,
    }

    if is_segmentation:
        info["mesh"] = "mesh"
        info["segment_properties"] = "segment_properties"

    vol = CloudVolume(dest, mip=0, info=info, compress=compress)
    vol.commit_info()
    vol[:, :, :] = arr

    if verbose:
        print(f"  Wrote precomputed volume to {dest_local}")
        print(f"  Layer type: {layer_type}")
        print(f"  Compression: {'gzip' if compress else 'none'}")

    # Generate mesh from the external boundary of all non-zero voxels
    if is_segmentation:
        _generate_external_mesh(arr, dest_local, vol, voxel_size, voxel_offset, dust_threshold, compress, verbose)

    return dest_local


def _setup_mesh_metadata(dest_local, vol, verbose):
    """Ensure mesh metadata is properly configured (matches meshes_generator.py)."""
    needs_update = False
    if "mesh" not in vol.info or vol.info["mesh"] is None:
        vol.info["mesh"] = "mesh"
        needs_update = True
    if "segment_properties" not in vol.info or vol.info["segment_properties"] is None:
        vol.info["segment_properties"] = "segment_properties"
        needs_update = True
    if needs_update:
        vol.commit_info()

    mesh_dir = os.path.join(dest_local, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    mesh_info = {
        "@type": "neuroglancer_legacy_mesh",
        "mip": 0,
        "vertex_quantization_bits": 10,
        "lod_scale_multiplier": 1.0,
    }
    with open(os.path.join(mesh_dir, "info"), "w") as f:
        json.dump(mesh_info, f, indent=2)

    if verbose:
        print("  Mesh metadata configured (legacy format)")


def _generate_external_mesh(arr, dest_local, vol, voxel_size, voxel_offset, dust_threshold, compress, verbose):
    """Generate a single mesh from the outer boundary of all non-zero voxels.

    Uses the same approach as meshes_generator.py with merge_segments=True:
    a single mesh with segment ID 1, vertices transformed to physical coordinates
    including voxel_offset.
    """

    _setup_mesh_metadata(dest_local, vol, verbose)

    # Create binary mask of all non-zero voxels
    mask = arr > 0
    voxel_count = int(np.sum(mask))

    if voxel_count == 0:
        if verbose:
            print("  No non-zero voxels found, skipping mesh generation")
        _write_segment_properties(dest_local, [], [], [])
        return

    if voxel_count < dust_threshold:
        if verbose:
            print(f"  Skipping mesh: only {voxel_count} non-zero voxels (< {dust_threshold})")
        _write_segment_properties(dest_local, [], [], [])
        return

    if verbose:
        print(f"  Creating merged mesh from {voxel_count} voxels...")

    try:
        vertices, faces, _, _ = measure.marching_cubes(mask, level=0.5, allow_degenerate=False)
    except (ValueError, RuntimeError) as e:
        if verbose:
            print(f"  Failed to generate merged mesh: {e}")
        _write_segment_properties(dest_local, [], [], [])
        return

    if len(vertices) == 0 or len(faces) == 0:
        if verbose:
            print("  Merged mesh has no geometry")
        _write_segment_properties(dest_local, [], [], [])
        return

    # Transform vertices to physical coordinates (resolution + offset)
    vertices = vertices.astype(np.float32)
    vertices[:, 0] = vertices[:, 0] * voxel_size[0] + voxel_offset[0]
    vertices[:, 1] = vertices[:, 1] * voxel_size[1] + voxel_offset[1]
    vertices[:, 2] = vertices[:, 2] * voxel_size[2] + voxel_offset[2]
    faces = faces.astype(np.uint32)

    if verbose:
        print(f"  Merged mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Use the lowest non-zero segment ID present in the volume for the mesh,
    # so it is visible when intensity filtering removes low segment IDs.
    all_segments = np.unique(arr)
    all_segments = all_segments[all_segments > 0]
    mesh_seg_id = int(all_segments[0])

    mesh_obj = Mesh(vertices, faces, segid=mesh_seg_id)
    vol.mesh.put(mesh_obj, compress=compress)

    _write_segment_properties(
        dest_local, [mesh_seg_id],
        [f"Segment {mesh_seg_id}"],
        ["Merged external boundary"],
    )

    if verbose:
        print(f"  Wrote merged external boundary mesh (segment ID {mesh_seg_id})")


def _write_segment_properties(dest_local, seg_ids, seg_labels, seg_descriptions):
    """Write segment_properties/info for Neuroglancer."""
    seg_dir = os.path.join(dest_local, "segment_properties")
    os.makedirs(seg_dir, exist_ok=True)
    ids = [str(s) for s in seg_ids]
    seg_info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": ids,
            "properties": [
                {"id": "label", "type": "label", "values": seg_labels},
                {"id": "description", "type": "description", "values": seg_descriptions},
            ],
        },
    }
    with open(os.path.join(seg_dir, "info"), "w") as f:
        json.dump(seg_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NRRD volumes to Neuroglancer precomputed format with meshes"
    )
    parser.add_argument("--input-nrrd", default=None,
                        help="Path to local NRRD file")
    parser.add_argument("--vfb-id", default=None,
                        help="VFB image ID to download NRRD from server")
    parser.add_argument("--template-id", default="VFB_00101567",
                        help="Template ID for URL construction (default: JRC2018Unisex)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for precomputed datasets")
    parser.add_argument("--dataset-name", default=None,
                        help="Name for the output dataset (default: derived from input)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="(unused, kept for CLI compatibility)")
    parser.add_argument("--dust-threshold", type=int, default=100,
                        help="Minimum voxel count for mesh generation (default: 100)")
    parser.add_argument("--merge-segments", action="store_true",
                        help="(unused, kept for CLI compatibility)")
    parser.add_argument("--min-intensity", type=int, default=None,
                        help="Minimum segment ID/intensity to keep (values below will be set to 0)")
    parser.add_argument("--max-intensity", type=int, default=None,
                        help="Maximum segment ID/intensity to keep (values above will be set to 0)")
    parser.add_argument("--compress", action="store_true",
                        help="Enable gzip compression (default: uncompressed raw chunks)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.input_nrrd and not args.vfb_id:
        parser.error("Must provide either --input-nrrd or --vfb-id")

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    nrrd_path = args.input_nrrd
    dataset_name = args.dataset_name
    tmp_nrrd = None

    if not nrrd_path:
        url = vfb_image_url(args.vfb_id, args.template_id, "volume.nrrd")
        if args.verbose:
            print(f"Downloading NRRD: {url}")
        tmp_nrrd = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False)
        nrrd_path = tmp_nrrd.name
        tmp_nrrd.close()
        download_file(url, nrrd_path)
        dataset_name = dataset_name or args.vfb_id

    if not dataset_name:
        dataset_name = os.path.splitext(os.path.basename(nrrd_path))[0]

    try:
        convert_nrrd(
            nrrd_path, output_dir, dataset_name,
            threshold=args.threshold,
            dust_threshold=args.dust_threshold,
            merge_segments=args.merge_segments,
            min_intensity=args.min_intensity,
            max_intensity=args.max_intensity,
            compress=args.compress,
            verbose=args.verbose,
        )
    finally:
        if tmp_nrrd:
            os.unlink(nrrd_path)

    print(f"Done. Output at: {output_dir}/{dataset_name}")


if __name__ == "__main__":
    main()
