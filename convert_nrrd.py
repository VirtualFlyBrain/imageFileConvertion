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
        _generate_external_mesh(arr, dest_local, vol, voxel_size, dust_threshold, verbose)

    return dest_local


def _generate_external_mesh(arr, dest_local, vol, voxel_size, dust_threshold, verbose):
    """Generate a single mesh from the outer boundary of all non-zero voxels."""

    # Setup mesh directory
    mesh_dir = os.path.join(dest_local, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    with open(os.path.join(mesh_dir, "info"), "w") as f:
        json.dump({"@type": "neuroglancer_legacy_mesh"}, f, indent=2)

    # All non-zero segment IDs
    all_segments = np.unique(arr)
    all_segments = all_segments[all_segments > 0]

    if len(all_segments) == 0:
        if verbose:
            print("  No non-zero segments found, skipping mesh generation")
        _write_segment_properties(dest_local, [], [])
        return

    # Create binary mask of all non-zero voxels for a single external mesh
    binary_mask = (arr > 0).astype(np.float32)
    voxel_count = int(np.sum(binary_mask > 0))

    if voxel_count < dust_threshold:
        if verbose:
            print(f"  Skipping mesh: only {voxel_count} non-zero voxels (< {dust_threshold})")
        _write_segment_properties(dest_local, [], [])
        return

    if verbose:
        print(f"  Generating external boundary mesh from {voxel_count} non-zero voxels "
              f"({len(all_segments)} segment(s))")

    try:
        vertices, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5, allow_degenerate=False)
    except (ValueError, RuntimeError) as e:
        if verbose:
            print(f"  Mesh generation failed: {e}")
        _write_segment_properties(dest_local, [], [])
        return

    if len(vertices) == 0 or len(faces) == 0:
        if verbose:
            print("  No mesh geometry produced")
        _write_segment_properties(dest_local, [], [])
        return

    # Scale vertices by voxel resolution
    vertices = vertices.astype(np.float32)
    vertices[:, 0] *= voxel_size[0]
    vertices[:, 1] *= voxel_size[1]
    vertices[:, 2] *= voxel_size[2]

    if verbose:
        print(f"    External mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Write the same external mesh for every segment ID so it is visible
    # regardless of which segment is selected in Neuroglancer
    for seg_id in all_segments:
        mesh_obj = Mesh(vertices, faces.astype(np.uint32), segid=int(seg_id))
        vol.mesh.put(mesh_obj, compress=True)

    seg_ids = [str(s) for s in all_segments]
    seg_labels = [f"Segment {s}" for s in all_segments]
    _write_segment_properties(dest_local, seg_ids, seg_labels)

    if verbose:
        print(f"  Wrote external mesh for {len(all_segments)} segment(s)")


def _write_segment_properties(dest_local, seg_ids, seg_labels):
    """Write segment_properties/info for Neuroglancer."""
    seg_dir = os.path.join(dest_local, "segment_properties")
    os.makedirs(seg_dir, exist_ok=True)
    seg_info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": seg_ids,
            "properties": [
                {"id": "label", "type": "label", "values": seg_labels},
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
