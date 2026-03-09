#!/usr/bin/env python3
"""
Category D: Convert NRRD volume files to Neuroglancer precomputed format
with meshes generated via marching cubes.

For images that only have volumetric NRRD data (expression patterns,
individual painted domain volumes, confocal images), this script:
  1. Reads the NRRD file
  2. Converts to precomputed segmentation format
  3. Generates meshes via marching cubes on thresholded/labeled regions
  4. Removes dust (small disconnected components)

Based on the approach from MetaCell/virtual-fly-brain PR #207.
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
from cloudvolume.lib import Bbox
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
            return [abs(float(np.linalg.norm(d))) if d is not None else 1.0 for d in dirs][::-1]
        except Exception:
            pass
    if "spacings" in header:
        try:
            return list(map(float, header["spacings"]))[::-1]
        except Exception:
            pass
    return [1.0, 1.0, 1.0]


def convert_nrrd(nrrd_path: str, output_dir: str, dataset_name: str,
                 threshold: float | None = None, dust_threshold: int = 100,
                 merge_segments: bool = False, verbose: bool = True):
    """Convert an NRRD volume to precomputed format with mesh generation."""

    if verbose:
        print(f"Reading NRRD: {nrrd_path}")

    data, header = nrrd.read(nrrd_path)
    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D volume, got ndim={data.ndim}")

    # Transpose from ZYX (NRRD) to XYZ (Neuroglancer)
    arr = np.transpose(data, (2, 1, 0)).copy()
    voxel_size = detect_spacing(header)

    if verbose:
        print(f"  Shape (XYZ): {arr.shape}")
        print(f"  Voxel size:  {voxel_size}")
        print(f"  Dtype:       {arr.dtype}")
        print(f"  Value range: [{arr.min()}, {arr.max()}]")

    # For floating-point data (confocal images), threshold to create a binary mask
    # then label connected components
    is_segmentation = np.issubdtype(arr.dtype, np.integer) and not merge_segments

    if not is_segmentation:
        if threshold is None:
            # Auto-threshold at mean + 1 std for intensity data
            nonzero = arr[arr > 0]
            if len(nonzero) > 0:
                threshold = float(np.mean(nonzero))
            else:
                threshold = 0.5

        if verbose:
            print(f"  Thresholding at {threshold}")

        binary = arr > threshold

        if merge_segments:
            # Create a single segment from all non-zero voxels
            arr = binary.astype(np.uint32)
        else:
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)
            arr = labeled.astype(np.uint32)
            if verbose:
                print(f"  Found {num_features} connected components")
    else:
        arr = arr.astype(np.uint32) if arr.dtype != np.uint32 else arr

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
            "voxel_offset": [0, 0, 0],
        }],
        "type": "segmentation",
        "mesh": "mesh",
        "segment_properties": "segment_properties",
    }

    vol = CloudVolume(dest, mip=0, info=info, compress=False)
    vol.commit_info()
    vol[:, :, :] = arr

    if verbose:
        print(f"  Wrote precomputed volume to {dest_local}")

    # Setup mesh
    mesh_dir = os.path.join(dest_local, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    with open(os.path.join(mesh_dir, "info"), "w") as f:
        json.dump({"@type": "neuroglancer_legacy_mesh"}, f, indent=2)

    # Find segments and generate meshes
    all_segments = np.unique(arr)
    all_segments = all_segments[all_segments > 0]

    if verbose:
        print(f"  Found {len(all_segments)} non-zero segments")

    resolution = voxel_size
    seg_ids = []
    seg_labels = []

    generated = 0
    for seg_id in all_segments:
        mask = arr == seg_id
        voxel_count = int(np.sum(mask))
        if voxel_count < dust_threshold:
            continue

        try:
            vertices, faces, _, _ = measure.marching_cubes(mask, level=0.5, allow_degenerate=False)
        except (ValueError, RuntimeError):
            continue

        if len(vertices) == 0 or len(faces) == 0:
            continue

        vertices = vertices.astype(np.float32)
        vertices[:, 0] *= resolution[0]
        vertices[:, 1] *= resolution[1]
        vertices[:, 2] *= resolution[2]

        mesh_obj = Mesh(vertices, faces.astype(np.uint32), segid=int(seg_id))
        vol.mesh.put(mesh_obj, compress=True)

        seg_ids.append(str(seg_id))
        seg_labels.append(f"Segment {seg_id} ({voxel_count} voxels)")
        generated += 1

        if verbose and generated <= 10:
            print(f"    Segment {seg_id}: {len(vertices)} verts, {len(faces)} faces ({voxel_count} voxels)")

    # Segment properties
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

    if verbose:
        print(f"  Generated {generated} meshes out of {len(all_segments)} segments")

    return dest_local


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
                        help="Intensity threshold for non-segmentation data (default: auto)")
    parser.add_argument("--dust-threshold", type=int, default=100,
                        help="Minimum voxel count for mesh generation (default: 100)")
    parser.add_argument("--merge-segments", action="store_true",
                        help="Merge all non-zero voxels into a single segment")
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
            verbose=args.verbose,
        )
    finally:
        if tmp_nrrd:
            os.unlink(nrrd_path)

    print(f"Done. Output at: {output_dir}/{dataset_name}")


if __name__ == "__main__":
    main()
