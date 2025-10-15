import os
import shutil
import zipfile
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import json
import torch
import cv2

# Import all necessary functions from your tool scripts
from tools.rename_resize import process_videos as rename_resize_videos
from tools.clip_video import clip_video
from tools.keyframe_selector import KeyframeSelector
from rfdetr import RFDETRMedium
from tools.create_proposals_from_tracks import generate_proposals_from_tracks
from tools.proposals_to_cvat import generate_xml_for_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(zip_file_path: str, output_dir: str, batch_name: str):
    """
    Runs the full, integrated AVA-Kinetics preprocessing pipeline.
    """
    base_output_path = Path(output_dir)
    work_dir = base_output_path / "temp_processing"

    # Define directories
    raw_video_dir = work_dir / "0_raw_videos"
    resized_dir = work_dir / "1_resized_videos"
    clipped_dir = work_dir / "2_clipped_videos"
    json_dir = work_dir / "3_tracking_json"

    batch_dir = base_output_path / batch_name
    keyframes_dir = batch_dir / "keyframes"

    for d in [work_dir, raw_video_dir, resized_dir, clipped_dir, json_dir,
              batch_dir, keyframes_dir, base_output_path]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("✅ Directory structure created successfully.")

    manifest_data = {}

    # Initialize models once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    detection_model = RFDETRMedium(device=device)
    keyframe_selector = KeyframeSelector(detection_model=detection_model, device=device, person_class_id=1)

    try:
        # --- Stage 1-3: Unzip, Resize, Clip ---
        logger.info("[Stage 1/7] Unzipping Master File...")
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            zf.extractall(raw_video_dir)

        logger.info("[Stage 2/7] Renaming & Resizing...")
        rename_resize_videos(str(raw_video_dir), str(resized_dir))

        logger.info("[Stage 3/7] Clipping Videos...")
        clip_video(str(resized_dir), str(clipped_dir))

        # --- Stage 4: Intelligent Keyframe Selection ---
        logger.info("[Stage 4/7] Selecting Keyframes & Generating Proposals...")
        all_clips_to_process = list(Path(clipped_dir).rglob("*.mp4"))
        for clip_path in tqdm(all_clips_to_process, desc="  -> Selecting keyframes"):
            # (The logic for this stage remains the same)
            clip_stem = clip_path.stem
            result = keyframe_selector.select_best_keyframe(str(clip_path))
            if result is None: continue
            best_frame_img, best_frame_idx, detections = result
            keyframe_name = f"{clip_stem}_frame_{best_frame_idx:04d}.jpg"
            cv2.imwrite(str(keyframes_dir / keyframe_name), best_frame_img)
            json_output_path = json_dir / f"{clip_stem}.json"
            formatted_detections = [
                {"video_id": clip_stem, "frame": keyframe_name, "track_id": d["track_id"], "bbox": d["bbox"]} for d in
                detections]
            with open(json_output_path, "w") as f:
                json.dump(formatted_detections, f, indent=2)
            manifest_data[keyframe_name] = {"source_video": clip_path.name, "source_frame": int(best_frame_idx)}

        # --- Stage 5: Create Manifest and Package Keyframes ---
        logger.info("[Stage 5/7] Creating Manifest and Packaging Keyframes...")
        manifest_path = batch_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        batch_zip_path = base_output_path / f"{batch_name}_keyframes.zip"
        with zipfile.ZipFile(batch_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for frame_file in keyframes_dir.glob("*.jpg"): zf.write(frame_file, arcname=frame_file.name)
        logger.info(f"✓ Keyframes and manifest created for batch '{batch_name}'.")

        # --- NEW STAGE 6: Aggregate Proposals ---
        logger.info("[Stage 6/7] Aggregating proposals...")
        proposals_pkl_path = work_dir / "proposals.pkl"
        generate_proposals_from_tracks(str(json_dir), str(proposals_pkl_path))

        # --- NEW STAGE 7: Generate Final CVAT XML ---
        logger.info("[Stage 7/7] Generating final CVAT XML...")
        final_xml_path = base_output_path / f"{batch_name}_annotations.xml"
        generate_xml_for_batch(
            batch_name=batch_name,
            pickle_path=str(proposals_pkl_path),
            keyframes_dir=str(keyframes_dir),
            output_xml_path=str(final_xml_path)
        )

        logger.info(f"\n🎉🎉🎉 Pipeline complete! Final outputs are in: {base_output_path}")

    finally:
        if work_dir.exists():
            logger.info(f"Cleaning up temporary directory: {work_dir}")
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full integrated AVA-Kinetics pre-processing pipeline.")
    PROJECT_ROOT = Path(__file__).resolve().parent
    parser.add_argument("--zip_file_name", required=True, help="Name of the master ZIP file with raw videos.")
    parser.add_argument("--batch_name", required=True, help="A unique name for this processing batch.")
    args = parser.parse_args()
    input_zip = PROJECT_ROOT / "uploads" / args.zip_file_name
    output_path = PROJECT_ROOT / "outputs"
    if not input_zip.exists():
        logger.error(f"Input file not found: {input_zip}")
    else:
        run_pipeline(str(input_zip), str(output_path), args.batch_name)