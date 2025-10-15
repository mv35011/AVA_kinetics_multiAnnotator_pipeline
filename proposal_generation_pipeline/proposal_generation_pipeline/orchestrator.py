import os
import shutil
import zipfile
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

# Import the functions/classes from your tool scripts
from tools.rename_resize import process_videos as rename_resize_videos
from tools.clip_video import clip_video
from tools.person_tracker import PersonTracker
from tools.create_proposals_from_tracks import generate_proposals_from_tracks
from tools.proposals_to_cvat import generate_all_xmls
from tools.s3_uploader import S3Uploader  # New import for our S3 service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(zip_file_path: str, output_dir: str, s3_bucket: str):
    """
    Runs the entire pre-processing pipeline from a master ZIP of raw videos
    to final packaged outputs, with stage-wise progress tracking and an S3 upload.
    """
    base_output_path = Path(output_dir)
    work_dir = base_output_path / "temp_processing"

    total_stages = 10  # Updated number of stages
    with tqdm(total=total_stages, desc="Initializing Pipeline", bar_format="{l_bar}{bar:10}{r_bar}") as pbar:
        try:
            # --- 1. Define and Create Directory Structure (MODIFIED) ---
            raw_video_dir = work_dir / "0_raw_videos"
            resized_dir = work_dir / "1_resized_videos"
            clipped_dir = work_dir / "2_clipped_videos"
            frames_dir = work_dir / "3_tracking_frames"  # Temporary dir for jpg frames
            json_dir = work_dir / "4_tracking_json"
            temp_xml_dir = work_dir / "5_cvat_xml"  # Temporary dir for XMLs

            # Final output directories that will be uploaded
            final_frames_dir = base_output_path / "frames"
            final_xml_dir = base_output_path / "annotations"

            for d in [work_dir, raw_video_dir, resized_dir, clipped_dir, frames_dir, json_dir,
                      temp_xml_dir, final_frames_dir, final_xml_dir, base_output_path]:
                d.mkdir(parents=True, exist_ok=True)
            logger.info("✅ Directory structure created successfully.")

            # --- Stage 1: Unzip Master Video File ---
            pbar.set_description("[Stage 1/10] Unzipping Master File")
            with zipfile.ZipFile(zip_file_path, 'r') as zf:
                zf.extractall(raw_video_dir)
            pbar.update(1)

            # --- Stage 2: Rename & Resize ---
            pbar.set_description("[Stage 2/10] Renaming & Resizing")
            rename_resize_videos(str(raw_video_dir), str(resized_dir))
            pbar.update(1)

            # --- Stage 3: Clip Videos ---
            pbar.set_description("[Stage 3/10] Clipping Videos")
            clip_video(str(resized_dir), str(clipped_dir))
            pbar.update(1)

            # --- Stage 4: Track & Extract Frames ---
            pbar.set_description("[Stage 4/10] Tracking & Extracting")
            all_clips_to_process = list(Path(clipped_dir).rglob("*.mp4"))

            for clip_path in tqdm(all_clips_to_process, desc="  -> Tracking individual clips", leave=False):
                clip_stem = clip_path.stem
                clip_frame_output_dir = frames_dir / clip_stem
                clip_frame_output_dir.mkdir(exist_ok=True)

                tracker = PersonTracker(video_id=clip_stem, conf=0.45, person_class_id=1)
                tracker.process_video(
                    video_path=str(clip_path),
                    output_json_dir=str(json_dir),
                    output_frame_dir=str(clip_frame_output_dir),
                    output_fps=1
                )
            pbar.update(1)

            # --- Stage 5: Generate Dense Proposals ---
            pbar.set_description("[Stage 5/10] Generating Proposals")
            proposals_pkl_path = base_output_path / "dense_proposals.pkl"
            generate_proposals_from_tracks(str(json_dir), str(proposals_pkl_path))
            pbar.update(1)

            # --- Stage 6: Package Tracking JSONs ---
            pbar.set_description("[Stage 6/10] Packaging JSONs")
            json_zip_path = base_output_path / "tracking_jsons.zip"
            with zipfile.ZipFile(json_zip_path, 'w', zipfile.ZIP_DEFLATED) as json_zip:
                for json_file in Path(json_dir).glob("*.json"):
                    json_zip.write(json_file, arcname=json_file.name)
            pbar.update(1)

            # --- Stage 7: Package Frames into Individual Zips (REPLACED) ---
            pbar.set_description("[Stage 7/10] Zipping Clip Frames")
            for clip_folder in tqdm(Path(frames_dir).iterdir(), desc="  -> Zipping individual clips", leave=False):
                if clip_folder.is_dir():
                    clip_zip_path = final_frames_dir / f"{clip_folder.name}.zip"
                    with zipfile.ZipFile(clip_zip_path, 'w', zipfile.ZIP_DEFLATED) as clip_zip:
                        for frame_file in clip_folder.glob("*.jpg"):
                            clip_zip.write(frame_file, arcname=frame_file.name)
            pbar.update(1)

            # --- Stage 8: Generate XML Preannotations ---
            pbar.set_description("[Stage 8/10] Generating XMLs")
            generate_all_xmls(
                pickle_path=str(proposals_pkl_path),
                frame_dir=str(frames_dir),
                output_xml_dir=str(temp_xml_dir)
            )
            pbar.update(1)

            # --- Stage 9: Move XMLs to Final Directory (REPLACED) ---
            pbar.set_description("[Stage 9/10] Preparing XMLs")
            for xml_file in Path(temp_xml_dir).glob("*.xml"):
                shutil.copy(xml_file, final_xml_dir / xml_file.name)
            pbar.update(1)

            # --- Stage 10: Upload to S3 (NEW) ---
            pbar.set_description("[Stage 10/10] Uploading to S3")
            if s3_bucket:
                try:
                    uploader = S3Uploader(bucket_name=s3_bucket)
                    # Upload zipped frames
                    uploader.upload_directory(str(final_frames_dir), "frames")
                    # Upload individual XMLs
                    uploader.upload_directory(str(final_xml_dir), "annotations")
                except Exception as e:
                    logger.error(f"S3 upload failed. Please check your credentials and bucket name. Error: {e}")
            else:
                logger.warning("No S3 bucket provided. Skipping upload.")
            pbar.update(1)

            pbar.set_description("✅ Pipeline Complete!")
            logger.info(f"\n🎉🎉🎉 Pipeline complete! Final outputs are in: {base_output_path}")

        finally:
            # --- Cleanup ---
            if work_dir.exists():
                logger.info(f"Cleaning up temporary directory: {work_dir}")
                shutil.rmtree(work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full video pre-processing pipeline from a single ZIP file and upload to S3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    PROJECT_ROOT = Path(__file__).resolve().parent

    parser.add_argument(
        "--zip_file_name",
        required=True,
        help="Name of the master ZIP file located in the 'uploads/' directory."
    )
    parser.add_argument(
        "--s3_bucket",
        required=True,
        help="Name of the AWS S3 bucket to upload the results to."
    )
    args = parser.parse_args()

    input_zip = PROJECT_ROOT / "uploads" / args.zip_file_name
    output_path = PROJECT_ROOT / "outputs"

    if not input_zip.exists():
        logger.error(f"Input file not found: {input_zip}")
    else:
        run_pipeline(str(input_zip), str(output_path), args.s3_bucket)