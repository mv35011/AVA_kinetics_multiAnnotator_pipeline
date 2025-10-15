import os
import json
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_proposals_from_tracks(tracking_dir, output_path):
    """
    MODIFIED: Aggregates single-keyframe JSON files into a single proposals PKL file.
    The structure will be {clip_id: {keyframe_name: [detections]}}.
    """
    if not os.path.isdir(tracking_dir):
        logger.error(f"❌ Tracking directory not found at '{tracking_dir}'")
        return

    json_files = [f for f in os.listdir(tracking_dir) if f.endswith('.json')]
    if not json_files:
        logger.error(f"❌ No tracking .json files found in '{tracking_dir}'")
        return

    logger.info(f"🔍 Found {len(json_files)} keyframe JSON files to process.")
    results_dict = defaultdict(lambda: defaultdict(list))

    for json_file in tqdm(json_files, desc="Processing keyframe detections"):
        file_path = os.path.join(tracking_dir, json_file)

        try:
            with open(file_path, 'r') as f:
                tracked_detections = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"⚠️ Skipping corrupted or empty JSON file: {json_file}")
            continue

        for det in tracked_detections:
            video_id = det.get('video_id')
            frame_name = det.get('frame')
            bbox = det.get('bbox')
            track_id = det.get('track_id')

            if not all([video_id, frame_name, bbox, track_id]):
                continue

            # The format [x1, y1, x2, y2, score, track_id] is kept for compatibility
            proposal_entry = [bbox[0], bbox[1], bbox[2], bbox[3], 1.0, track_id]
            results_dict[video_id][frame_name].append(proposal_entry)

    if not results_dict:
        logger.error("❌ No detections were processed. Check your JSON files.")
        return

    # Convert defaultdicts to regular dicts for pickling
    final_dict = {vid: dict(frames) for vid, frames in results_dict.items()}

    with open(output_path, "wb") as pkl_file:
        pickle.dump(final_dict, pkl_file)
    logger.info(f"💾 Successfully aggregated proposals to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate keyframe JSONs to a dense proposals PKL file.")
    parser.add_argument('--tracking_dir', required=True, help="Directory containing keyframe JSON files.")
    parser.add_argument('--output_path', required=True, help="Path to save the final dense_proposals.pkl file.")
    args = parser.parse_args()
    generate_proposals_from_tracks(args.tracking_dir, args.output_path)