import os
import cv2
import json
import torch
import numpy as np
from rfdetr import RFDETRMedium
from tqdm import tqdm
import logging
from argparse import Namespace

# Note: The original byte_tracker might be overkill if you only process one frame,
# but we keep it for consistency in generating track IDs.
from .byte_tracker import BYTETracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PersonTracker:
    def __init__(self, video_id: str, conf=0.5, person_class_id=0):  # COCO person_class_id is 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RFDETRMedium(device=self.device)
        self.model.optimize_for_inference()
        self.conf = conf
        self.person_class_id = person_class_id
        self.video_id = video_id

        tracker_args = Namespace(
            track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False
        )
        self.tracker = BYTETracker(tracker_args, frame_rate=30)

    def _parse_detections(self, dets):
        # This helper function remains the same as your original
        if hasattr(dets, 'xyxy'):
            boxes, scores, labels = dets.xyxy, dets.confidence, dets.class_id
        else:
            return np.empty((0, 5))

        if torch.is_tensor(boxes): boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores): scores = scores.cpu().numpy()
        if torch.is_tensor(labels): labels = labels.cpu().numpy()

        if labels.size == 0:
            return np.empty((0, 5))

        mask = (labels == self.person_class_id) & (scores >= self.conf)
        person_boxes, person_scores = boxes[mask], scores[mask]

        if len(person_boxes) == 0:
            return np.empty((0, 5))

        return np.hstack((person_boxes, person_scores[:, np.newaxis]))

    def process_single_keyframe(self, video_path: str, output_json_dir: str, output_frame_dir: str) -> dict:
        """
        NEW: Extracts only the middle frame of a video, runs detection and tracking on it,
        saves the frame, and returns its metadata for the manifest.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_idx = total_frames // 2

        # Seek to the middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error(f"Could not read middle frame from {video_path}")
            return None

        # --- Run detection and tracking on this single frame ---
        raw_detections = self.model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=self.conf)
        detections = self._parse_detections(raw_detections)
        frame_shape = [frame.shape[0], frame.shape[1]]
        online_tracks = self.tracker.update(detections, frame_shape, frame_shape)

        # --- Save the keyframe with a unique, informative name ---
        keyframe_name = f"{self.video_id}_frame_{middle_frame_idx:04d}.jpg"
        frame_save_path = os.path.join(output_frame_dir, keyframe_name)
        cv2.imwrite(frame_save_path, frame)

        # --- Collect data for this frame ---
        all_detections_data = []
        for track in online_tracks:
            all_detections_data.append({
                "video_id": self.video_id,
                "frame": keyframe_name,
                "track_id": int(track.track_id),
                "bbox": [float(c) for c in track.tlbr]
            })

        # --- Save the JSON for this single frame ---
        json_output_path = os.path.join(output_json_dir, f"{self.video_id}.json")
        with open(json_output_path, "w") as f:
            json.dump(all_detections_data, f, indent=2)

        logger.info(f"✅ Keyframe processed for {self.video_id}.")

        # Return metadata for the manifest file
        return {
            "keyframe_name": keyframe_name,
            "source_video": os.path.basename(video_path),
            "source_frame": middle_frame_idx
        }