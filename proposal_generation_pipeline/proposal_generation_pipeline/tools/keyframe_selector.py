import os

import cv2
import numpy as np
import torch
from rfdetr import RFDETRMedium
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class KeyframeSelector:
    """
    Selects the best keyframe from a video clip based on a combined score of
    detector confidence and person-centric motion, with tie-breaker logic.
    """

    def __init__(self, detection_model: RFDETRMedium, device: str, person_class_id: int = 1):
        self.model = detection_model
        self.person_class_id = person_class_id
        self.device = device

    def _z_normalize(self, scores: List[float]) -> np.ndarray:
        """Applies z-score normalization to a list of scores."""
        scores_arr = np.array(scores)
        mean = np.mean(scores_arr)
        std = np.std(scores_arr)
        if std < 1e-6:
            return np.zeros_like(scores_arr)
        return (scores_arr - mean) / std

    def select_best_keyframe(
            self,
            video_path: str,
            center_window_secs: float = 4.0,
            candidate_stride: int = 3,
            w_motion: float = 0.7,
            w_confidence: float = 0.3
    ) -> Optional[Tuple[np.ndarray, int, List[Dict]]]:
        """
        Analyzes a window of frames in a video and returns the best one.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30

        middle_frame = total_frames // 2
        window_half_frames = int(center_window_secs / 2 * fps)
        start_frame = max(0, middle_frame - window_half_frames)
        end_frame = min(total_frames, middle_frame + window_half_frames)

        candidate_indices = np.array(range(start_frame, end_frame, candidate_stride))
        if candidate_indices.size == 0:
            logger.warning(f"No candidate frames found for video {video_path}")
            cap.release()
            return None

        candidate_frames_rgb, all_detections = [], []
        for frame_idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                candidate_frames_rgb.append(frame_rgb)
                dets = self.model.predict(frame_rgb, threshold=0.5)
                all_detections.append(dets)
        cap.release()

        if not candidate_frames_rgb: return None

        confidence_scores, motion_scores = [], []
        prev_gray = None
        for i, (frame_rgb, dets) in enumerate(zip(candidate_frames_rgb, all_detections)):
            person_mask = np.zeros(0, dtype=bool)
            if hasattr(dets, 'class_id') and dets.class_id is not None:
                person_mask = (dets.class_id == self.person_class_id)
                score = dets.confidence[person_mask].sum().item() if hasattr(dets,
                                                                             'confidence') and person_mask.any() else 0
            else:
                score = 0
            confidence_scores.append(score)

            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            motion_score = 0
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                if hasattr(dets, 'xyxy') and dets.xyxy is not None and person_mask.any():
                    # --- FIX IS HERE ---
                    # The .cpu() call is removed as dets.xyxy is already a numpy array
                    person_boxes = dets.xyxy[person_mask].astype(int)
                    for x1, y1, x2, y2 in person_boxes:
                        box_mag = mag[y1:y2, x1:x2]
                        if box_mag.size > 0: motion_score += box_mag.mean()

            motion_scores.append(motion_score)
            prev_gray = gray

        norm_conf = self._z_normalize(confidence_scores)
        norm_motion = self._z_normalize(motion_scores)

        combined_scores = (w_motion * norm_motion) + (w_confidence * norm_conf)

        distance_from_middle = np.abs(candidate_indices - middle_frame)
        tie_breaker_scores = 1.0 - (distance_from_middle / (total_frames / 2))

        final_scores = combined_scores + (tie_breaker_scores * 1e-6)

        if len(final_scores) == 0:
            logger.warning(f"Could not compute scores for {video_path}")
            return None

        logger.debug(f"Video: {os.path.basename(video_path)}")
        logger.debug(f"Candidate Indices: {candidate_indices}")
        logger.debug(f"Confidence Scores (Normalized): {np.round(norm_conf, 2)}")
        logger.debug(f"Motion Scores (Normalized): {np.round(norm_motion, 2)}")
        logger.debug(f"Final Scores: {np.round(final_scores, 2)}")

        best_idx = np.argmax(final_scores)

        best_frame_image_bgr = cv2.cvtColor(candidate_frames_rgb[best_idx], cv2.COLOR_RGB2BGR)
        best_frame_original_idx = candidate_indices[best_idx]
        best_dets = all_detections[best_idx]

        final_detections = []
        if hasattr(best_dets, 'xyxy') and best_dets.xyxy is not None:
            person_mask = (best_dets.class_id == self.person_class_id)
            if person_mask.any():
                for i, (box, conf) in enumerate(zip(best_dets.xyxy[person_mask], best_dets.confidence[person_mask])):
                    final_detections.append({"track_id": i + 1, "bbox": [c.item() for c in box]})

        return best_frame_image_bgr, best_frame_original_idx, final_detections