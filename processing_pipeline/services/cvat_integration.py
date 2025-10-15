import requests
import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CVATClient:
    def __init__(self, host: str, username: str, password: str):
        self.host = host.rstrip('/')
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.token = None
        self.authenticated = self.login()

    def login(self) -> bool:
        """Log in to CVAT and store API token."""
        try:
            url = f"{self.host}/api/auth/login"
            resp = self.session.post(url, json={"username": self.username, "password": self.password}, timeout=30)
            resp.raise_for_status()
            self.token = resp.json()["key"]
            self.session.headers.update({"Authorization": f"Token {self.token}"})
            logger.info(f"✓ Login successful for user: {self.username}")
            return True
        except Exception as e:
            logger.error(f"Login exception: {e}")
            return False

    def _make_authenticated_request(self, method: str, url: str, **kwargs) -> requests.Response:
        if not self.authenticated:
            raise RuntimeError("Client is not authenticated.")
        kwargs.setdefault("timeout", 300)
        try:
            return self.session.request(method.upper(), url, **kwargs)
        except Exception as e:
            logger.error(f"Request failed: {method} {url} - {e}")
            raise

    def create_task(self, name: str, project_id: int) -> Optional[int]:
        """Creates a single, empty task."""
        try:
            payload = {"name": name, "project_id": project_id}
            resp = self._make_authenticated_request('POST', f"{self.host}/api/tasks", json=payload)
            if resp.status_code == 201:
                task_id = resp.json()["id"]
                logger.info(f"✓ Task '{name}' created with ID: {task_id}")
                return task_id
            logger.error(f"Failed to create task: {resp.status_code} - {resp.text}")
            return None
        except Exception as e:
            logger.error(f"Exception creating task: {e}")
            return None

    def upload_data_to_task(self, task_id: int, zip_file_path: str) -> bool:
        """Uploads a local ZIP file of data to a task."""
        try:
            with open(zip_file_path, 'rb') as fh:
                files = {'client_files[0]': (os.path.basename(zip_file_path), fh, 'application/zip')}
                data = {'image_quality': '95'}
                resp = self._make_authenticated_request('POST', f"{self.host}/api/tasks/{task_id}/data", files=files,
                                                        data=data)
            if resp.status_code != 202:
                logger.error(f"Data upload failed to start: {resp.status_code} - {resp.text}")
                return False

            rq_id = resp.json()['rq_id']
            while True:
                status_resp = self._make_authenticated_request("GET", f"{self.host}/api/requests/{rq_id}")
                status = status_resp.json().get("status")
                if status == "finished":
                    logger.info(f"✓ Data upload for task {task_id} complete.")
                    return True
                if status == "failed":
                    logger.error(f"Data upload processing failed: {status_resp.json()}")
                    return False
                time.sleep(5)
        except Exception as e:
            logger.error(f"Exception uploading data: {e}")
            return False

    def import_annotations(self, task_id: int, xml_file: str) -> bool:
        """Uploads a local XML annotation file to a specific task."""
        try:
            url = f"{self.host}/api/tasks/{task_id}/annotations?action=upload&format=CVAT%201.1"
            with open(xml_file, "rb") as fh:
                files = {"annotation_file": (os.path.basename(xml_file), fh, "application/xml")}
                resp = self._make_authenticated_request("POST", url, files=files)
            if resp.status_code not in (201, 202):
                logger.error(f"Annotation import failed: {resp.status_code} - {resp.text}")
                return False
            return True
        except Exception as e:
            logger.error(f"Exception importing annotations: {e}")
            return False

    def _get_user_id(self, username: str) -> Optional[int]:
        """Helper to get a numeric user ID from a username."""
        try:
            resp = self._make_authenticated_request('GET', f"{self.host}/api/users", params={"search": username})
            if resp.status_code == 200 and resp.json().get('results'):
                return resp.json()['results'][0]['id']
            logger.error(f"Could not find user '{username}'")
            return None
        except Exception:
            return None

    def _update_job_assignee(self, job_id: int, user_id: int) -> bool:
        """Assigns an existing job to a user."""
        try:
            resp = self._make_authenticated_request('PATCH', f"{self.host}/api/jobs/{job_id}",
                                                    json={'assignee': user_id})
            return resp.status_code == 200
        except Exception:
            return False

    def _create_and_assign_job(self, task_id: int, user_id: int) -> Optional[int]:
        """Creates a new job for a task and assigns it to a user."""
        try:
            task_data = self._make_authenticated_request('GET', f"{self.host}/api/tasks/{task_id}").json()
            segment_size = task_data.get('size', 0)

            payload = {'task_id': task_id, 'frame_count': segment_size}
            job_resp = self._make_authenticated_request('POST', f"{self.host}/api/jobs", json=payload)
            if job_resp.status_code != 201:
                logger.error(f"Failed to create new job for task {task_id}: {job_resp.text}")
                return None

            job_id = job_resp.json()['id']
            logger.info(f"✓ Created new job with ID {job_id} for task {task_id}")

            if self._update_job_assignee(job_id, user_id):
                return job_id
            return None
        except Exception as e:
            logger.error(f"Exception creating/assigning new job: {e}")
            return None

    def create_batch_task_with_multiple_jobs(
            self,
            project_id: int,
            task_name: str,
            keyframes_zip_path: str,
            annotations_xml_path: str,
            annotators: List[str]
    ) -> Optional[Dict]:
        """
        Creates one task from local files, then creates multiple jobs and assigns them.
        """
        if not annotators:
            logger.error("Annotator list cannot be empty.")
            return None

        logger.info(f"Creating master task '{task_name}'...")
        task_id = self.create_task(task_name, project_id)
        if not task_id: return None

        logger.info(f"Uploading data for task {task_id}...")
        if not self.upload_data_to_task(task_id, keyframes_zip_path):
            logger.error(f"Failed to upload data for task {task_id}. Aborting.")
            return None

        logger.info(f"Uploading annotations for task {task_id}...")
        if not self.import_annotations(task_id, annotations_xml_path):
            logger.warning(f"Failed to upload annotations for task {task_id}. Continuing with an empty task.")

        logger.info(f"Assigning {len(annotators)} annotators to task {task_id}...")
        assigned_jobs = []

        resp_jobs = self._make_authenticated_request('GET', f"{self.host}/api/jobs", params={"task_id": task_id})
        first_job = resp_jobs.json().get('results', [{}])[0]

        first_annotator_username = annotators[0]
        user_id = self._get_user_id(first_annotator_username)
        if user_id and self._update_job_assignee(first_job['id'], user_id):
            logger.info(f"✓ Assigned existing job {first_job['id']} to {first_annotator_username}")
            assigned_jobs.append({'job_id': first_job['id'], 'annotator': first_annotator_username})
        else:
            logger.error(f"✗ Failed to assign existing job to {first_annotator_username}")

        for i in range(1, len(annotators)):
            annotator_username = annotators[i]
            user_id = self._get_user_id(annotator_username)
            if user_id:
                new_job_id = self._create_and_assign_job(task_id, user_id)
                if new_job_id:
                    logger.info(f"✓ Created and assigned new job {new_job_id} to {annotator_username}")
                    assigned_jobs.append({'job_id': new_job_id, 'annotator': annotator_username})
                else:
                    logger.error(f"✗ Failed to create/assign new job for {annotator_username}")

        return {
            "task_id": task_id,
            "task_name": task_name,
            "assigned_jobs": assigned_jobs
        }


def get_default_labels() -> List[Dict[str, Any]]:
    """Defines the label schema for the CVAT project."""
    return [
        {
            "name": "person",
            "color": "#ff0000",
            "attributes": [
                {"name": "walking_behavior", "mutable": True, "input_type": "select", "default_value": "normal_walk",
                 "values": ["normal_walk", "fast_walk", "slow_walk", "standing_still", "jogging", "window_shopping"]},
                {"name": "phone_usage", "mutable": True, "input_type": "select", "default_value": "no_phone",
                 "values": ["no_phone", "talking_phone", "texting", "taking_photo", "listening_music"]},
                {"name": "social_interaction", "mutable": True, "input_type": "select", "default_value": "alone",
                 "values": ["alone", "talking_companion", "group_walking", "greeting_someone", "asking_directions",
                            "avoiding_crowd"]},
                {"name": "carrying_items", "mutable": True, "input_type": "select", "default_value": "empty_hands",
                 "values": ["empty_hands", "shopping_bags", "backpack", "briefcase_bag", "umbrella", "food_drink",
                            "multiple_items"]},
                {"name": "street_behavior", "mutable": True, "input_type": "select",
                 "default_value": "sidewalk_walking",
                 "values": ["sidewalk_walking", "crossing_street", "waiting_signal", "looking_around", "checking_map",
                            "entering_building", "exiting_building"]},
                {"name": "posture_gesture", "mutable": True, "input_type": "select", "default_value": "upright_normal",
                 "values": ["upright_normal", "looking_down", "looking_up", "hands_in_pockets", "arms_crossed",
                            "pointing_gesture", "bowing_gesture"]},
                {"name": "clothing_style", "mutable": True, "input_type": "select", "default_value": "business_attire",
                 "values": ["business_attire", "casual_wear", "tourist_style", "school_uniform", "sports_wear",
                            "traditional_wear"]},
                {"name": "time_context", "mutable": True, "input_type": "select", "default_value": "rush_hour",
                 "values": ["rush_hour", "leisure_time", "shopping_time", "tourist_hours", "lunch_break",
                            "evening_stroll"]},
            ]
        }
    ]