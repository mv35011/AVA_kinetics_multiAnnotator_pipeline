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

    def import_annotations(self, task_id: int, xml_file: str) -> bool:
        """Uploads a local XML annotation file to a specific task."""
        try:
            url = f"{self.host}/api/tasks/{task_id}/annotations?action=upload&format=CVAT%201.1"
            with open(xml_file, "rb") as fh:
                files = {"annotation_file": (os.path.basename(xml_file), fh, "application/xml")}
                resp = self._make_authenticated_request("POST", url, files=files)

            if resp.status_code not in (201, 202):
                logger.error(f"Annotation import failed to start: {resp.status_code} - {resp.text}")
                return False

            rq_id = resp.json().get("rq_id")
            if not rq_id:
                logger.info(f"✓ Annotation import for task {task_id} completed synchronously.")
                return True

            logger.info(f"Started annotation import job {rq_id} for task {task_id}")
            return self._wait_for_request_completion(rq_id)
        except Exception as e:
            logger.error(f"Exception importing annotations for task {task_id}: {e}")
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

    def create_project(self, name: str, labels: List[Dict[str, Any]], org_slug: str = None) -> Optional[int]:
        """Creates a new project, optionally within an organization."""
        try:
            payload = {"name": name, "labels": labels}
            if org_slug: payload['org'] = org_slug

            resp = self._make_authenticated_request('POST', f"{self.host}/api/projects", json=payload)
            if resp.status_code == 201:
                project_id = resp.json()["id"]
                logger.info(f"✓ Project '{name}' created with ID: {project_id}")
                return project_id
            else:
                logger.error(f"Failed to create project: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Exception creating project: {e}")
            return None

    def _get_cloud_storage_id(self, s3_bucket_name: str) -> Optional[int]:
        """Dynamically finds the Cloud Storage ID for a given S3 bucket name."""
        try:
            url = f"{self.host}/api/cloudstorages"
            resp = self._make_authenticated_request("GET", url)
            resp.raise_for_status()
            for storage in resp.json().get("results", []):
                if s3_bucket_name in storage.get("display_name", ""):
                    logger.info(f"Found Cloud Storage '{storage['display_name']}' with ID: {storage['id']}")
                    return storage["id"]
            logger.error(f"✗ Cloud Storage for bucket '{s3_bucket_name}' not found in CVAT.")
            return None
        except Exception as e:
            logger.error(f"Failed to get cloud storages: {e}")
            return None

    def get_all_tasks_for_project(self, project_id: int) -> List[Dict]:
        """Fetches the full details of all tasks within a project."""
        try:
            url = f"{self.host}/api/tasks?project_id={project_id}"
            resp = self._make_authenticated_request("GET", url)
            resp.raise_for_status()
            logger.info(f"✓ Successfully fetched all tasks for project {project_id}")
            return resp.json().get("results", [])
        except Exception as e:
            logger.error(f"Failed to get tasks for project ID {project_id}: {e}")
            return []

    def _wait_for_request_completion(self, rq_id: str, timeout: int = 600) -> bool:
        """Polls a request ID until it is finished or failed."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_resp = self._make_authenticated_request("GET", f"{self.host}/api/requests/{rq_id}")
            if status_resp.status_code != 200: return False
            status = status_resp.json().get("status")
            if status == "finished":
                logger.info(f"✓ Job {rq_id} finished successfully.")
                return True
            if status == "failed":
                logger.error(f"✗ Job {rq_id} failed: {status_resp.json()}")
                return False
            time.sleep(5)
        logger.error(f"✗ Job {rq_id} timed out.")
        return False

    def assign_user_to_task(self, task_id: int, username: str) -> bool:
        """Assigns all jobs in a task to a specific user."""
        try:
            resp_user = self._make_authenticated_request('GET', f"{self.host}/api/users", params={"search": username})
            user_id = resp_user.json()['results'][0]['id']

            resp_jobs = self._make_authenticated_request('GET', f"{self.host}/api/jobs", params={"task_id": task_id})
            jobs = resp_jobs.json().get('results', [])

            for job in jobs:
                self._make_authenticated_request('PATCH', f"{self.host}/api/jobs/{job['id']}",
                                                 json={'assignee': user_id})
            logger.info(f"✓ Assigned all jobs in task {task_id} to '{username}'")
            return True
        except Exception as e:
            logger.error(f"Exception assigning user to task {task_id}: {e}")
            return False

    def create_s3_backed_task(self, name: str, project_id: int, cloud_storage_id: int, s3_files: List[str]) -> Optional[
        int]:
        """Creates a single task in CVAT using files from a configured S3 cloud storage."""
        try:
            logger.info(f"Creating S3-backed task '{name}' in project {project_id}...")

            payload = {
                "name": name,
                "project_id": project_id,
                "cloud_storage_id": cloud_storage_id,
                "server_files": s3_files,
                "storage_method": "cloud_storage",
                "storage": "cloud_storage",
                "image_quality": 95,
            }

            resp = self._make_authenticated_request('POST', f"{self.host}/api/tasks", json=payload)

            if resp.status_code == 201:
                task_id = resp.json().get("id")
                logger.info(f"✓ Task '{name}' created with ID: {task_id}")
                return task_id

            logger.error(f"Failed to create S3-backed task: {resp.status_code} - {resp.text}")
            return None
        except Exception as e:
            logger.error(f"Exception creating S3-backed task: {e}")
            return None

    # In cvat_integration.py, REPLACE the existing function with this one

    def create_project_and_tasks_s3(
            self,
            project_name: str,
            assignments: Dict[str, List[str]],
            s3_bucket_name: str,
            local_xml_dir: Path,
            org_slug: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Creates a project, then loops to create one task per clip from S3 and
        uploads annotations to each task individually.
        """
        # Step 1: Create the Project
        project_id = self.create_project(project_name, get_default_labels(), org_slug)
        if not project_id: return None

        # Step 2: Get Cloud Storage ID
        cloud_storage_id = self._get_cloud_storage_id(s3_bucket_name)
        if not cloud_storage_id: return None

        all_clips = sorted(list(set(clip for clip_list in assignments.values() for clip in clip_list)))

        # --- CHANGED LOGIC: Loop, create task, then immediately upload annotations ---
        logger.info(f"Starting creation and annotation for {len(all_clips)} clips...")
        for clip_name in all_clips:
            task_name = clip_name.replace('.zip', '')
            s3_file_path = f"frames/{clip_name}"

            # Create the task from S3 data
            task_id = self.create_s3_backed_task(task_name, project_id, cloud_storage_id, [s3_file_path])
            if not task_id:
                logger.error(f"Skipping annotation and assignment for failed task '{task_name}'")
                continue

            # Give CVAT a moment to finish creating the task from S3 data
            time.sleep(5)

            # Now, upload annotations to this specific task
            xml_file = local_xml_dir / f"{task_name}_annotations.xml"
            if xml_file.exists():
                # NOTE: We are calling the original 'import_annotations' that targets a task_id
                if not self.import_annotations(task_id, str(xml_file)):
                    logger.error(f"Failed to import annotations for task {task_id}")
            else:
                logger.warning(f"XML file not found for {clip_name}, skipping annotation.")

        logger.info("✅ All tasks created. Waiting for server to process...")
        time.sleep(10)

        # Step 4: Assign Tasks
        logger.info("Assigning tasks to annotators...")
        all_tasks = self.get_all_tasks_for_project(project_id)
        task_map = {task['name']: task['id'] for task in all_tasks}

        tasks_assigned_count = 0
        for annotator, clips in assignments.items():
            for clip_name in clips:
                task_name = clip_name.replace('.zip', '')
                if task_name in task_map:
                    task_id = task_map[task_name]
                    if self.assign_user_to_task(task_id, annotator):
                        tasks_assigned_count += 1
                else:
                    logger.error(f"Could not find task named '{task_name}' to assign.")

        return {"project_id": project_id, "tasks_created": len(all_tasks), "tasks_assigned": tasks_assigned_count}


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