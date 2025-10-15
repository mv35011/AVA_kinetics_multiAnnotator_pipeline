import streamlit as st
import os
from pathlib import Path
import time

# Assuming your services are in a 'processing_pipeline/services' directory
# Adjust the import path if your file structure is different
try:
    from cvat_integration import CVATClient
    from assignment_generator import AssignmentGenerator
except ImportError:
    st.error("Could not import custom services. Make sure this app is run from your project's root directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="CVAT Task Creation Test", layout="wide")
st.title("🛰️ CVAT S3 Task Creation Tester")
st.markdown("This app directly tests the task creation workflow using S3 cloud storage.")

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Connection Settings")
st.sidebar.info("These values should point to your cloud-hosted services.")

cvat_host = st.sidebar.text_input(
    "CVAT Host URL",
    os.getenv("CVAT_HOST", "http://184.73.121.6:8080")
)
cvat_user = st.sidebar.text_input(
    "CVAT Username",
    os.getenv("CVAT_USERNAME", "Anurag_03")
)
cvat_pass = st.sidebar.text_input(
    "CVAT Password",
    os.getenv("CVAT_PASSWORD", "Test@123"),
    type="password"
)
s3_bucket = st.sidebar.text_input(
    "S3 Bucket Name",
    os.getenv("S3_BUCKET_NAME", "cvat-data-uploader")
)
project_id = st.sidebar.number_input(
    "Target CVAT Project ID",
    min_value=1,
    value=1
)
# This path is relative to where you run the streamlit app
local_xml_dir_str = st.sidebar.text_input(
    "Local Path to XMLs Folder",
    "annotations/annotations/"
)
local_xml_dir = Path(local_xml_dir_str)

# --- Main Page for Task Creation ---
st.header("Step 1: Define the Task Batch")

clips_str = st.text_area(
    "Clip Filenames from S3 (one per line, e.g., 'clip1.zip')",
    "1_clip_000.zip\n1_clip_001.zip\n1_clip_002.zip\n1_clip_003.zip\n1_clip_004.zip\n1_clip_005.zip"
)

#1_clip_000.zip,1_clip_001.zip,1_clip_002.zip,1_clip_003.zip,1_clip_004.zip,1_clip_005.zip
annotators_str = st.text_area(
    "Annotator Usernames (comma-separated)",
    "annotator1, annotator2"
)
overlap_percent = st.slider("Overlap Percentage for QC", 0, 100, 20)

st.header("Step 2: Execute and Monitor")

if st.button("🚀 Create Tasks in CVAT", type="primary"):
    clips = [c.strip() for c in clips_str.split('\n') if c.strip()]
    annotators = [a.strip() for a in annotators_str.split(',') if a.strip()]

    # --- Input Validation ---
    if not all([cvat_host, cvat_user, cvat_pass, s3_bucket]):
        st.error("Please fill in all connection settings in the sidebar.")
        st.stop()
    if not clips or not annotators:
        st.warning("Please provide at least one clip filename and one annotator.")
        st.stop()
    if not local_xml_dir.exists() or not local_xml_dir.is_dir():
        st.error(f"The local XML directory was not found at: '{local_xml_dir.resolve()}'")
        st.info("This path must be valid on the machine running this Streamlit app.")
        st.stop()

    with st.spinner("Processing... Please wait."):
        try:
            # --- 1. Generate Assignment Plan ---
            st.subheader("📋 Generated Assignment Plan")
            assignment_service = AssignmentGenerator()
            assignments = assignment_service.generate_random_assignments(
                clips=clips,
                annotators=annotators,
                overlap_percentage=overlap_percent
            )
            st.json(assignments)

            # --- 2. Connect to CVAT and Execute ---
            st.subheader("⚡️ CVAT Communication Log")
            log_container = st.empty()
            log_container.info("Authenticating with CVAT...")

            client = CVATClient(host=cvat_host, username=cvat_user, password=cvat_pass)

            if not client.authenticated:
                log_container.error("CVAT Authentication Failed! Please check host, username, and password.")
                st.stop()

            log_container.success("Authentication successful. Starting task creation...")

            # --- 3. Run the S3 Task Creation ---
            # --- AFTER (Correct) ---
            created_tasks = client.create_project_and_tasks_s3(
                project_name=f"Project_For_Batch_{int(time.time())}",  # Create a unique project name
                assignments=assignments,
                s3_bucket_name=s3_bucket,
                local_xml_dir=local_xml_dir
            )

            # --- 4. Display Results ---
            st.subheader("✅ Results")
            if created_tasks:
                st.success("Task creation process completed successfully!")
                st.json(created_tasks)
                st.balloons()
            else:
                st.warning(
                    "The process completed, but no tasks were created. Check the logs in your terminal for specific errors (e.g., missing XML files, S3 connection issues).")

        except Exception as e:
            st.error(f"An unexpected error occurred:")
            st.exception(e)