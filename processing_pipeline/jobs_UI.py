import streamlit as st
import os
from pathlib import Path

# Adjust the import path if your file structure is different
try:
    from services.cvat_integration import CVATClient
    from services.assignment_generator import AssignmentGenerator
except ImportError:
    st.error("Could not import custom services. Make sure this app is run from your project's root directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="CVAT Multi-Job Creator", layout="wide")
st.title("🚀 CVAT Multi-Job Task Creator")
st.markdown("This app creates a single CVAT task from a batch and assigns it to multiple annotators as separate jobs.")

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration")
st.sidebar.info("These settings point to your local services and files.")

cvat_host = st.sidebar.text_input("CVAT Host URL", "http://localhost:8080")
cvat_user = st.sidebar.text_input("CVAT Username", "mv350")
cvat_pass = st.sidebar.text_input("CVAT Password", "Amazon123", type="password")
project_id = st.sidebar.number_input("Target CVAT Project ID", min_value=1, value=1)

st.sidebar.subheader("File Paths")
keyframes_zip_path_str = st.sidebar.text_input("Path to Keyframes ZIP", "outputs/factory_test_batch_01_keyframes.zip")
annotations_xml_path_str = st.sidebar.text_input("Path to Annotations XML",
                                                 "outputs/factory_test_batch_01_annotations.xml")

# --- Main Page for Task Creation ---
st.header("Step 1: Define the Batch Task")

task_name = st.text_input("Task Name (e.g., the batch name)", "factory_test_batch_01")
all_annotators_str = st.text_area("List of ALL Available Annotators (comma-separated)",
                                  "annotator1, annotator2")
num_to_assign = st.number_input("Number of Annotators to Assign to this Task", min_value=1, value=2)

st.header("Step 2: Execute")

if st.button("Create Task with Multiple Jobs", type="primary"):
    all_annotators = [a.strip() for a in all_annotators_str.split(',') if a.strip()]
    keyframes_zip_path = Path(keyframes_zip_path_str)
    annotations_xml_path = Path(annotations_xml_path_str)

    # --- Input Validation ---
    if not all([cvat_host, cvat_user, cvat_pass, task_name]):
        st.error("Please fill in all configuration settings in the sidebar and provide a task name.")
        st.stop()
    if not all_annotators:
        st.warning("Please provide at least one available annotator.")
        st.stop()
    if not keyframes_zip_path.exists():
        st.error(f"Keyframes ZIP file not found at: '{keyframes_zip_path.resolve()}'")
        st.stop()
    if not annotations_xml_path.exists():
        st.error(f"Annotations XML file not found at: '{annotations_xml_path.resolve()}'")
        st.stop()

    with st.spinner("Processing... Please wait."):
        try:
            # --- 1. Select Annotators for the Batch ---
            st.subheader("📋 Assignment Plan")
            assignment_service = AssignmentGenerator()
            selected_annotators = assignment_service.select_annotators_for_batch(
                all_annotators=all_annotators,
                num_required=num_to_assign
            )
            st.info(f"The following annotators will be assigned to this batch: **{', '.join(selected_annotators)}**")

            # --- 2. Connect to CVAT and Execute ---
            st.subheader("⚡️ CVAT Communication Log")
            log_container = st.empty()
            log_container.info("Authenticating with CVAT...")

            client = CVATClient(host=cvat_host, username=cvat_user, password=cvat_pass)
            if not client.authenticated:
                log_container.error("CVAT Authentication Failed! Check host, username, and password.")
                st.stop()

            log_container.success("Authentication successful. Creating task and jobs...")

            # --- 3. Run the Multi-Job Creation Function ---
            result = client.create_batch_task_with_multiple_jobs(
                project_id=project_id,
                task_name=task_name,
                keyframes_zip_path=str(keyframes_zip_path),
                annotations_xml_path=str(annotations_xml_path),
                annotators=selected_annotators
            )

            # --- 4. Display Results ---
            st.subheader("✅ Results")
            if result:
                st.success("Task and jobs created successfully!")
                st.json(result)
                st.balloons()
            else:
                st.error("Process failed. Check the terminal logs for specific errors from `cvat_integration.py`.")

        except Exception as e:
            st.error("An unexpected error occurred:")
            st.exception(e)