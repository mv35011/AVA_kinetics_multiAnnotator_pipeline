import os
import pickle
import argparse
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# (prettify_xml function remains the same)
def prettify_xml(elem):
    """Returns a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


# Main logic is now in this importable function
def generate_xml_for_batch(batch_name: str, pickle_path: str, keyframes_dir: str, output_xml_path: str):
    """
    Main function to generate a single CVAT XML for a batch of keyframes.
    This function will be called by orchestrator.py.
    """
    try:
        with open(pickle_path, 'rb') as f:
            proposals_data = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Pickle file not found at {pickle_path}")
        return

    # Example attributes - you should move this to a shared config file
    attributes_dict = {
        'work_activity': dict(aname='work_activity', default='idle', options=['idle', 'welding', 'cutting', 'lifting']),
        'ppe_helmet': dict(aname='ppe_helmet', default='no_helmet', options=['no_helmet', 'helmet_worn']),
        # Add your other ~10 attribute groups here
    }

    annotations = ET.Element('annotations')
    ET.SubElement(annotations, 'version').text = '1.1'

    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    ET.SubElement(task, 'name').text = batch_name
    ET.SubElement(task, 'mode').text = 'annotation'

    labels_xml = ET.SubElement(task, 'labels')
    person_label = ET.SubElement(labels_xml, 'label')
    ET.SubElement(person_label, 'name').text = 'person'
    ET.SubElement(person_label, 'color').text = '#ff0000'
    attributes_xml = ET.SubElement(person_label, 'attributes')
    for attr_data in attributes_dict.values():
        attribute = ET.SubElement(attributes_xml, 'attribute')
        ET.SubElement(attribute, 'name').text = attr_data['aname']
        ET.SubElement(attribute, 'mutable').text = 'true'
        ET.SubElement(attribute, 'input_type').text = 'select'
        ET.SubElement(attribute, 'default_value').text = attr_data['default']
        ET.SubElement(attribute, 'values').text = '\n'.join(attr_data['options'])

    image_id = 0
    all_keyframes = sorted([
        frame_name for clip_data in proposals_data.values() for frame_name in clip_data.keys()
    ])

    for frame_name in tqdm(all_keyframes, desc="  -> Adding keyframes to XML"):
        frame_path = os.path.join(keyframes_dir, frame_name)
        if not os.path.exists(frame_path): continue

        try:
            img = cv2.imread(frame_path)
            height, width, _ = img.shape
        except Exception:
            continue

        clip_id = '_'.join(frame_name.split('_')[:-2])
        detections = proposals_data.get(clip_id, {}).get(frame_name, [])

        image_xml = ET.SubElement(annotations, 'image', {
            'id': str(image_id), 'name': frame_name, 'width': str(width), 'height': str(height)
        })

        for det in detections:
            bbox = det[0:4]
            box_xml = ET.SubElement(image_xml, 'box', {
                'label': 'person', 'occluded': '0',
                'xtl': str(bbox[0]), 'ytl': str(bbox[1]), 'xbr': str(bbox[2]), 'ybr': str(bbox[3])
            })
            for attr_data in attributes_dict.values():
                ET.SubElement(box_xml, 'attribute', {'name': attr_data['aname']}).text = attr_data['default']
        image_id += 1

    xml_content = prettify_xml(annotations)
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)

    logger.info(f"✅ Successfully created consolidated CVAT XML at: {output_xml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a single CVAT XML for a batch of keyframes.")
    parser.add_argument('--pickle_path', required=True, help="Path to the aggregated dense_proposals.pkl file.")
    parser.add_argument('--keyframes_dir', required=True, help="Directory containing the keyframe .jpg files.")
    parser.add_argument('--output_xml_path', required=True, help="Full path for the final output XML file.")
    parser.add_argument('--batch_name', required=True, help="Name of the batch, used as the CVAT task name.")
    args = parser.parse_args()

    generate_xml_for_batch(
        batch_name=args.batch_name,
        pickle_path=args.pickle_path,
        keyframes_dir=args.keyframes_dir,
        output_xml_path=args.output_xml_path
    )