import os
import json
import cv2
import tempfile
import torch
import numpy as np
from torchvision import models, transforms
import sys
from scipy.spatial.transform import Rotation
import hashlib
from functools import cached_property
from PIL import Image
from dataclasses import dataclass
from tqdm import tqdm
import gc  # For garbage collection

def convert_extrinsic_vectors_to_matrix(extrinsic_translation: np.ndarray,
                                        extrinsic_rotation: np.ndarray) -> np.ndarray:
    return np.column_stack([Rotation.from_rotvec(extrinsic_rotation).as_matrix(), extrinsic_translation])

def convert_extrinsic_matrix_to_vectors(extrinsic_matrix: np.ndarray) -> tuple:
    return np.squeeze(extrinsic_matrix[:, 3]), Rotation.from_matrix(extrinsic_matrix[:3, :3]).as_rotvec()

def get_camera_transform(extrinsic_matrix: np.ndarray) -> np.ndarray:
    R, t = extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3]
    camera_transform = np.eye(4)
    camera_transform[:3, :3] = R.T
    camera_transform[:3, 3] = -R.T.dot(t)
    return camera_transform

def get_transform(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    transform[:3, 3] = translation
    return transform

def project_to_image(pts_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_2d = np.dot(camera_matrix, pts_3d_homo.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d

def inverse_extrinsics(extrinsic_matrix: np.ndarray) -> np.ndarray:
    R, t = (extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3])
    return np.column_stack([R.T, -R.T.dot(t)])

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

@dataclass
class Object:
    trans: np.ndarray
    rot: np.ndarray
    dim: np.ndarray
    cat_id: int
    track_id: int
    att_id: int
    towed_by: int | None = None

    @classmethod
    def get_field_data(cls, ann: dict) -> dict:
        return {
            'trans': np.array(ann['translation']),
            'rot': np.array(ann['rotation']),
            'dim': np.array(ann['dimension']),
            'cat_id': ann['category_id'],
            'track_id': ann['track_id'],
            'att_id': ann.get('attribute_id') or 0
        }

    @classmethod
    def deserialize(cls, ann: dict) -> 'Object':
        return cls(**cls.get_field_data(ann))

    @cached_property
    def box_3d(self) -> np.ndarray:
        length, width, height = self.dim
        corners_x = [ length/2,  length/2, -length/2, -length/2,  length/2,  length/2, -length/2, -length/2]
        corners_y = [ width/2, -width/2,  -width/2,  width/2,  width/2, -width/2,  -width/2,  width/2]
        corners_z = [0, 0, 0, 0, height, height, height, height]
        T = get_transform(self.trans, self.rot)
        corners_local = np.array([corners_x, corners_y, corners_z])
        return np.dot(T[:3, :3], corners_local).T + T[:3, 3].reshape(1, 3)

    def get_bbox(self, camera_matrix: np.ndarray) -> np.ndarray:
        box_corners_2d = project_to_image(self.box_3d, camera_matrix)
        u_min, v_min = np.min(box_corners_2d[:, 0]), np.min(box_corners_2d[:, 1])
        u_max, v_max = np.max(box_corners_2d[:, 0]), np.max(box_corners_2d[:, 1])
        return np.array([u_min, v_min, (u_max - u_min), (v_max - v_min)])


def load_object_detection_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(model, image, threshold=0.7):
    transform = transforms.Compose([transforms.ToTensor()])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image)
    with torch.no_grad():
        predictions = model([image_tensor])
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    boxes = boxes[scores >= threshold]
    return boxes.cpu().numpy()

def draw_bounding_boxes(image, bounding_boxes, color=(0,255,0), thickness=2):
    for bbox in bounding_boxes:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

def get_unique_color(obj_id):
    hash_val = hashlib.md5(str(obj_id).encode()).hexdigest()
    r = int(hash_val[:2], 16)
    g = int(hash_val[2:4], 16)
    b = int(hash_val[4:6], 16)
    return [r, g, b]


def rotation_to_camera_frame(rotation: list, extrinsic_matrix: np.ndarray) -> np.ndarray:
    rotation_in_camera_frame = np.dot(extrinsic_matrix[:3, :3], Rotation.from_rotvec(rotation).as_matrix())
    return Rotation.from_matrix(rotation_in_camera_frame).as_rotvec()

def transform_to_camera_frame(pt_3d: np.ndarray, extrinsic_matrix: np.ndarray) -> np.ndarray:
    """Transform pt_3d from world frame to camera frame."""
    pt_3d_homo = np.append(pt_3d, 1)
    return np.dot(extrinsic_matrix, pt_3d_homo)[:3]

def process_recording(
    recording_dir,
    inference_state,
    predictor,
    detection_model,
    output_database_dir="object_database",
    storage_mode="ground_truth",
    batch_size=4
):
    """
    Updated to use predictor.predict_batch(...) in batches of size `batch_size`.
    """

    recording_name = os.path.basename(recording_dir)
    output_mode_dir = os.path.join(output_database_dir, storage_mode)
    base_output_dir = os.path.join(output_mode_dir, recording_name)
    print(f"Saving outputs to: {base_output_dir}")
    os.makedirs(base_output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(recording_dir) if f.endswith(".jpg")])
    if not image_files:
        print(f"No JPEG images found in {recording_dir}. Skipping.")
        return

    base_dir = os.path.join("/", *recording_dir.split("/")[:-2])
    meta_data_path = os.path.join(base_dir, "labels", recording_name, "preprocessing_meta.json")
    meta_data = load_json(meta_data_path)

    all_bboxes_per_frame = []
    all_objects_per_frame = []
    intrinsic_matrices = []
    camera_matrices = []
    extrinsic_matrices = []
    annotation_data = {}

    print("\tExtracting Bounding boxes...")
    for f_idx, image_file in enumerate(image_files):
        label_path = os.path.join(base_dir, "labels", recording_name, image_file.replace(".jpg", ".json"))
        frame_labels = load_json(label_path)

        frame_metadata = [e for e in meta_data['images'] if e['file_name'] == image_file]
        if not frame_metadata:
            print(f"No meta entry found for {image_file}. Skipping frame.")
            all_bboxes_per_frame.append({})
            all_objects_per_frame.append({})
            camera_matrices.append(None)
            extrinsic_matrices.append(None)
            continue

        frame_metadata = frame_metadata[-1]
        translation = np.array(frame_metadata["extrinsic_translation"])
        rotation = np.array(frame_metadata["extrinsic_rotation"])
        extrinsic_matrix = convert_extrinsic_vectors_to_matrix(translation, rotation)
        extrinsic_matrices.append(extrinsic_matrix)

        intrinsic_matrix = np.array(meta_data["intrinsic_matrix"])
        intrinsic_matrices.append(intrinsic_matrix)

        camera_matrix = intrinsic_matrix @ extrinsic_matrix[:3, :]
        camera_matrices.append(camera_matrix)

        objects_in_frame = {}
        for ann in frame_labels['annotations']:
            obj = Object.deserialize(ann)
            objects_in_frame[obj.track_id] = obj

        all_objects_per_frame.append(objects_in_frame)
        bboxes_2d = {obj.track_id : obj.get_bbox(camera_matrix) for obj in objects_in_frame.values()}
        all_bboxes_per_frame.append(bboxes_2d)

    print(f"\tFound {sum([len(l) for l in all_bboxes_per_frame])} bounding boxes ")

    for frame_idx, image_file in enumerate(tqdm(image_files, desc='\tRunning SAM2 Image Predictor...', ncols=80)):
        img_path = os.path.join(recording_dir, image_file)
        with Image.open(img_path).convert("RGB") as pil_image:  # Use context manager
            np_image = np.array(pil_image)

        curr_bboxes = all_bboxes_per_frame[frame_idx]
        # Convert to xyxy
        boxes_xyxy = []
        for box in curr_bboxes.values():
            x, y, w, h = box
            boxes_xyxy.append([x, y, x + w, y + h])
        boxes_xyxy = np.asarray(boxes_xyxy)


        predictor.set_image(np_image)
        # Inference in a single batch
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                box=boxes_xyxy,
                multimask_output=False
            )

        # Store in video_segments_per_frame at the correct frame index
        camera_matrix = camera_matrices[frame_idx]
        extrinsic_matrix = extrinsic_matrices[frame_idx]
        f_objects = all_objects_per_frame[frame_idx]

        for obj, mask in zip(f_objects.values(), masks):
            track_id = obj.track_id

            if mask.shape[0] == 1:
                mask = mask[0]

            # Find bounding box coordinates
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue  # Skip empty masks

            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()

            # Crop the object and mask using the bounding box
            cropped_object = np_image[y_min:y_max + 1, x_min:x_max + 1]
            cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

            # Convert the cropped image to PIL format
            cropped_object_pil = Image.fromarray(cropped_object)

            # Create the alpha channel using the cropped mask
            alpha_channel = (cropped_mask * 255).astype(np.uint8)

            # Ensure the cropped object has 3 color channels (RGB)
            if cropped_object_pil.mode != 'RGB':
                cropped_object_pil = cropped_object_pil.convert('RGB')

            # Combine the RGB image and alpha channel to create an RGBA image
            cropped_object_with_alpha = Image.merge(
                'RGBA', (cropped_object_pil.split()[0],  # R
                        cropped_object_pil.split()[1],  # G
                        cropped_object_pil.split()[2],  # B
                        Image.fromarray(alpha_channel))  # A
            )


            # Prepare output folder structure
            obj_folder = os.path.join(base_output_dir, f"{track_id}")
            ## If track ID already exists, then create a new one
            if os.path.exists(obj_folder):
                track_id = int(track_id) + 1000
                obj_folder = os.path.join(base_output_dir, f"{track_id}")
                obj.track_id = track_id

            images_folder = os.path.join(obj_folder, "images")
            os.makedirs(images_folder, exist_ok=True)
            cropped_output_path = os.path.join(
                images_folder, f'{image_file.replace(".jpg", ".png")}'
            )
            cropped_object_with_alpha.save(cropped_output_path)

            # Prepare annotation data
            if track_id not in annotation_data:
                annotation_data[track_id] = {
                    "track_id": track_id,
                    "recording_name": recording_name,
                    "attribute_id": obj.att_id,
                    "category_id": obj.cat_id,
                    "intrinsic_matrix": meta_data["intrinsic_matrix"],
                    "frame_data": {}
                }

            annotation_data[track_id]["frame_data"][image_file] = {
                'translation' : obj.trans.tolist(),
                'rotation' : obj.rot.tolist(),
                'dimension' : obj.dim.tolist(),
                'bbox' : obj.get_bbox(camera_matrix).tolist(),
                'camera_matrix' : camera_matrix.tolist(),
                'extrinsic_matrix' : extrinsic_matrix.tolist(),
                'translation_in_camera_frame' : transform_to_camera_frame(obj.trans, extrinsic_matrix).tolist(),
                'rotation_in_camera_frame' : rotation_to_camera_frame(obj.rot, extrinsic_matrix).tolist()

            }
        gc.collect()
        
        for track_id, ann_dict in annotation_data.items():
            obj_folder = os.path.join(base_output_dir, f"{track_id}")
            json_path = os.path.join(obj_folder, "annotations.json")
            with open(json_path, 'w') as f:
                json.dump(ann_dict, f, indent=4)
    print(f"[INFO] Processing complete for recording: {recording_name}")
    print(f"[INFO] Outputs saved under: {base_output_dir}")


def build_sam2_image_predictor(model_config, checkpoint_path):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_model = build_sam2(model_config, checkpoint_path, device='cuda')
    return SAM2ImagePredictor(sam2_model)

def main(root_directory, checkpoint_path, model_config, validation_recordings=None):
    predictor = build_sam2_image_predictor(model_config, checkpoint_path)
    detection_model = None
    inference_state = None
    recordings_dir = os.path.join(root_directory, "images")

    if validation_recordings:
        with open(validation_recordings, 'r') as file:
            data = json.load(file)
        validation_recordings = set(data.get("folders", []))

    for recording_name in os.listdir(recordings_dir):
        if validation_recordings is not None:
            if recording_name in validation_recordings:
                continue
            
        recording_path = os.path.join(recordings_dir, recording_name)

        if not os.path.isdir(recording_path):
            continue
        if inference_state is not None:
            predictor.reset_state(inference_state)

        print(f"Processing recording: {recording_name}")
        process_recording(
            recording_dir=recording_path,
            inference_state=inference_state,
            predictor=predictor,
            detection_model=detection_model,
            output_database_dir=os.path.join(root_directory, "object_database"),
            storage_mode="ground_truth",
            batch_size=4  # can adjust your desired batch size here
        )

# def filter_recordings(recordings, json_file):
#     # Load JSON file
#     with open(json_file, 'r') as file:
#         data = json.load(file)
    
#     # Get the folder list from JSON
#     existing_folders = set(data.get("folders", []))
    
#     # Filter out recordings that exist in the JSON folder list
#     filtered_recordings = [rec for rec in recordings if rec not in existing_folders]
    
#     return filtered_recordings

if __name__ == "__main__":
    root_directory = "/home/stud/ukh/workspace/datasets/dev_dataset/"
    validation_recordings = "/home/stud/ukh/workspace/datasets/validation_recordings.json"
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    main(root_directory, checkpoint, model_cfg, validation_recordings)