import json
import os
import cv2
import yaml
import random
from tqdm import tqdm
import math

def convert_to_yolo_segmentation(annotation_path, videos_dir, use_split=True, split_ratio=(0.7, 0.2, 0.1), output_dir="yolo_dataset"):
    """
    Converts a JSON annotation file to the YOLO segmentation format.

    This function reads a JSON file with polygon annotations, extracts the annotated
    frames from the corresponding videos, and saves them in YOLO segmentation format.
    It creates .txt label files and a data.yaml file for model training.

    Args:
        annotation_path (str): The path to the input JSON annotation file.
        videos_dir (str): The path to the directory containing the video files.
        use_split (bool): If True, splits data into 'train', 'val', and 'test' sets.
                          If False, saves all data into a single 'train' set.
        split_ratio (tuple): A tuple of 3 floats (train, val, test) that must sum to 1.
                             Only used if use_split is True.
        output_dir (str): The name of the directory to save the output dataset.
    """
    # --- 1. Validate Arguments and Create Directories ---
    if use_split:
        if not (isinstance(split_ratio, (list, tuple)) and len(split_ratio) == 3 and math.isclose(sum(split_ratio), 1.0)):
            raise ValueError("If 'use_split' is True, 'split_ratio' must be a list or tuple of 3 numbers that sum to 1.")
        print(f"Using split ratio: Train={split_ratio[0]}, Val={split_ratio[1]}, Test={split_ratio[2]}")
        split_names = ['train', 'val', 'test']
    else:
        print("Not splitting data. All files will be in the 'train' directory.")
        split_names = ['train']

    print(f"Creating output directory structure in: {output_dir}")
    images_base = os.path.join(output_dir, "images")
    labels_base = os.path.join(output_dir, "labels")

    for name in split_names:
        os.makedirs(os.path.join(images_base, name), exist_ok=True)
        os.makedirs(os.path.join(labels_base, name), exist_ok=True)

    # --- 2. Load JSON and Extract Class Names ---
    print("Loading JSON annotation file...")
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    class_names = set()
    for item in data:
        if "latest_answer" in item:
            for answer in item["latest_answer"]:
                class_names.add(answer["question_name"])

    if not class_names:
        print("Error: No class names found in the JSON file. Exiting.")
        return

    class_names = sorted(list(class_names))
    class_map = {name: i for i, name in enumerate(class_names)}
    print(f"Found classes: {class_names}")

    # --- 3. Process Annotations and Collect Frame Information ---
    all_frames_info = []
    print("Aggregating annotations from JSON...")
    for item in tqdm(data, desc="Processing files in JSON"):
        video_name = item.get("file_name")
        video_path = os.path.join(videos_dir, video_name)

        if not os.path.exists(video_path):
            print(f"Warning: Video '{video_name}' not found at '{video_path}'. Skipping.")
            continue

        width = item["file_metadata"]["width"]
        height = item["file_metadata"]["height"]

        frame_annotations = {}
        for answer_section in item.get("latest_answer", []):
            class_name = answer_section["question_name"]
            class_id = class_map[class_name]

            for annotation in answer_section.get("answer", []):
                for frame_num_str, frame_data in annotation.get("frames", {}).items():
                    # Skip frames with empty answers to save storage
                    if not frame_data.get("answer"):
                        continue

                    frame_num = int(frame_num_str)
                    if frame_num not in frame_annotations:
                        frame_annotations[frame_num] = []

                    polygon = frame_data["answer"]
                    normalized_points = []
                    for point in polygon:
                        norm_x = round(point['x'] / width, 6)
                        norm_y = round(point['y'] / height, 6)
                        normalized_points.extend([norm_x, norm_y])

                    frame_annotations[frame_num].append(f"{class_id} " + " ".join(map(str, normalized_points)))

        for frame_num, annotations in frame_annotations.items():
            all_frames_info.append({
                "video_path": video_path,
                "frame_num": frame_num,
                "annotations": annotations,
                "base_name": f"{os.path.splitext(video_name)[0]}_frame_{frame_num}"
            })

    # --- 4. Split Data and Save Files ---
    if use_split:
        random.shuffle(all_frames_info)
        train_ratio, val_ratio, _ = split_ratio
        total_frames = len(all_frames_info)

        split_1 = int(total_frames * train_ratio)
        split_2 = int(total_frames * (train_ratio + val_ratio))

        train_frames = all_frames_info[:split_1]
        val_frames = all_frames_info[split_1:split_2]
        test_frames = all_frames_info[split_2:]

        sets = {"train": train_frames, "val": val_frames, "test": test_frames}
    else:
        sets = {"train": all_frames_info}

    print("\nExtracting frames and saving to YOLO format...")
    for set_name, frames in sets.items():
        if not frames:
            print(f"No frames to process for the '{set_name}' set.")
            continue

        print(f"Processing {set_name} set ({len(frames)} frames)...")
        cap = None
        current_video_path = None

        frames.sort(key=lambda x: (x['video_path'], x['frame_num']))

        for info in tqdm(frames, desc=f"Exporting {set_name} set"):
            if info['video_path'] != current_video_path:
                if cap:
                    cap.release()
                cap = cv2.VideoCapture(info['video_path'])
                current_video_path = info['video_path']

            cap.set(cv2.CAP_PROP_POS_FRAMES, info['frame_num'])
            ret, frame_img = cap.read()

            if ret:
                image_path = os.path.join(images_base, set_name, f"{info['base_name']}.jpg")
                label_path = os.path.join(labels_base, set_name, f"{info['base_name']}.txt")
                cv2.imwrite(image_path, frame_img)
                with open(label_path, 'w') as f:
                    f.write("\n".join(info['annotations']))

        if cap:
            cap.release()

    # --- 5. Create data.yaml File ---
    print("\nCreating data.yaml file...")
    yaml_path = os.path.join(output_dir, "data.yaml")
    
    # Define paths for the YAML file, using relative paths for portability
    train_path = os.path.join("images", "train")
    val_path = os.path.join("images", "val") if use_split else train_path
    test_path = os.path.join("images", "test") if use_split else train_path

    yaml_data = {
        'path': f'../{output_dir}',  # Path from YOLOv8 project root to dataset
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(class_names),
        'names': class_names
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print("\nConversion complete! ✨")
    print(f"Dataset saved to: {os.path.abspath(output_dir)}")
    print(f"YAML file saved to: {os.path.abspath(yaml_path)}")


# --- Example of how to run the function ---
if __name__ == '__main__':
    # Define the paths based on your file structure
    ANNOTATION_FILE = "annotations.json"
    VIDEOS_DIRECTORY = "videos"
    OUTPUT_DATASET_DIR = "dataset_final"

    # Ensure the videos directory exists
    if not os.path.exists(VIDEOS_DIRECTORY):
        os.makedirs(VIDEOS_DIRECTORY)
        print(f"Created '{VIDEOS_DIRECTORY}' directory. Please add your video files there.")
        # You might want to exit here if videos are required for a real run
        # exit() 

    # Call the function with a custom split ratio
    convert_to_yolo_segmentation(
        annotation_path=ANNOTATION_FILE,
        videos_dir=VIDEOS_DIRECTORY,
        use_split=True,
        split_ratio=(0.7, 0.2, 0.1),  # 70% train, 20% val, 10% test
        output_dir=OUTPUT_DATASET_DIR
    )
