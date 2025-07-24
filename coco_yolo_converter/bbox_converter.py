"""
COCO to YOLO Format Converter Module

This module provides functions to convert COCO JSON annotations to YOLO format
with optional train-test splitting capabilities.

Author: AI Assistant
Version: 1.0
"""

import json
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bounding box [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    All values normalized to 0-1 range.
    
    Args:
        bbox (list): COCO bounding box [x, y, width, height]
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
        
    Returns:
        list: YOLO format bounding box [x_center, y_center, width, height]
    """
    x, y, width, height = bbox
    
    # Convert to center coordinates
    x_center = x + width / 2
    y_center = y + height / 2
    
    # Normalize to 0-1 range
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]


def create_directory_structure(output_dir, use_split=True):
    """
    Create YOLO dataset directory structure.
    
    Args:
        output_dir (str): Output directory path
        use_split (bool): Whether to create train/val/test subdirectories
        
    Returns:
        Path: Path object of the output directory
    """
    output_path = Path(output_dir)
    
    if use_split:
        # Create train/val/test structure
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    else:
        # Create simple images/labels structure
        (output_path / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    return output_path


def split_dataset(image_ids, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        image_ids (list): List of image IDs
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio  
        test_ratio (float): Test set ratio
        
    Returns:
        dict: Dictionary with 'train', 'val', 'test' keys containing image ID lists
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Shuffle image IDs for random split
    shuffled_ids = image_ids.copy()
    random.shuffle(shuffled_ids)
    
    total_images = len(shuffled_ids)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    
    train_ids = shuffled_ids[:train_count]
    val_ids = shuffled_ids[train_count:train_count + val_count]
    test_ids = shuffled_ids[train_count + val_count:]
    
    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }


def copy_image_file(src_path, dst_path):
    """
    Copy image file, handling different possible locations.
    
    Args:
        src_path (Path): Source path
        dst_path (Path): Destination path
        
    Returns:
        bool: True if file was copied successfully, False otherwise
    """
    # Try different possible source paths
    possible_paths = [
        src_path,
        Path("images") / Path(src_path).name,
        Path(src_path).name
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            shutil.copy2(path, dst_path)
            return True
    
    return False


def create_dataset_yaml(output_path, class_names, use_split, images_dir):
    """
    Create dataset.yaml file for YOLO training.
    
    Args:
        output_path (Path): Output directory path
        class_names (dict): Dictionary mapping class indices to names
        use_split (bool): Whether dataset is split into train/val/test
        images_dir (str): Images directory path
        
    Returns:
        Path: Path to created dataset.yaml file
    """
    if use_split:
        yaml_content = {
            'path': str(output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
    else:
        yaml_content = {
            'path': str(output_path.absolute()),
            'train': 'images',
            'val': 'images',  # Use same for validation if no split
            'nc': len(class_names),
            'names': class_names
        }
    
    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def coco_to_yolo_converter(json_path, images_dir, output_dir, use_split=False, 
                          train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42, verbose=True):
    """
    Convert COCO JSON annotations to YOLO format with optional train-test splitting.
    
    Args:
        json_path (str): Path to COCO JSON file
        images_dir (str): Directory containing images  
        output_dir (str): Output directory for YOLO dataset
        use_split (bool): Whether to split into train/val/test sets
        train_ratio (float): Ratio for training set (default: 0.7)
        val_ratio (float): Ratio for validation set (default: 0.2) 
        test_ratio (float): Ratio for test set (default: 0.1)
        seed (int): Random seed for reproducible splits (default: 42)
        verbose (bool): Whether to print progress messages (default: True)
        
    Returns:
        dict: Dictionary containing:
            - 'output_path': Path to created YOLO dataset
            - 'yaml_path': Path to dataset.yaml file
            - 'stats': Dictionary with conversion statistics
            
    Raises:
        FileNotFoundError: If JSON file or images directory doesn't exist
        ValueError: If split ratios don't sum to 1.0
    """
    # Validate inputs
    if not Path(json_path).exists():
        raise FileNotFoundError(f"COCO JSON file not found: {json_path}")
    
    if not Path(images_dir).exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Set random seed for reproducible splits
    random.seed(seed)
    
    # Load COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    if verbose:
        print(f"Loading COCO dataset from {json_path}")
        print(f"Found {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    # Create directory structure
    output_path = create_directory_structure(output_dir, use_split)
    
    # Create category mapping (COCO ID -> YOLO class index)
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    class_names = {idx: cat['name'] for idx, cat in enumerate(categories)}
    
    if verbose:
        print(f"Categories mapping:")
        for coco_id, yolo_id in category_mapping.items():
            cat_name = next(c['name'] for c in categories if c['id'] == coco_id)
            print(f"  COCO ID {coco_id} ({cat_name}) -> YOLO class {yolo_id}")
    
    # Index images and annotations
    image_info = {img['id']: img for img in coco_data['images']}
    annotations_by_image = defaultdict(list)
    
    for ann in coco_data['annotations']:
        if ann.get('iscrowd', 0) == 0:  # Skip crowd annotations
            annotations_by_image[ann['image_id']].append(ann)
    
    # Filter images that have annotations
    image_ids_with_annotations = list(annotations_by_image.keys())
    if verbose:
        print(f"Images with annotations: {len(image_ids_with_annotations)}")
    
    # Split dataset if requested
    if use_split:
        splits = split_dataset(image_ids_with_annotations, train_ratio, val_ratio, test_ratio)
        if verbose:
            print(f"Dataset split:")
            for split_name, split_ids in splits.items():
                print(f"  {split_name}: {len(split_ids)} images")
    else:
        splits = {'all': image_ids_with_annotations}
    
    # Process each split
    successful_copies = 0
    failed_copies = 0
    total_annotations = 0
    
    for split_name, split_image_ids in splits.items():
        if verbose:
            print(f"\nProcessing {split_name} split...")
        
        for image_id in split_image_ids:
            img_info = image_info[image_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Determine paths
            if use_split:
                img_dst_dir = output_path / split_name / 'images'
                label_dst_dir = output_path / split_name / 'labels'
            else:
                img_dst_dir = output_path / 'images'
                label_dst_dir = output_path / 'labels'
            
            # Copy image file
            img_src_path = Path(images_dir) / Path(img_filename).name
            img_dst_path = img_dst_dir / Path(img_filename).name
            
            if copy_image_file(img_src_path, img_dst_path):
                successful_copies += 1
            else:
                if verbose:
                    print(f"Warning: Could not find image {img_filename}")
                failed_copies += 1
                continue
            
            # Create YOLO label file
            label_filename = Path(img_filename).stem + '.txt'
            label_dst_path = label_dst_dir / label_filename
            
            yolo_annotations = []
            for ann in annotations_by_image[image_id]:
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    category_id = ann['category_id']
                    yolo_class = category_mapping[category_id]
                    
                    # Convert to YOLO format
                    yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                    
                    # Format: class x_center y_center width height
                    yolo_line = f"{yolo_class} {' '.join([f'{coord:.6f}' for coord in yolo_bbox])}"
                    yolo_annotations.append(yolo_line)
                    total_annotations += 1
            
            # Write label file
            with open(label_dst_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    # Create dataset.yaml
    yaml_path = create_dataset_yaml(output_path, class_names, use_split, images_dir)
    
    # Prepare statistics
    stats = {
        'total_images': len(coco_data['images']),
        'images_with_annotations': len(image_ids_with_annotations),
        'successful_copies': successful_copies,
        'failed_copies': failed_copies,
        'total_annotations': total_annotations,
        'categories': len(categories),
        'category_mapping': category_mapping,
        'class_names': class_names
    }
    
    if verbose:
        print(f"\nConversion completed:")
        print(f"  Successfully processed: {successful_copies} images")
        print(f"  Failed to find: {failed_copies} images")
        print(f"  Total annotations converted: {total_annotations}")
        print(f"  Categories: {len(categories)}")
        print(f"\nYOLO dataset created at: {output_path}")
        print(f"Dataset configuration: {yaml_path}")
    
    return {
        'output_path': str(output_path),
        'yaml_path': str(yaml_path),
        'stats': stats
    }


# Convenience function for quick conversion
def quick_convert(json_path, images_dir, output_dir="yolo_dataset"):
    """
    Quick conversion with default parameters.
    
    Args:
        json_path (str): Path to COCO JSON file
        images_dir (str): Directory containing images
        output_dir (str): Output directory (default: "yolo_dataset")
        
    Returns:
        str: Path to created YOLO dataset
    """
    result = coco_to_yolo_converter(json_path, images_dir, output_dir)
    return result['output_path']


if __name__ == "__main__":
    # Example usage when script is run directly
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python coco_yolo_converter.py <json_path> <images_dir> [output_dir]")
        print("Example: python coco_yolo_converter.py annotations.json ./images ./yolo_dataset")
        sys.exit(1)
    
    json_file = sys.argv[1]
    images_directory = sys.argv[2]
    output_directory = sys.argv[3] if len(sys.argv) > 3 else "yolo_dataset"
    
    try:
        result = coco_to_yolo_converter(json_file, images_directory, output_directory)
        print(f"\n✅ Conversion successful!")
        print(f"Dataset ready for training: {result['output_path']}")
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        sys.exit(1)
