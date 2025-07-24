#!/usr/bin/env python3
"""
seg_converter.py

Convert COCO JSON segmentation annotations to YOLOv8 segmentation format.

Functions:
- convert_polygon_to_yolo()
- create_structure()
- split_images()
- copy_image_file()
- write_labels()
- write_data_yaml()
- coco_to_yolo_segmentation()
- coco_to_yolo_converter()
- quick_convert()

Usage (CLI):
    python seg_converter.py \
      --input annotations.json \
      --images images/ \
      --output yolo_dataset/ \
      [--use-split] \
      [--train-ratio 0.7] \
      [--val-ratio 0.2] \
      [--test-ratio 0.1] \
      [--no-shuffle] \
      [--seed 42] \
      [--quiet]
"""

import json
import os
import shutil
import random
from pathlib import Path
from typing import List, Dict
import yaml

def convert_polygon_to_yolo(
    segmentation: List[float],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Normalize a polygon segmentation to YOLO format.
    """
    yolo_coords = []
    for i in range(0, len(segmentation), 2):
        x = segmentation[i] / img_width
        y = segmentation[i+1] / img_height
        yolo_coords.extend([round(x, 6), round(y, 6)])
    return yolo_coords

def create_dataset_structure(
    output_dir: str,
    use_split: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
) -> Dict[str, Path]:
    """
    Create dataset folders. If use_split=False, all splits point to train.
    """
    base = Path(output_dir)
    paths = {}
    # Train
    paths['train_imgs'] = base / 'images' / 'train'
    paths['train_lbls'] = base / 'labels' / 'train'
    paths['train_imgs'].mkdir(parents=True, exist_ok=True)
    paths['train_lbls'].mkdir(parents=True, exist_ok=True)

    # Val
    if use_split and val_ratio > 0:
        paths['val_imgs'] = base / 'images' / 'val'
        paths['val_lbls'] = base / 'labels' / 'val'
        paths['val_imgs'].mkdir(parents=True, exist_ok=True)
        paths['val_lbls'].mkdir(parents=True, exist_ok=True)
    else:
        paths['val_imgs'] = paths['train_imgs']
        paths['val_lbls'] = paths['train_lbls']

    # Test
    if use_split and test_ratio > 0:
        paths['test_imgs'] = base / 'images' / 'test'
        paths['test_lbls'] = base / 'labels' / 'test'
        paths['test_imgs'].mkdir(parents=True, exist_ok=True)
        paths['test_lbls'].mkdir(parents=True, exist_ok=True)
    else:
        paths['test_imgs'] = paths['train_imgs']
        paths['test_lbls'] = paths['train_lbls']

    return paths

def split_images(
    image_list: List[Dict],
    use_split: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool,
    seed: int
) -> (List[Dict], List[Dict], List[Dict]):
    """
    Split images into train/val/test lists.
    """
    imgs = image_list.copy()
    if shuffle:
        random.seed(seed)
        random.shuffle(imgs)
    if use_split:
        n = len(imgs)
        t_end = int(n * train_ratio)
        v_end = t_end + int(n * val_ratio)
        return imgs[:t_end], imgs[t_end:v_end], imgs[v_end:]
    else:
        return imgs, [], []

def copy_image_file(src: Path, dst: Path) -> bool:
    """
    Copy image if exists; return True if copied.
    """
    if src.exists():
        shutil.copy2(src, dst)
        return True
    alt = Path('images') / src.name
    if alt.exists():
        shutil.copy2(alt, dst)
        return True
    return False

def write_labels(
    anns: List[Dict],
    img_info: Dict,
    category_map: Dict[int,int],
    paths: Dict[str, Path],
    split: str
) -> int:
    """
    Write YOLO segmentation labels for one image. Returns 1 if written.
    """
    width, height = img_info['width'], img_info['height']
    lines = []
    for ann in anns:
        if ann.get('iscrowd', 0) == 1:
            continue
        seg = ann.get('segmentation', [])
        if not seg or not seg[0] or len(seg[0]) < 6:
            continue
        yolo_poly = convert_polygon_to_yolo(seg[0], width, height)
        cls_id = category_map[ann['category_id']]
        lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in yolo_poly))
    if not lines:
        return 0

    img_name = Path(img_info['file_name']).name
    src_img = Path(paths['root']) / img_name
    dst_img = paths[f'{split}_imgs'] / img_name
    copy_image_file(src_img, dst_img)

    label_file = paths[f'{split}_lbls'] / Path(img_name).with_suffix('.txt').name
    label_file.write_text("\n".join(lines) + "\n")
    return 1

def write_data_yaml(
    output_dir: str,
    names: List[str],
    use_split: bool
) -> None:
    """
    Write data.yaml for YOLO.
    """
    cfg = {
        'path': str(Path(output_dir).resolve()),
        'train': 'images/train',
        'val':   'images/val' if use_split else 'images/train',
        'test':  'images/test' if use_split else 'images/train',
        'nc':    len(names),
        'names': names
    }
    with open(Path(output_dir) / 'data.yaml', 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)

def coco_to_yolo_segmentation(
    coco_json: str,
    images_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    shuffle: bool = True
) -> Dict:
    """
    Legacy single train/val split support.
    """
    return coco_to_yolo_converter(
        json_path=coco_json,
        images_dir=images_dir,
        output_dir=output_dir,
        use_split=False,
        train_ratio=train_split,
        val_ratio=1 - train_split,
        test_ratio=0.0,
        seed=42,
        shuffle=shuffle,
        verbose=True
    )

def coco_to_yolo_converter(
    json_path: str,
    images_dir: str,
    output_dir: str,
    use_split: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Main conversion function.
    """
    # Load COCO
    coco = json.load(open(json_path, 'r'))
    imgs = coco['images']
    cats = coco['categories']
    anns = coco['annotations']

    # Map category IDs to contiguous class indices
    category_map = {c['id']: i for i, c in enumerate(sorted(cats, key=lambda x: x['id']))}
    names = [c['name'] for c in sorted(cats, key=lambda x: x['id'])]

    # Prepare output directories
    paths = create_dataset_structure(output_dir, use_split, train_ratio, val_ratio, test_ratio)
    paths['root'] = images_dir

    # Split images
    train_imgs, val_imgs, test_imgs = split_images(
        imgs, use_split, train_ratio, val_ratio, test_ratio, shuffle, seed
    )

    # Index annotations by image_id
    anns_map: Dict[int, List[Dict]] = {}
    for a in anns:
        anns_map.setdefault(a['image_id'], []).append(a)

    # Process each split
    stats = {'train': 0, 'val': 0, 'test': 0}
    for split, img_list in (('train', train_imgs), ('val', val_imgs), ('test', test_imgs)):
        for img in img_list:
            stats[split] += write_labels(
                anns_map.get(img['id'], []),
                img,
                category_map,
                paths,
                split
            )

    # Write data.yaml
    write_data_yaml(output_dir, names, use_split)

    if verbose:
        print(f"Conversion complete. Stats: {stats}")
    return {'stats': stats, 'output_dir': output_dir}

def quick_convert(
    coco_json: str,
    images_dir: str,
    output_dir: str = "yolo_dataset"
) -> str:
    """
    Shortcut for default conversion (train only).
    """
    res = coco_to_yolo_converter(
        json_path=coco_json,
        images_dir=images_dir,
        output_dir=output_dir,
        use_split=False,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        shuffle=True,
        verbose=False
    )
    return res['output_dir']

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="COCO → YOLOv8 Segmentation Converter")
    parser.add_argument('-i', '--input',    required=True, help="Path to COCO JSON file")
    parser.add_argument('-m', '--images',   required=True, help="Directory of source images")
    parser.add_argument('-o', '--output',   required=True, help="Output YOLO dataset directory")
    parser.add_argument('--use-split',      action='store_true',        help="Enable train/val/test split")
    parser.add_argument('--train-ratio',    type=float, default=0.7,     help="Train split fraction")
    parser.add_argument('--val-ratio',      type=float, default=0.2,     help="Validation split fraction")
    parser.add_argument('--test-ratio',     type=float, default=0.1,     help="Test split fraction")
    parser.add_argument('--seed',           type=int,   default=42,      help="Random seed for shuffling")
    parser.add_argument('--no-shuffle',     dest='shuffle', action='store_false', help="Disable shuffling")
    parser.add_argument('--quiet',          dest='verbose', action='store_false', help="Suppress console output")
    args = parser.parse_args()

    coco_to_yolo_converter(
        json_path=args.input,
        images_dir=args.images,
        output_dir=args.output,
        use_split=args.use_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle=args.shuffle,
        verbose=args.verbose
    )
