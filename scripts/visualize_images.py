import os
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import yaml


# Constants for FMask and Ground Truth categories
FM_NULL, FM_CLEAR, FM_CLOUD, FM_SHADOW, FM_SNOW, FM_WATER = 0, 1, 2, 3, 4, 5
GT_NULL, GT_SHADOW, GT_NO_CLOUD, GT_THIN_CLOUD, GT_CLOUD = 0, 64, 128, 192, 255


def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration settings from a YAML file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def convert_to_grayscale(img_file: Path, output_file: Path) -> None:
    """
    Converts an image to a grayscale PNG.

    Args:
        img_file (Path): Path to the input image file.
        output_file (Path): Path to save the grayscale PNG.
    """
    with rasterio.open(img_file) as src:
        data = src.read(1)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_file, data, cmap="gray", vmin=0, vmax=255)
    print(f"Saved grayscale image: {output_file}")


def generate_difference_map(fmask_path: Path, gt_path: Path, output_path: Path) -> None:
    """
    Generates a difference map and saves as PNG.

    Args:
        fmask_path (Path): Path to the FMask image file.
        gt_path (Path): Path to the ground truth image file.
        output_path (Path): Path to save the difference map PNG.
    """
    with rasterio.open(fmask_path) as fmask_src:
        fmask = fmask_src.read(1)
    with rasterio.open(gt_path) as gt_src:
        gtrue = gt_src.read(1)
    
    diff_map = np.full(fmask.shape, -2, dtype=np.int8)
    non_null_mask = (fmask != FM_NULL) & (gtrue != GT_NULL)
    diff_map[(fmask == FM_CLOUD) & (gtrue != GT_THIN_CLOUD) & (gtrue != GT_CLOUD) & non_null_mask] = -1
    diff_map[(fmask != FM_CLOUD) & ((gtrue == GT_THIN_CLOUD) | (gtrue == GT_CLOUD)) & non_null_mask] = 1
    diff_map[non_null_mask & ((fmask == FM_CLOUD) & ((gtrue == GT_THIN_CLOUD) | (gtrue == GT_CLOUD)))] = 0
    
    cmap = ListedColormap(['black', 'red', 'green', 'blue'])
    norm = BoundaryNorm([-2, -1, 0, 1, 2], cmap.N)
    plt.imshow(diff_map, cmap=cmap, norm=norm)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved difference map: {output_path}")


def process_scene(scene_id: str, raw_dir: Path, processed_dir: Path) -> None:
    """
    Processes a single scene: generates grayscale FMask, fixedmask images, and a difference map.

    Args:
        scene_id (str): ID of the scene to process.
        raw_dir (Path): Path to the raw directory containing FMask and fixedmask images.
        processed_dir (Path): Path to the processed directory to save outputs.
    """
    raw_scene_dir = raw_dir / scene_id
    processed_scene_dir = processed_dir / scene_id

    # Convert FMask to grayscale
    fmask_file = raw_scene_dir / f"{scene_id}_fmask.img"
    if fmask_file.exists():
        convert_to_grayscale(fmask_file, processed_scene_dir / "fmask.png")
    else:
        print(f"FMask file not found for scene {scene_id}. Skipping.")

    # Convert fixedmask to grayscale
    fixedmask_file = raw_scene_dir / f"{scene_id}_fixedmask.img"
    if fixedmask_file.exists():
        convert_to_grayscale(fixedmask_file, processed_scene_dir / "ground_truth.png")
    else:
        print(f"Fixed mask file not found for scene {scene_id}. Skipping.")

    # Generate difference map
    if fmask_file.exists() and fixedmask_file.exists():
        generate_difference_map(
            fmask_path=fmask_file,
            gt_path=fixedmask_file,
            output_path=processed_scene_dir / "difference_map.png"
        )
    else:
        print(f"Cannot generate difference map for scene {scene_id} due to missing files.")


def process_all_scenes(raw_dir: Path, processed_dir: Path) -> None:
    """
    Processes all scenes in the raw directory.

    Args:
        raw_dir (Path): Path to the raw directory containing scenes.
        processed_dir (Path): Path to the processed directory to save outputs.
    """
    for scene_dir in tqdm(raw_dir.iterdir(), desc="Processing Scenes"):
        if scene_dir.is_dir():
            process_scene(scene_dir.name, raw_dir, processed_dir)


if __name__ == "__main__":
    # Load configuration from config.yaml
    config = load_config("config.yaml")

    raw_dir = Path(config["paths"]["raw_dir"])          # Raw input directory
    processed_dir = Path(config["paths"]["processed_dir"])  # Processed output directory

    # Process all scenes
    process_all_scenes(raw_dir, processed_dir)
