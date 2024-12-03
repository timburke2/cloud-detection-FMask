import os
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants for FMask and Ground Truth categories
FM_NULL, FM_CLEAR, FM_CLOUD, FM_SHADOW, FM_SNOW, FM_WATER = 0, 1, 2, 3, 4, 5
GT_NULL, GT_SHADOW, GT_NO_CLOUD, GT_THIN_CLOUD, GT_CLOUD = 0, 64, 128, 192, 255

# Mapping from FMask values to Ground Truth grayscale system
FMASK_TO_GT_MAP = {
    FM_NULL: GT_NULL,
    FM_CLEAR: GT_NO_CLOUD,
    FM_CLOUD: GT_CLOUD,
    FM_SHADOW: GT_SHADOW,
    FM_SNOW: GT_NO_CLOUD,
    FM_WATER: GT_NO_CLOUD
}

def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration settings from a YAML file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def validate_directories(*paths: Path) -> None:
    """Ensure all provided directories exist."""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")


def convert_to_grayscale(img_file: Path, output_file: Path) -> None:
    """
    Converts an FMask image to a grayscale PNG using the Ground Truth grayscale system.

    Args:
        img_file (Path): Path to the input FMask image file.
        output_file (Path): Path to save the grayscale PNG.
    """
    try:
        # Open the FMask image and read the data
        with rasterio.open(img_file) as src:
            fmask_data = src.read(1)  # Read the first band
        
        # Apply the mapping to convert FMask values to Ground Truth grayscale
        remapped_data = np.vectorize(FMASK_TO_GT_MAP.get)(fmask_data)

        # Ensure the output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the remapped data as a grayscale PNG
        plt.imsave(output_file, remapped_data, cmap="gray", vmin=GT_NULL, vmax=GT_CLOUD)
        logging.info(f"Saved remapped grayscale image: {output_file}")
    except Exception as e:
        logging.error(f"Error converting {img_file} to grayscale: {e}")


def generate_difference_map(fmask_path: Path, gt_path: Path, output_path: Path) -> None:
    """Generates a difference map and saves as PNG."""
    try:
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
        logging.info(f"Saved difference map: {output_path}")
    except Exception as e:
        logging.error(f"Error generating difference map: {e}")


def process_scene(scene_id: str, raw_dir: Path, processed_dir: Path) -> None:
    """Processes a single scene."""
    raw_scene_dir = raw_dir / scene_id
    processed_scene_dir = processed_dir / scene_id

    try:
        # Convert FMask to grayscale
        fmask_file = raw_scene_dir / f"{scene_id}_fmask.img"
        if fmask_file.exists():
            convert_to_grayscale(fmask_file, processed_scene_dir / "fmask.png")
        else:
            logging.warning(f"FMask file not found for scene {scene_id}.")

        # Convert fixedmask to grayscale
        fixedmask_file = raw_scene_dir / f"{scene_id}_fixedmask.img"
        if fixedmask_file.exists():
            convert_to_grayscale(fixedmask_file, processed_scene_dir / "ground_truth.png")
        else:
            logging.warning(f"Fixed mask file not found for scene {scene_id}.")

        # Generate difference map
        if fmask_file.exists() and fixedmask_file.exists():
            generate_difference_map(
                fmask_path=fmask_file,
                gt_path=fixedmask_file,
                output_path=processed_scene_dir / "difference_map.png"
            )
    except Exception as e:
        logging.error(f"Error processing scene {scene_id}: {e}")


def process_all_scenes(raw_dir: Path, processed_dir: Path) -> None:
    """Processes all scenes in the raw directory."""
    scene_dirs = [scene for scene in raw_dir.iterdir() if scene.is_dir()]
    for scene_dir in tqdm(scene_dirs, desc="Processing Scenes", total=len(scene_dirs)):
        process_scene(scene_dir.name, raw_dir, processed_dir)


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)

    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])

    # Validate directories
    validate_directories(raw_dir, processed_dir)

    # Process all scenes
    process_all_scenes(raw_dir, processed_dir)
