import subprocess
import os
from pathlib import Path
from tqdm import tqdm
import yaml
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def load_config(config_file: str = "config.yaml") -> dict:
    """
    Loads configuration settings from a YAML file.

    Args:
        config_file (str): Path to the YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    config_path = Path(config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file {config_file} not found.")
    
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def generate_fmask(input_dir: Path, output_dir: Path) -> None:
    """
    Generates FMask files for each scene in the input directory.

    Args:
        input_dir (Path): Path to the directory containing scene subdirectories.
        output_dir (Path): Path to the directory where FMask output files will be saved.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for scenedir in tqdm(input_dir.iterdir(), desc="Generating FMasks"):
        if not scenedir.is_dir():
            continue

        output_file = output_dir / f"{scenedir.name}_fmask.img"

        if output_file.exists():
            print(f"FMask for {scenedir.name} already exists. Skipping.")
            continue

        try:
            subprocess.run(
                ["fmask_usgsLandsatStacked", "-o", str(output_file), "--scenedir", str(scenedir)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"FMask generated for {scenedir.name}")
        except subprocess.CalledProcessError:
            print(f"Error generating FMask for {scenedir.name}")


def move_fixedmask(input_dir: Path, output_dir: Path) -> None:
    """
    Moves fixedmask.img files into the raw output directory.

    Args:
        input_dir (Path): Path to the directory containing scene subdirectories.
        output_dir (Path): Path to the directory where fixedmask.img files will be moved.

    Returns:
        None
    """
    for scenedir in tqdm(input_dir.iterdir(), desc="Moving Fixed Masks"):
        if not scenedir.is_dir():
            continue

        fixedmask_file = scenedir / f"{scenedir.name}_fixedmask.img"
        if not fixedmask_file.exists():
            print(f"Fixed mask for {scenedir.name} not found. Skipping.")
            continue

        output_scene_dir = output_dir / scenedir.name
        output_scene_dir.mkdir(parents=True, exist_ok=True)

        destination = output_scene_dir / f"{scenedir.name}_fixedmask.img"
        fixedmask_file.rename(destination)
        print(f"Moved fixed mask for {scenedir.name} to {destination}")


def enhance_contrast(band: np.ndarray) -> np.ndarray:
    """
    Enhances the contrast of a band image using the 2nd and 98th percentiles.

    Args:
        band (np.ndarray): A 2D array representing the band data.

    Returns:
        np.ndarray: A normalized 2D array with enhanced contrast.
    """
    p2, p98 = np.percentile(band, (2, 98))
    return np.clip((band - p2) / (p98 - p2), 0, 1)


def create_natural_color(input_dir: Path, output_dir: Path) -> None:
    """
    Creates natural color composites for each scene.

    Args:
        input_dir (Path): Path to the directory containing scene subdirectories with band data.
        output_dir (Path): Path to the directory where natural color composites will be saved.

    Returns:
        None
    """
    for scenedir in tqdm(input_dir.iterdir(), desc="Creating Natural Color Composites"):
        if not scenedir.is_dir():
            continue

        scene_id = scenedir.name
        band_paths = [scenedir / f"{scene_id}_B{i}.TIF" for i in (4, 3, 2)]
        if not all(path.exists() for path in band_paths):
            print(f"Missing bands for {scene_id}. Skipping.")
            continue

        try:
            # Read and enhance contrast for each band.
            bands = [
                enhance_contrast(rasterio.open(str(path)).read(1).astype(float))
                for path in band_paths
            ]
            rgb_image = np.dstack(bands)
            
            # Save the natural color composite.
            output_scene_dir = output_dir / scene_id
            output_scene_dir.mkdir(parents=True, exist_ok=True)
            plt.imsave(output_scene_dir / "natural_color.png", rgb_image)
            print(f"Natural color composite saved for {scene_id}.")
        except Exception as e:
            print(f"Error creating natural color composite for {scene_id}: {e}")


if __name__ == "__main__":
    config = load_config("config.yaml")
    
    dataset_dir = Path(config["paths"]["dataset_dir"])
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    
    generate_fmask(dataset_dir, raw_dir)
    
    move_fixedmask(dataset_dir, raw_dir)
    
    create_natural_color(dataset_dir, processed_dir)
