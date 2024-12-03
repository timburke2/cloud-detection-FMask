import json
import numpy as np
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
from pathlib import Path
import yaml

# Constants for FMask and Ground Truth categories
FM_NULL, FM_CLEAR, FM_CLOUD, FM_SHADOW, FM_SNOW, FM_WATER = 0, 1, 2, 3, 4, 5
GT_NULL, GT_SHADOW, GT_NO_CLOUD, GT_THIN_CLOUD, GT_CLOUD = 0, 64, 128, 192, 255

# Utility function for safe division
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0

def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration settings from a YAML file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def generate_scene_metrics(raw_dir: Path, output_dir: Path) -> None:
    """Evaluates scenes and writes results to a JSON file."""
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    # Iterate over scene directories in raw_dir
    scene_dirs = [scene for scene in raw_dir.iterdir() if scene.is_dir()]
    results = []

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        print(f"Evaluating scene: {scene_id}")

        # Paths for FMask and Ground Truth .img files
        fmask_path = scene_dir / f"{scene_id}_fmask.img"
        gt_path = scene_dir / f"{scene_id}_fixedmask.img"

        # Check if files exist
        if not fmask_path.exists() or not gt_path.exists():
            print(f"Skipping scene {scene_id} due to missing files.")
            continue

        # Read FMask and Ground Truth data
        with rasterio.open(fmask_path) as fmsrc, rasterio.open(gt_path) as gtsrc:
            fmask = fmsrc.read(1)
            gtrue = gtsrc.read(1)

            thick_cloud_count = np.count_nonzero(gtrue == GT_CLOUD)
            clear_sky_count = np.count_nonzero(gtrue == GT_NO_CLOUD)
            cloud_shadow_count = np.count_nonzero(gtrue == GT_SHADOW)

            total = thick_cloud_count + clear_sky_count + cloud_shadow_count

            metrics = {}

            # Evaluate Thick Cloud Metrics
            if thick_cloud_count > 0:
                tp = np.count_nonzero((gtrue == GT_CLOUD) & (fmask == FM_CLOUD))
                fp = np.count_nonzero((gtrue != GT_CLOUD) & (fmask == FM_CLOUD))
                fn = np.count_nonzero((gtrue == GT_CLOUD) & (fmask != FM_CLOUD))

                precision = safe_divide(tp, (tp + fp))
                recall = safe_divide(tp, (tp + fn))
                f1 = 2 * safe_divide((precision * recall), (precision + recall))

                metrics["Cloud"] = {
                    "Proportion": round(thick_cloud_count / total, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1": round(f1, 4)
                }

            # Evaluate Clear Sky Metrics
            if clear_sky_count > 0:
                tp = np.count_nonzero((gtrue == GT_NO_CLOUD) & (fmask != FM_CLOUD))
                fp = np.count_nonzero((gtrue != GT_NO_CLOUD) & (fmask != FM_CLOUD))
                fn = np.count_nonzero((gtrue == GT_NO_CLOUD) & (fmask == FM_CLOUD))

                precision = safe_divide(tp, (tp + fp))
                recall = safe_divide(tp, (tp + fn))
                f1 = 2 * safe_divide((precision * recall), (precision + recall))

                metrics["Clear Sky"] = {
                    "Proportion": round(clear_sky_count / total, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1": round(f1, 4)
                }

            # Evaluate Cloud Shadow Metrics
            if cloud_shadow_count > 0:
                tp = np.count_nonzero((gtrue == GT_SHADOW) & (fmask == FM_SHADOW))
                fp = np.count_nonzero((gtrue != GT_SHADOW) & (fmask == FM_SHADOW))
                fn = np.count_nonzero((gtrue == GT_SHADOW) & (fmask != FM_SHADOW))

                precision = safe_divide(tp, (tp + fp))
                recall = safe_divide(tp, (tp + fn))
                f1 = 2 * safe_divide((precision * recall), (precision + recall))

                metrics["Shadow"] = {
                    "Proportion": round(cloud_shadow_count / total, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1": round(f1, 4)
                }

            # Append results for this scene
            results.append({
                "Scene": scene_id,
                "Metrics": metrics
            })

    # Write results to output JSON
    output_file = output_dir / "scene_metrics.json"
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)
    print(f"Evaluation results written to: {output_file}")

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)

    raw_dir = Path(config["paths"]["raw_dir"])
    output_dir = Path(config["paths"]["output_dir"])

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate scenes
    generate_scene_metrics(raw_dir, output_dir)
