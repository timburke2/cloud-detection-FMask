import json
import numpy as np
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
from pathlib import Path
import pandas as pd
import yaml
import matplotlib.pyplot as plt

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

def calculate_scene_metrics(raw_dir: Path, output_dir: Path) -> None:
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


def generate_metrics_table(json_file: Path, output_dir: Path) -> None:
    """
    Calculate and save the mean proportion, precision, recall, and F1 for each type across all scenes.

    Args:
        json_file (Path): Path to the JSON file containing evaluation data.
        output_dir (Path): Path to the directory where the output PNG will be saved.

    Returns:
        None
    """
    # Load the evaluation data
    with open(json_file, "r", encoding="utf-8") as infile:
        results = json.load(infile)

    # Transform data into a DataFrame
    data = []
    for result in results:
        scene = result["Scene"]
        for metric_type, metrics in result["Metrics"].items():
            data.append({
                "Scene": scene,
                "Type": metric_type,
                "Proportion": metrics["Proportion"],
                "Precision": float(metrics["Precision"]) if metrics["Precision"] != "N/A" else np.nan,
                "Recall": float(metrics["Recall"]) if metrics["Recall"] != "N/A" else np.nan,
                "F1": float(metrics["F1"]) if metrics["F1"] != "N/A" else np.nan
            })

    df = pd.DataFrame(data)

    # Calculate the mean metrics for each type
    summary_table = df.groupby("Type").agg(
        Mean_Proportion=("Proportion", "mean"),
        Mean_Precision=("Precision", "mean"),
        Mean_Recall=("Recall", "mean"),
        Mean_F1=("F1", "mean")
    ).reset_index()

    # Plot the table as an image
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust size as needed
    ax.axis("tight")
    ax.axis("off")

    # Create the table and render it on the plot
    table = plt.table(
        cellText=summary_table.values,
        colLabels=summary_table.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(summary_table.columns))))

    # Save the table as a PNG in the output directory
    output_file = output_dir / "summary_metrics_table.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Summary metrics table saved to: {output_file}")


def generate_metrics_plot(json_file: Path, output_dir: Path) -> None:
    """
    Generates boxplots for Precision, Recall, and F1 scores across metric types.

    Args:
        json_file (Path): Path to the JSON file containing evaluation data.
        output_dir (Path): Path to the directory where the boxplots will be saved.

    Returns:
        None
    """
    # Load evaluation data
    with open(json_file, "r", encoding="utf-8") as infile:
        results = json.load(infile)

    # Transform data into a DataFrame
    data = []
    for result in results:
        scene = result["Scene"]
        for metric_type, metrics in result["Metrics"].items():
            data.append({
                "Scene": scene,
                "Type": metric_type,
                "Proportion": metrics["Proportion"],
                "Precision": float(metrics["Precision"]) if metrics["Precision"] != "N/A" else np.nan,
                "Recall": float(metrics["Recall"]) if metrics["Recall"] != "N/A" else np.nan,
                "F1": float(metrics["F1"]) if metrics["F1"] != "N/A" else np.nan
            })

    df = pd.DataFrame(data)

    # List of metrics to plot
    metrics = ["Precision", "Recall", "F1"]

    for metric in metrics:
        # Generate a boxplot for each metric
        plt.figure(figsize=(8, 6))
        df.boxplot(column=metric, by="Type", grid=False)
        plt.title(f"{metric} Score Distribution by Type")
        plt.suptitle("")  # Suppress default title
        plt.ylabel(f"{metric} Score")
        plt.xlabel("Type")

        # Save the boxplot
        output_file = output_dir / f"{metric.lower()}_boxplot.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"{metric} boxplot saved to: {output_file}")


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)

    raw_dir = Path(config["paths"]["raw_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    json_file = Path(config["paths"]["output_dir"]) / "scene_metrics.json"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate scenes
    calculate_scene_metrics(raw_dir, output_dir)

    generate_metrics_table(json_file, output_dir)

    generate_metrics_plot(json_file, output_dir)
