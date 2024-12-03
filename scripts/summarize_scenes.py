import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json
import yaml

def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration settings from a YAML file."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def generate_combined_plot(scene_id: str, scene_dir: Path, metrics: dict, output_dir: Path) -> None:
    """
    Creates a combined plot of natural color, FMask, ground truth, and difference map,
    along with precision, recall, and F1 metrics for clouds.

    Args:
        scene_id (str): The ID of the scene.
        scene_dir (Path): The directory containing scene-specific images.
        metrics (dict): The metrics dictionary for the scene.
        output_dir (Path): The directory where the combined plot will be saved.

    Returns:
        None
    """
    # Paths for the images
    paths = {
        "natural_color": scene_dir / "natural_color.png",
        "fmask": scene_dir / "fmask.png",
        "ground_truth": scene_dir / "ground_truth.png",
        "difference_map": scene_dir / "difference_map.png",
    }

    # Check if all required images exist
    if not all(path.exists() for path in paths.values()):
        print(f"Missing images for scene {scene_id}. Skipping.")
        return

    # Prepare output directory and file
    summary_dir = output_dir / "scene_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    combined_output_path = summary_dir / f"{scene_id}.png"

    # Extract cloud metrics
    cloud_metrics = metrics.get("Cloud", {})
    precision = cloud_metrics.get("Precision", "N/A")
    recall = cloud_metrics.get("Recall", "N/A")
    f1 = cloud_metrics.get("F1", "N/A")

    # Create the combined plot with a 2x2 grid for images and a metrics row
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.2])

    # Plot the 2x2 images
    axes = [
        fig.add_subplot(gs[0, 0]),  # Top-left
        fig.add_subplot(gs[0, 1]),  # Top-right
        fig.add_subplot(gs[1, 0]),  # Bottom-left
        fig.add_subplot(gs[1, 1]),  # Bottom-right
    ]

    images = [mpimg.imread(paths[key]) for key in paths]
    titles = ["Natural Color", "FMask Output", "Ground Truth", "Difference Map"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    # Add the metrics as a centered text box in the last row
    metrics_ax = fig.add_subplot(gs[2, :])  # Span the entire bottom row
    metric_text = (
        f"Cloud Metrics:\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}"
    )
    metrics_ax.text(
        0.5, 0.5, metric_text,
        fontsize=16, ha="center", va="center", wrap=True, weight="bold"
    )
    metrics_ax.axis("off")

    # Adjust layout and save
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(combined_output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Combined plot saved for scene {scene_id} at {combined_output_path}")


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    output_dir = Path(config["paths"]["output_dir"])

    # Load metrics from the JSON file
    json_file = output_dir / "scene_metrics.json"
    with open(json_file, "r", encoding="utf-8") as infile:
        results = json.load(infile)

    # Iterate through scenes and generate combined plots
    for result in results:
        scene_id = result["Scene"]
        metrics = result["Metrics"]
        scene_dir = processed_dir / scene_id
        generate_combined_plot(scene_id, scene_dir, metrics, output_dir)
