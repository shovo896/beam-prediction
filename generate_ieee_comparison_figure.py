from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

modalities = [
    "Position",
    "Position +\nHeight",
    "Position + Height\n+ Distance",
    "Image",
]
present_results = {
    "Top-1": [57.5944, 69.7103, 74.1879, 88.3231],
    "Top-2": [82.0018, 89.0255, 89.9912, 97.8051],
    "Top-3": [91.9227, 95.1712, 96.0492, 99.5610],
    "Top-5": [97.8051, 99.0342, 98.9464, 99.8244],
}

direct_metrics = ["Image\nTop-1", "Image\nTop-3", "Image\nTop-5", "Position\nTop-1"]
reference_results = [86.32, 99.41, 99.69, 59.00]
present_direct_results = [88.3231, 99.5610, 99.8244, 57.5944]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def save_figure(fig, name):
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def label_bars(ax, bars, *, inside=False, approximate_index=None):
    for index, bar in enumerate(bars):
        value = bar.get_height()
        prefix = r"$\sim$" if index == approximate_index else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 2.0 if inside else value + 1.0,
            f"{prefix}{value:.1f}",
            ha="center",
            va="top" if inside else "bottom",
            fontsize=6,
            color="white" if inside else "black",
            fontweight="bold",
        )


# Figure 1: complete findings and direct reference-paper comparison.
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), constrained_layout=True)

ax = axes[0]
x = np.arange(len(modalities))
width = 0.19
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"]
hatches = ["", "//", "\\\\", "xx"]

for index, (metric, values) in enumerate(present_results.items()):
    ax.bar(
        x + (index - 1.5) * width,
        values,
        width,
        label=metric,
        color=colors[index],
        edgecolor="black",
        linewidth=0.4,
        hatch=hatches[index],
    )

ax.set_title("(a) Present implementation")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(modalities)
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.8)
ax.legend(ncol=2, loc="lower left")

ax = axes[1]
x = np.arange(len(direct_metrics))
width = 0.36
reference_bars = ax.bar(
    x - width / 2,
    reference_results,
    width,
    label="Reference [1]",
    color="#bdbdbd",
    edgecolor="black",
    linewidth=0.5,
    hatch="//",
)
present_bars = ax.bar(
    x + width / 2,
    present_direct_results,
    width,
    label="Present work",
    color="#1f77b4",
    edgecolor="black",
    linewidth=0.5,
)

ax.set_title("(b) Direct comparison with [1]")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(direct_metrics)
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.8)
ax.legend(loc="lower left")
label_bars(ax, reference_bars, inside=True, approximate_index=3)
label_bars(ax, present_bars, inside=True)
save_figure(fig, "ieee_findings_comparison")


# Figure 2: final top-k accuracy for every implemented modality.
fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.3), constrained_layout=True)
topk_labels = list(present_results)
for ax, modality_index, title in zip(axes.flat, range(4), modalities):
    values = [present_results[key][modality_index] for key in topk_labels]
    bars = ax.bar(
        topk_labels,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        hatch=hatches,
    )
    ax.set_title(title.replace("\n", " "))
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.8)
    label_bars(ax, bars, inside=True)
save_figure(fig, "all_model_test_accuracy")


# Figure 3: contribution of each additional sensing modality to Top-1.
fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)
top1_values = present_results["Top-1"]
bars = ax.barh(
    modalities,
    top1_values,
    color=["#9e9e9e", "#ff7f0e", "#2ca02c", "#1f77b4"],
    edgecolor="black",
    linewidth=0.5,
)
ax.set_title("Top-1 Accuracy by Modality")
ax.set_xlabel("Accuracy (%)")
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 20))
ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.8)
for bar, value in zip(bars, top1_values):
    ax.text(
        value - 1.5,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}%",
        ha="right",
        va="center",
        color="white",
        fontsize=7,
        fontweight="bold",
    )
save_figure(fig, "top1_modality_comparison")


# Figure 4: image-model refinement and direct reference comparison.
fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)
image_metrics = ["Top-1", "Top-3", "Top-5"]
reference_image = [86.32, 99.41, 99.69]
initial_image = [87.6207, 99.5610, 99.8244]
refined_image = [88.3231, 99.5610, 99.8244]
x = np.arange(len(image_metrics))
width = 0.25
ax.bar(
    x - width,
    reference_image,
    width,
    label="Reference [1]",
    color="#bdbdbd",
    edgecolor="black",
    linewidth=0.5,
    hatch="//",
)
ax.bar(
    x,
    initial_image,
    width,
    label="Initial model",
    color="#ff7f0e",
    edgecolor="black",
    linewidth=0.5,
)
bars = ax.bar(
    x + width,
    refined_image,
    width,
    label="Refined model",
    color="#1f77b4",
    edgecolor="black",
    linewidth=0.5,
)
ax.set_title("Image Model Accuracy")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(image_metrics)
ax.set_ylim(80, 102)
ax.set_yticks(np.arange(80, 101, 5))
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.8)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=3,
    frameon=False,
)
label_bars(ax, bars, inside=True)
save_figure(fig, "image_model_refinement")
