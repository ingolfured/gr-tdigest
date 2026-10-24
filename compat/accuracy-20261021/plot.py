"""Render the README accuracy figure from the measured summary (requires matplotlib)."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
summary = json.loads((folder / "summary.json").read_text())
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "gr-tdigest-default-accuracy-20261021",
    }
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor("#f7fafc")
for axis, key, title in zip(axes, ["all", "recorded_data"], ["All 396 profiles", "108 recorded-data profiles"]):
    group = summary["aggregates"][key]
    metric = group["metrics"]["rank_mean"]
    values = [metric["original_mean"] * 100, metric["modern_mean"] * 100]
    axis.set_facecolor("#f7fafc")
    axis.barh([1, 0], values, color=["#94a3b8", "#0d9488"], height=0.42)
    axis.set_yticks([1, 0], ["Original", "Modern"])
    axis.set_xlim(0, values[0] * 1.3)
    axis.set_ylim(-0.6, 1.7)
    for y, value in zip([1, 0], values):
        axis.text(value + values[0] * 0.03, y, f"{value:.5f}", va="center", color="#102a43", fontsize=11)
    axis.set_title(title, loc="left", color="#102a43", fontsize=14, fontweight="bold", pad=17)
    axis.text(0, 1.63, f"{metric['reduction_percent']:.1f}% lower average position error", color="#0f766e", fontsize=11)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#cbd5e1")
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", colors="#64748b")
    axis.set_xlabel("Percentile-position error (percentage points)", color="#475569", labelpad=10)
fig.suptitle(
    "More accurate percentiles at the default settings",
    x=0.06,
    y=0.97,
    ha="left",
    fontsize=18,
    fontweight="bold",
    color="#102a43",
)
fig.text(
    0.06,
    0.085,
    "Original delta 300 / Modern capacity 1,000. Lower is better. Equal weight per profile; precision held equal.",
    fontsize=10,
    color="#475569",
)
fig.text(
    0.06,
    0.025,
    "Storage differs: median paired field-byte ratio modern/original = 5.67 across all profiles; 0.71 on recorded data.",
    fontsize=10,
    color="#475569",
)
fig.subplots_adjust(left=0.09, right=0.98, top=0.77, bottom=0.31, wspace=0.35)
fig.savefig(folder / "default-accuracy.svg", metadata={"Date": None})
svg = folder / "default-accuracy.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
# A PNG preview is useful for local visual inspection; only the SVG is committed.
fig.savefig(folder / "default-accuracy.png", dpi=130)
