import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
results = pd.read_csv('visdrone_evaluation.csv')

# Extract unique models and metrics
metrics = ['mAP50', 'mAP50-95', 'Average Precision', 'Average Recall']
colors = {
    'Yolov8_SAHI': '#1f77b4', 'Yolov8_NORMAL': '#aec7e8',
    'Yolov11_SAHI': '#ff7f0e', 'Yolov11_VANILLA': '#ffbb78',
    'RT-DETR_SAHI': '#2ca02c', 'RT-DETR_VANILLA': '#98df8a'
}  # Extended color palette

# Separate models into vanilla and SAHI
vanilla_models = results[results['Model'].str.contains('NORMAL|VANILLA')]
sahi_models = results[results['Model'].str.contains('SAHI')]

# Combine plots into one figure
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# Function to add values on top of bars
def add_values_on_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            rotation=90
        )

# Plot vanilla models
bar_width = 0.2
x = np.arange(len(metrics))
for idx, model in enumerate(vanilla_models['Model']):
    y_values = [
        vanilla_models.loc[vanilla_models['Model'] == model, metric].values[0]
        for metric in metrics
    ]
    bars = axes[0].bar(x + idx * bar_width, y_values, bar_width, label=model, color=colors[model])
    add_values_on_bars(axes[0], bars)

axes[0].set_title("VisDrone Vanilla Model Evaluation", fontsize=16)
axes[0].set_xticks(x + bar_width * (len(vanilla_models) / 2 - 0.5))
axes[0].set_xticklabels(metrics)
axes[0].legend(title="Models", fontsize=10)
axes[0].set_ylabel("Scores")

# Plot SAHI models
for idx, model in enumerate(sahi_models['Model']):
    y_values = [
        sahi_models.loc[sahi_models['Model'] == model, metric].values[0]
        for metric in metrics
    ]
    bars = axes[1].bar(x + idx * bar_width, y_values, bar_width, label=model, color=colors[model])
    add_values_on_bars(axes[1], bars)

axes[1].set_title("VisDrone SAHI Model Evaluation", fontsize=16)
axes[1].set_xticks(x + bar_width * (len(sahi_models) / 2 - 0.5))
axes[1].set_xticklabels(metrics)
axes[1].legend(title="Models", fontsize=10)
axes[1].set_ylabel("Scores")
axes[1].set_xlabel("Metrics")

plt.tight_layout()
plt.show()
