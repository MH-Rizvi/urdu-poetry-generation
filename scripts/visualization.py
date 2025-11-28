import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Ensure visualizations folder exists
os.makedirs("visualizations", exist_ok=True)

# Load CSV
df = pd.read_csv("results/metrics/model_performance.csv")

# Perplexity plot

plt.figure(figsize=(14,7))
sns.barplot(
    data=df,
    x="Optimizer",
    y="Perplexity",
    hue="Model",
    ci=None  # remove error bars
)

plt.title("Perplexity Comparison by Model and Optimizer")
plt.ylabel("Perplexity")
plt.xlabel("Optimizer")
plt.legend(title="Model")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("visualizations/perplexity_bar_plot.png")
plt.show()

# Training time plot

plt.figure(figsize=(14,7))
sns.barplot(
    data=df,
    x="Optimizer",
    y="Time Taken (s)",
    hue="Model",
    ci=None
)

plt.title("Training Time Comparison by Model and Optimizer")
plt.ylabel("Time Taken (seconds)")
plt.xlabel("Optimizer")
plt.legend(title="Model")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("visualizations/training_time_bar_plot.png")
plt.show()

import seaborn as sns

# Correct pivot using keyword arguments
heatmap_data = df.pivot(index="Model", columns="Optimizer", values="Perplexity")

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Perplexity Heatmap by Model and Optimizer")
plt.xlabel("Optimizer")
plt.ylabel("Model")

# Save the plot
plt.savefig("visualizations/perplexity_heatmap.png")
plt.show()