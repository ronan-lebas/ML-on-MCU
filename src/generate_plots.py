import numpy as np
from matplotlib import pyplot as plt

folder_path = "reports/20250611-120913"
path = folder_path + "/class_report.txt"

with open(path, 'r') as f:
    lines = f.readlines()
lines = [line.split() for line in lines if len(line.split()) > 0]
classes = [line[0] for line in lines[1:-3]]
f1_scores = [float(line[3]) for line in lines[1:-3]]
accuracy = float(lines[-3][1])

plt.figure(figsize=(10, 6))
plt.barh(classes, f1_scores)
plt.xlabel('F1 Score')
plt.title(f'F1 Scores by Class (Accuracy: {100 * accuracy:.0f}%)')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(folder_path + "/f1_scores_by_class.png")
plt.close()
models = [
    {
        "accuracy": 0.91,
        "latency": 2.2,
        "params": 295208,
        "flash": 326,
        "ram": 79,
        "macc": 73697544,
    },
    {
        "accuracy": 0.7,
        "latency": 0.3,
        "params": 19336,
        "flash": 53,
        "ram": 43,
        "macc": 9900136,
    },
    # {
    #     "accuracy": 0,
    #     "latency": 0,
    #     "params": 1213352,
    #     "flash": 1230,
    #     "ram": 81,
    #     "macc": 100280712,
    # },
    {
        "accuracy": 0.92,
        "latency": 2.6,
        "params": 592200,
        "flash": 619,
        "ram": 81,
        "macc": 82550824,
    },
    {
        "accuracy": 0.93,
        "latency": 1.3,
        "params": 67336,
        "flash": 102,
        "ram": 78,
        "macc": 47134696,
    }
]

for key1 in models[0].keys():
    for key2 in models[0].keys():
        if key1 <= key2:
            continue
        plt.figure(figsize=(10, 6))
        X = [model[key1] for model in models]
        Y = [model[key2] for model in models]
        numbers = [str(i + 1) for i in range(len(models))]
        idx = np.argsort(X)
        X = np.array(X)[idx]
        Y = np.array(Y)[idx]
        numbers = np.array(numbers)[idx]
        plt.plot(
            X,
            Y,
            marker='o'
        )
        for i, (x, y) in enumerate(zip(X, Y)):
            plt.text(x, y, numbers[i], fontsize=12, ha='right', va='bottom')
        plt.xlabel(key1.capitalize())
        plt.ylabel(key2.capitalize())
        plt.title(f'{key2.capitalize()} vs {key1.capitalize()}')
        plt.tight_layout()
        plt.savefig(folder_path + f"/{key2}_vs_{key1}.png")
        plt.close()


nodes = [
    (0, 'Conv2D', 49.263, 3.8, 3.8),
    (1, 'Conv2dPool', 493.912, 38.2, 42.0), 
    (2, 'Conv2D', 229.235, 17.7, 59.7), 
    (3, 'Conv2D', 517.350, 40.0, 99.7), 
    (4, 'Pool', 3.405, 0.3, 100.0), 
    (5, 'Dense', 0.091, 0.0, 100.0), 
    (6, 'Dense', 0.035, 0.0, 100.0)
]

type_to_color = {
    "Conv2D": "#1f77b4",
    "Conv2dPool": "#2ca02c",
    "Pool": "#ff7f0e",
    "Dense": "#d62728",
}

labels = [f"{n[1]} ({n[0]})" for n in nodes]
durations = [n[2] for n in nodes]
layer_types = [n[1] for n in nodes]
colors = [type_to_color[t] for t in layer_types]

y = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(12, 6))

bars = ax1.barh(y, durations, color=colors, edgecolor='black')
ax1.set_xlabel("Inference Time (ms)", fontsize=12)
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=10)
ax1.invert_yaxis()
ax1.grid(axis='x', linestyle='--', alpha=0.6)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width + 5, bar.get_y() + bar.get_height()/2,
             f"{durations[i]:.1f} ms\n({nodes[i][3]:.1f}%)", va='center', fontsize=9)

legend_elements = [
    plt.Line2D([0], [0], color=color, lw=8, label=lt)
    for lt, color in type_to_color.items()
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Layout
plt.title("Inference Time per Layer", fontsize=14)
plt.tight_layout()
plt.savefig(folder_path + "/inference_time_per_layer.png")
plt.close()