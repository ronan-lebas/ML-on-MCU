import numpy as np
from matplotlib import pyplot as plt

folder_path = "reports/20250601-154841"
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