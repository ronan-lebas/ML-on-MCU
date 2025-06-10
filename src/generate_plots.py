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