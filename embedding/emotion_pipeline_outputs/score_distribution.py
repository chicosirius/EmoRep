import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "/data/home/xixian_yong/EmoRep/embedding/emotion_pipeline_outputs/sc-records-raw.jsonl"

# Step 1: 统计
total_per_emotion = Counter()
correct_per_emotion = Counter()
wrong_pred_distribution = Counter()  # {(gold, pred): count}

all_emotions = set()

with open(file_path, "r") as f:
    for line in f:
        item = json.loads(line)
        gold = item["gold_emotion"]
        pred = item["pred_candidate"]
        
        all_emotions.update([gold, pred])
        
        total_per_emotion[gold] += 1
        if gold == pred:
            correct_per_emotion[gold] += 1
        else:
            wrong_pred_distribution[(gold, pred)] += 1

all_emotions = sorted(list(all_emotions))

# Step 2: 输出每个情绪的正确率
print("\n=== ✅ 每种情绪的预测正确率 ===")
for emotion in all_emotions:
    total = total_per_emotion[emotion]
    correct = correct_per_emotion[emotion]
    acc = correct / total if total > 0 else 0.0
    print(f"{emotion:12s}  accuracy = {acc:.2%}  ({correct}/{total})")

# Step 3: 构建混淆矩阵
confusion_matrix = pd.DataFrame(0, index=all_emotions, columns=all_emotions)
for (gold, pred), count in wrong_pred_distribution.items():
    confusion_matrix.loc[gold, pred] = count

# Step 4: 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds")
plt.title("Emotion Prediction Confusion Matrix (Wrong Predictions Only)")
plt.ylabel("Gold Emotion")
plt.xlabel("Predicted Emotion")
plt.show()