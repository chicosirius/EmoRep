#!/usr/bin/env python3
"""
visualize_emotion_space.py

可视化高质量样本文本及其8种情绪标签在语义空间中的分离模式。

- 从 emotion_scenario_pipeline + emotion_vector_probing 的输出加载数据
- 计算样本的语义向量（使用模型 hidden states）
- 用 PCA / t-SNE / UMAP 降维到2D
- 绘制每条样本的情绪分布图
- 叠加 emotion persona 向量方向（可选）

示例用法:
python visualize_emotion_space.py \
    --records_jsonl qwen2.5-7b-outputs/nb-records-raw.jsonl \
    --hf_model Qwen/Qwen2.5-7B-Instruct \
    --emotion_vecs qwen2.5-7b-outputs/emotion_vectors_enhanced.npz \
    --output_dir qwen2.5-7b-outputs
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from emo_text_generation import EmotionEmbeddingExtractor, EMOTIONAL_PROMPT_TEMPLATE

# Automatically select GPUs based on available devices
gpu_count = torch.cuda.device_count()
if gpu_count > 0:
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    print("No GPU found. Defaulting to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_records(jsonl_path):
    df = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            df.append(json.loads(line))
    return pd.DataFrame(df)


def compute_sample_embeddings(df, tokenizer, model, device="cuda", layer_idx=-1):
    """
    计算每条样本的句子级embedding。
    可以用情绪回答的token embedding取平均。
    """
    extractor = EmotionEmbeddingExtractor(tokenizer, model, device)
    embeddings, labels, texts = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding samples"):
        emo = row["gold_emotion"]
        prompt_emotional = EMOTIONAL_PROMPT_TEMPLATE.format(
            emotion=emo, question=row["scenario"] + "\n" + row["question"]
        )
        full_em = prompt_emotional + "\n\n" + row["emotional_answer"]
        try:
            acts = extractor.forward_response_tokens_mean(full_em, prompt_emotional)
            if layer_idx not in acts:
                layer_idx = max(acts.keys())
            vec = acts[layer_idx]
            embeddings.append(vec)
            labels.append(emo)
            texts.append(row["emotional_answer"][:60])
        except Exception as e:
            print("[Skip]", e)
            continue

    return np.array(embeddings), labels, texts


def visualize_2d(embeddings, labels, texts, output_dir, method="tsne", emotion_vecs=None):
    """
    绘制样本在二维空间的情绪分布。
    可选叠加 emotion_vectors 的方向箭头。
    """
    print(f"[Stage] Reducing dimensionality via {method.upper()}...")
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)

    reduced = reducer.fit_transform(embeddings)
    df_plot = pd.DataFrame(reduced, columns=["x", "y"])
    df_plot["emotion"] = labels
    df_plot["text"] = texts

    # 色彩映射
    emotions = sorted(df_plot["emotion"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {emo: cmap(i % 10) for i, emo in enumerate(emotions)}

    plt.figure(figsize=(10, 8))
    for emo in emotions:
        subset = df_plot[df_plot["emotion"] == emo]
        plt.scatter(subset["x"], subset["y"], s=25, alpha=0.7, label=emo, color=colors[emo])

    # 可选：叠加情绪方向向量
    if emotion_vecs is not None:
        base_layer = min(set(int(k.split("layer")[-1]) for k in emotion_vecs.keys()))
        layer_vecs = {k.split("__")[0]: v for k, v in emotion_vecs.items() if f"layer{base_layer}" in k}
        vecs = np.stack(list(layer_vecs.values()))
        pca = PCA(n_components=2)
        arrow2d = pca.fit_transform(vecs)
        origin = np.zeros((len(emotions), 2))
        for i, emo in enumerate(layer_vecs.keys()):
            plt.arrow(origin[i,0], origin[i,1],
                      arrow2d[i,0]*3, arrow2d[i,1]*3,
                      color=colors[emo], head_width=0.1, alpha=0.8)
            plt.text(arrow2d[i,0]*3.2, arrow2d[i,1]*3.2, emo, fontsize=9)

    plt.legend()
    plt.title(f"Emotion Space Visualization ({method.upper()})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"nb-emotion_space_{method}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def main(records_jsonl, hf_model, emotion_vecs_path, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_records(records_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True, device_map="auto")

    embeddings, labels, texts = compute_sample_embeddings(df, tokenizer, model, device=device)

    # 加载情绪方向
    emotion_vecs = np.load(emotion_vecs_path) if emotion_vecs_path else None

    visualize_2d(embeddings, labels, texts, output_dir, method="pca", emotion_vecs=emotion_vecs)
    visualize_2d(embeddings, labels, texts, output_dir, method="tsne", emotion_vecs=emotion_vecs)
    print("[Done] Visualizations ready.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_jsonl", default="/data/home/xixian_yong/EmoRep/embedding/qwen2.5-7b-outputs/nb-records-raw.jsonl")
    parser.add_argument("--hf_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--emotion_vecs", default="/data/home/xixian_yong/EmoRep/embedding/emotion_probing_outputs/nb-emotion_vectors_enhanced.npz")
    parser.add_argument("--output_dir", default="emotion_visualization")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args.records_jsonl, args.hf_model, args.emotion_vecs, args.output_dir, device=args.device)