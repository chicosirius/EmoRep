#!/usr/bin/env python3
"""
emotion_vector_probing.py

从 emotion_scenario_pipeline.py 输出的 records 文件中分析得分分布，
筛选高质量样本（情绪表达强 & 中性表达稳定），
并生成 8 种情绪的可分离向量（emotion persona embeddings）。

支持的方法 (--method):
1) original
    - 平均层间差
    - PCA 主方向
    - Gram-Schmidt 正交化增强
2) centered_pca
    - 构造情绪迁移向量 Δ_{e,t}
    - 按任务均值中心化
    - PCA 去除任务主子空间
    - 得到每个情绪的残差方向 v_e
    - 归一化

结果输出 emotion_vectors_*.npz

python emotion_vector_probing.py \
    --records_jsonl qwen2.5-7b-outputs/nb-records-raw.jsonl \
    --hf_model Qwen/Qwen2.5-7B-Instruct \
    --method centered_pca \
    --output_dir qwen2.5-7b-outputs
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from emo_text_generation import (
    EmotionEmbeddingExtractor,
    NEUTRAL_PROMPT_TEMPLATE,
    EMOTIONAL_PROMPT_TEMPLATE
)

# Automatically select GPUs based on available devices
gpu_count = torch.cuda.device_count()
print("Detected GPU count:", gpu_count)
if gpu_count > 0:
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    print("No GPU found. Defaulting to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_records(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def analyze_distributions(df, output_dir):
    import matplotlib.pyplot as plt

    emo_scores = df["score_emotional_answer"].fillna(0)
    neu_scores = df["score_neutral_answer"].fillna(0)

    plt.figure()
    plt.hist(emo_scores, bins=30, alpha=0.7, label="Emotional")
    plt.hist(neu_scores, bins=30, alpha=0.7, label="Neutral")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of Emotional & Neutral Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distribution.png"))
    plt.close()

    print("[Stats] Emotional mean:", emo_scores.mean(), "Neutral mean:", neu_scores.mean())


def select_high_quality_pairs(df):
    emo_thresh = df["score_emotional_answer"].median()*0.9
    neu_thresh = df["score_neutral_answer"].median()*0.9
    filtered = df[(df["score_emotional_answer"] >= emo_thresh) &
                  (df["score_neutral_answer"] >= neu_thresh)].copy()
    print(f"[Selection] Retained {len(filtered)} / {len(df)} samples ({len(filtered)/len(df)*100:.1f}%)")
    return filtered


def extract_delta_vectors(df, tokenizer, model, device="cuda"):
    """
    逐样本提取 Δ_{e,t} = h_emotional - h_neutral
    返回:
        layer_deltas[e] = [vector_t1, vector_t2, ...]
    """
    extractor = EmotionEmbeddingExtractor(tokenizer, model, device)
    emotions = sorted(df["gold_emotion"].unique())
    layer_deltas = {e: [] for e in emotions}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Δ vectors"):
        emo = row["gold_emotion"]
        prompt_neutral = NEUTRAL_PROMPT_TEMPLATE.format(question=row["scenario"] + "\n" + row["question"])
        prompt_emotional = EMOTIONAL_PROMPT_TEMPLATE.format(emotion=emo, question=row["scenario"] + "\n" + row["question"])
        full_em = prompt_emotional + "\n\n" + row["emotional_answer"]
        full_ne = prompt_neutral + "\n\n" + row["neutral_answer"]

        try:
            em = extractor.forward_response_tokens_mean(full_em, prompt_emotional)
            ne = extractor.forward_response_tokens_mean(full_ne, prompt_neutral)
            diff = {li: em[li] - ne[li] for li in em.keys()}
            layer_deltas[emo].append(diff)
        except Exception as e:
            print(f"[Warning] extraction failed for sample {row['idx']}: {e}")
            continue

    return layer_deltas


##########################################
# ==========  方法1：原始方法 ============
##########################################
def enhance_original(persona_vectors):
    emotions = list(persona_vectors.keys())
    layers = sorted(next(iter(persona_vectors.values())).keys())
    enhanced = {}

    for li in layers:
        matrix = np.stack([persona_vectors[e][li] for e in emotions if li in persona_vectors[e]], axis=0)
        # 去均值
        matrix -= matrix.mean(axis=0)
        # PCA
        pca = PCA(n_components=1)
        pca.fit(matrix)

        # 正交化
        ortho_vecs = []
        for e in emotions:
            v = persona_vectors[e][li]
            v = v - v.mean()
            v = v / (np.linalg.norm(v) + 1e-6)
            for u in ortho_vecs:
                v -= np.dot(v, u) * u
            v = v / (np.linalg.norm(v) + 1e-6)
            ortho_vecs.append(v)
        enhanced[li] = {e: v for e, v in zip(emotions, ortho_vecs)}

    return enhanced


##########################################
# 方法2：均值中心化 + 去任务子空间
##########################################
def enhance_centered_pca(layer_deltas):
    """
    layer_deltas[e] = [ {layer_i : Δ_{e,t}}, ... ]

    Step:
    1) 聚合所有样本 Δ_{e,t} → Δ_{all}
    2) 对每 layer:
        a) 拼成矩阵 M
        b) 按样本均值中心化 Eq.3.1
        c) PCA 去除 top-K 主成分（任务主子空间） Eq.3.2
        d) 对 residual per emotion 求平均 → v_e
        e) 归一化
    """
    emotions = sorted(layer_deltas.keys())
    # 假设所有情绪都有样本
    layers = sorted(layer_deltas[emotions[0]][0].keys())
    enhanced = {}

    for li in layers:
        # -------- 1) 构造所有 Δ 的矩阵 --------
        all_vecs = []
        emo_labels = []
        for e in emotions:
            for d in layer_deltas[e]:
                all_vecs.append(d[li])
                emo_labels.append(e)

        M = np.stack(all_vecs, axis=0)               # (N_samples, D)
        M_centered = M - M.mean(axis=0, keepdims=True)  # Eq.3.1

        # -------- 2) PCA 去除任务主子空间 --------
        # 保留 top-(k=3) 主成分，可改超参
        pca = PCA(n_components=min(3, M_centered.shape[0]))
        pca.fit(M_centered)
        P = pca.components_                          # (k, D)

        # 投影残差  Eq.3.2
        # r = x - P^T (P x)
        residuals = M_centered - (M_centered @ P.T) @ P  # (N, D)

        # -------- 3) 对残差按情绪分组求平均 --------
        enhanced_li = {}
        for e in emotions:
            idx = [i for i, lab in enumerate(emo_labels) if lab == e]
            if len(idx) == 0:
                continue
            v = residuals[idx].mean(axis=0)

            # -------- 4) 归一化 --------
            v = v / (np.linalg.norm(v) + 1e-8)
            enhanced_li[e] = v

        enhanced[li] = enhanced_li

    return enhanced


##########################################
# 主执行入口
##########################################
def main(records_jsonl, hf_model, output_dir, method="original", device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    print("[Stage] Loading data and model...")

    df = load_records(records_jsonl)
    analyze_distributions(df, output_dir)
    selected = select_high_quality_pairs(df)

    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True, device_map="auto")

    # 先提取 Δ_{e,t}
    layer_deltas = extract_delta_vectors(selected, tokenizer, model, device=device)

    if method == "original":
        # 计算每种情绪平均差向量
        persona_vectors = {}
        emotions = sorted(layer_deltas.keys())
        layers = sorted(layer_deltas[emotions[0]][0].keys())

        for e in emotions:
            persona_vectors[e] = {}
            for li in layers:
                mats = np.stack([d[li] for d in layer_deltas[e]], axis=0)
                persona_vectors[e][li] = mats.mean(axis=0)

        enhanced = enhance_original(persona_vectors)
        out_file = os.path.join(output_dir, "emotion_vectors_original.npz")

    elif method == "centered_pca":
        enhanced = enhance_centered_pca(layer_deltas)
        out_file = os.path.join(output_dir, "emotion_vectors_centered_pca.npz")

    else:
        raise ValueError(f"Unknown method: {method}")

    # 保存
    np.savez_compressed(out_file, **{
        f"{emo}__layer{li}": enhanced[li][emo]
        for li in enhanced
        for emo in enhanced[li]
    })
    print(f"[Output] Saved emotion vectors → {out_file}")
    print("[Done]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_jsonl", type=str,
                        default="/data/home/xixian_yong/EmoRep/embedding/qwen2.5-7b-outputs/nb-records-raw.jsonl")
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="emotion_probing_outputs")
    parser.add_argument("--method", type=str, default="centered_pca",
                        choices=["original", "centered_pca"],
                        help="选择计算情绪向量的方法")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args.records_jsonl, args.hf_model, args.output_dir, method=args.method, device=args.device)