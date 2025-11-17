#!/usr/bin/env python3
"""
emotion_injection_and_viz.py

功能：
- 加载 nb-records-raw.jsonl（或任意包含 scenario/question/gold_emotion 的 records）
- 加载情绪向量文件 emotion_vectors_*.npz（每 key 格式 like "anger__layerX"）
- 对若干 prompt（默认 80）：
    - 构造 NEUTRAL_PROMPT_TEMPLATE
    - 生成 baseline (no injection)
    - 生成 injected (prefix-embedding injection using emotion vector)
- 提取两种生成的 response token 平均 hidden-state 向量（同 extractor）
- 计算投影（cosine）到情绪向量（作为情绪强度 proxy）
- 用 PCA / t-SNE 可视化注入前后 embedding 分布（同一图显示并以连线表示位移）
- 保存图与 csv 对比表

注意：
- 需要 GPU 才理想（device 默认 cuda）。
- prefix embedding 注入要求情绪向量维度与模型 embedding_dim 相同（通常一致）。
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from emo_text_generation import (
    EmotionEmbeddingExtractor, 
    NEUTRAL_PROMPT_TEMPLATE, 
    EMOTIONAL_PROMPT_TEMPLATE,
    TASK_PROMPT_TEMPLATE,
    JUDGE_PROMPT_TEMPLATE
)

# ---------------------------
# Utilities
# ---------------------------
def load_records(jsonl_path: str) -> pd.DataFrame:
    recs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            recs.append(json.loads(line))
    df = pd.DataFrame(recs)
    return df

def load_emotion_vectors(npz_path, layer_prefer=None):
    """
    加载 emotion_vectors_centered_pca.npz
    key 格式: {emotion}__layer{li}

    返回结构:
        {
            layer_id: {
                emotion: vector(np.array)
            }
        }
    """
    data = np.load(npz_path)
    out = {}

    for name in data.files:
        # name example: "anger__layer15"
        emo, layer_str = name.split("__")
        li = int(layer_str.replace("layer", ""))

        if li not in out:
            out[li] = {}

        vec = data[name]

        # 归一化（避免数值尺度差异）
        # norm = np.linalg.norm(vec)
        # if norm > 1e-8:
        #     vec = vec / norm

        out[li][emo] = vec

    # 如果用户指定 layer，则只返回该层
    if layer_prefer is not None:
        if layer_prefer in out:
            print(f"[OK] Using preferred layer {layer_prefer}")
            return {layer_prefer: out[layer_prefer]}
        else:
            print(f"[Warning] preferred layer {layer_prefer} not found, using all layers.")

    print(f"[Stage] Loaded {len(out)} layers from {npz_path}")
    return out

def make_prefix_embeddings_from_vector(em_vector: np.ndarray, pref_len: int, alpha: float, device: torch.device):
    """
    Create a prefix embedding tensor shaped [pref_len, dim] by repeating scaled em_vector.
    """
    v = torch.from_numpy(em_vector).to(device=device, dtype=torch.float32)
    # normalize then scale
    v = v / (v.norm() + 1e-12)
    v = v * float(alpha)
    pref = v.unsqueeze(0).repeat(pref_len, 1)  # [pref_len, dim]
    return pref  # not batched

def generate_with_prefix(model, tokenizer, prompt_text: str, prefix_embeds: Optional[torch.Tensor],
                         max_new_tokens=200, device='cuda', do_sample=False):
    """
    Generate by using inputs_embeds = concat(prefix_embeds, token_embeddings(prompt))
    Returns generated text (only the generated portion after prompt).
    """
    model.eval()
    # Tokenize prompt to get ids and token embeddings
    enc = tokenizer(prompt_text, return_tensors='pt', truncation=True).to(device)
    input_ids = enc['input_ids']
    # get token embeddings
    with torch.no_grad():
        inputs_emb = model.get_input_embeddings()(input_ids)  # [1, L, d]
    if prefix_embeds is not None:
        # prefix_embeds shape [pref_len, d]
        # build inputs_embeds = [prefix, prompt_embeds]
        pref = prefix_embeds.unsqueeze(0)  # [1, pref_len, d]
        inputs_embeds = torch.cat([pref, inputs_emb], dim=1)  # [1, pref_len+L, d]
        # need to build a corresponding attention_mask
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        # generate: pass inputs_embeds and attention_mask
        gen_out = model.generate(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=do_sample,
                                 pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id))
    else:
        gen_out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                 pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id))
    # decode generated portion
    # If prefix was used, tokenizer.decode on gen_out will include the prompt tokens; we need to remove prompt tokens length
    gen_ids = gen_out[0]
    if prefix_embeds is not None:
        # compute prompt token length
        prompt_len = input_ids.shape[1]
        # generated ids include prefix? Actually inputs_embeds replaced input_ids; gen_out contains tokens corresponding to prompt tokens and newly generated tokens.
        # We assume gen_out includes the prompt tokens appended; so slice from prompt_len
        gen_text = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    else:
        gen_text = tokenizer.decode(gen_ids[input_ids.shape[1]:], skip_special_tokens=True)
    return gen_text.strip()

def compute_response_embedding_mean(full_text: str, prompt_text: str, extractor: EmotionEmbeddingExtractor, layer_prefer: Optional[int] = None):
    """
    Returns a 1-D numpy vector: mean embedding of response tokens at chosen layer (or last layer).
    """
    layer_means = extractor.forward_response_tokens_mean(full_text, prompt_text)
    # choose layer
    if layer_prefer is None:
        li = sorted(layer_means.keys())[-1]
    else:
        li = min(layer_means.keys(), key=lambda x: abs(x - layer_prefer))
    return layer_means[li]  # numpy array

# def add_emotion_to_token_embeddings(inputs_emb: torch.Tensor, em_vector: np.ndarray, alpha: float):
#     """
#     inputs_emb: [1, L, d], em_vector: (d,)
#     returns modified tensor (same shape)
#     """
#     device = inputs_emb.device
#     v = torch.from_numpy(em_vector).to(device=device, dtype=torch.float32)
#     v = v / (v.norm() + 1e-12)
#     v = v * float(alpha)
#     return inputs_emb + v.view(1, 1, -1)  # broadcast across seq

def add_emotion_to_token_embeddings(inputs_emb: torch.Tensor, em_vector: np.ndarray, alpha: float,
                                    model: Optional[torch.nn.Module] = None, layer_idx: Optional[int] = None,
                                    device: Optional[torch.device] = None):
    """
    两种用法（向后兼容）：
    1) 仅传 inputs_emb, em_vector, alpha:
         -> 返回修改后的 inputs_emb（在 input-embedding 层上把情绪向量加到每个 token embedding 上）
    2) 传入额外的 model 与 layer_idx:
         -> 在 model 指定层注册 forward hook，使得该层的输出 hidden states 在 forward 时被加上情绪向量。
         -> 返回 (inputs_emb, hook_handle) 以便调用方在生成后移除 hook_handle.remove()
    说明：
      - em_vector 会先归一化再乘以 alpha。
      - device 默认为 inputs_emb.device（如果 inputs_emb 为 None，可显式传入 device）。
    """
    # 准备向量 tensor
    if device is None:
        if isinstance(inputs_emb, torch.Tensor):
            device = inputs_emb.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.from_numpy(em_vector).to(device=device, dtype=torch.float32)
    # v = v / (v.norm() + 1e-12)
    v = v * float(alpha)

    # case A: 注入到输入 embedding（默认/兼容旧调用）
    if model is None or layer_idx is None:
        if not isinstance(inputs_emb, torch.Tensor):
            raise ValueError("inputs_emb must be a torch.Tensor when not providing model+layer_idx")
        return inputs_emb + v.view(1, 1, -1)  # broadcast 加到每个 token

    # case B: 在指定 transformer 层注册 hook（返回 hook handle）
    # 尝试寻找常见的 layer 容器
    layer_container = None
    potential_attrs = [
        ("transformer", "h"),
        ("transformer", "layers"),
        ("model", "decoder", "layers"),
        ("base_model", "layers"),
        ("gpt_neox", "blocks"),
        ("model", "encoder", "layer"),
        ("model", "decoder", "block"),
        ("transformer", "blocks"),
    ]
    for path in potential_attrs:
        obj = model
        found = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                found = False
                break
        if found:
            layer_container = obj
            break

    # 找不到内部层容器，退回到 input embedding 注入
    if layer_container is None:
        if isinstance(inputs_emb, torch.Tensor):
            return inputs_emb + v.view(1, 1, -1)
        else:
            raise RuntimeError("Cannot locate transformer layers in model to register hook.")

    # 规范 layer_idx（允许用户传真实层号或绝对 id）
    if hasattr(layer_container, "__len__"):
        num_layers = len(layer_container)
        target_idx = min(range(num_layers), key=lambda i: abs(i - layer_idx))
    else:
        target_idx = layer_idx

    # 定义 hook：对 layer 输出（tensor 或 tuple）加上 v（广播到 batch/seq）
    def _make_hook(v_tensor):
        def _hook(module, input, output):
            if isinstance(output, tuple):
                out0 = output[0] + v_tensor.view(1, 1, -1)
                return (out0,) + tuple(output[1:])
            else:
                return output + v_tensor.view(1, 1, -1)
        return _hook

    target_layer = layer_container[target_idx]
    hook_handle = target_layer.register_forward_hook(_make_hook(v))

    # 返回原始 inputs_emb 以便兼容调用；同时返回 hook handle，调用方需在生成后调用 hook_handle.remove()
    return hook_handle

# -------------------------
# Main pipeline
# -------------------------
def run_injection_and_viz(records_jsonl: str,
                          emotion_vecs_npz: str,
                          hf_model: str,
                          scorer_model: Optional[str],
                          output_dir: str,
                          device: str = 'cuda',
                          alpha: float = 1,
                          pref_len: int = 6,
                          num_prompts: int = 120,
                          layer_prefer: Optional[int] = None):
    os.makedirs(output_dir, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    print(f"[Info] Using device: {device_t}")
    print("[Stage] Loading records...")
    df = load_records(records_jsonl)

    # take subset of high-quality filtered prompts if available
    # if "score_emotional_answer" in df.columns and "score_neutral_answer" in df.columns:
    #     emo_thresh = df["score_emotional_answer"].median()*0.9
    #     neu_thresh = df["score_neutral_answer"].median()*0.9
    #     df = df[(df["score_emotional_answer"] >= emo_thresh) & (df["score_neutral_answer"] >= neu_thresh)].reset_index(drop=True)
    #     print(f"[Filtered] using {len(df)} high-quality records after median filter.")
    # else:
    #     print("[Warning] score columns not found; using all records.")
    # limit
    # df_sub = df.sample(n=min(num_prompts, len(df)), random_state=42).reset_index(drop=True)

    df_sub = df.head(num_prompts).reset_index(drop=True)
    print(f"[Info] Using {len(df_sub)} records for injection and visualization.")

    print("[Stage] Loading model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True, device_map="auto")
    model.to(device_t)
    model.eval()

    print("[Stage] Loading emotion vectors:", emotion_vecs_npz)
    em_vecs = load_emotion_vectors(emotion_vecs_npz, layer_prefer=layer_prefer)
    # 打印em_vecs的shape信息
    # for layer_id, emo_dict in em_vecs.items():
    #     print(f"[Info] Layer {layer_id} has {len(emo_dict)} emotions.")
    #     for emo, vec in emo_dict.items():
    #         print(f"  Emotion: {emo}, Vector shape: {vec.shape}")

    # 选择要使用的 layer（优先使用指定的 layer_prefer，否则选最大的 layer id）
    if layer_prefer is not None and layer_prefer in em_vecs:
        layer_to_use = layer_prefer
    else:
        layer_to_use = sorted(em_vecs.keys())[-1]
        if layer_prefer is not None:
            print(f"[Warning] preferred layer {layer_prefer} not present, using layer {layer_to_use}")

    em_vecs_layer = em_vecs[layer_to_use]  # dict: emotion -> vector
    emotions = sorted(list(em_vecs_layer.keys()))
    print(f"[Info] emotions loaded (using layer {layer_to_use}):", emotions)
    print(f"[Info] Example emotion vector shape:", next(iter(em_vecs_layer.values())).shape)

    # extractor for computing response embeddings
    extractor = EmotionEmbeddingExtractor(tokenizer, model, device=device_t)


    # load scorer model (third-party judge). If not provided, use generation model as judge.
    scorer_name = scorer_model if scorer_model else hf_model
    print(f"[Stage] Loading scorer model: {scorer_name}")
    scorer_tok = AutoTokenizer.from_pretrained(scorer_name, trust_remote_code=True)
    scorer_model_hf = AutoModelForCausalLM.from_pretrained(scorer_name, trust_remote_code=True, device_map="auto")
    # scorer_model_hf.to(device_t)
    scorer_model_hf.eval()

    records_out = []
    embed_before = []
    embed_after = []
    labels = []
    proj_before = []
    proj_after = []

    for idx, row in tqdm(df_sub.iterrows(), total=len(df_sub), desc="Processing prompts"):
        emo = row["gold_emotion"]
        labels.append(emo)
        question_combined = row["scenario"] + "\n" + row["question"]

        # 1) ordinary generation (no injection)
        prompt_emotional = TASK_PROMPT_TEMPLATE.format(question=question_combined)
        print("[Prompt] Emotional:\n", prompt_emotional)
        enc = tokenizer(prompt_emotional, return_tensors='pt', truncation=True).to(device_t)
        with torch.no_grad():
            gen_ids = model.generate(**enc, max_new_tokens=512,
                                     do_sample=True,
                                     temperature=0.6,
                                     top_p=0.9,
                                     repetition_penalty=1.2,
                                     no_repeat_ngram_size=3,
                                     pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id))[0]
        prompt_len = enc['input_ids'].shape[1]
        gen_ord = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True).strip()
        print("[Generated - ordinary]:\n", gen_ord)
        full_ord = prompt_emotional + "\n\n" + gen_ord
        emb_ord = compute_response_embedding_mean(full_ord, prompt_emotional, extractor, layer_prefer=layer_prefer)

        # 2) injection generation
        em_vector = em_vecs_layer[emo]
        if em_vector is None:
            raise ValueError(f"Emotion vector for {emo} not found in layer {layer_to_use}.")
        hook_handle = add_emotion_to_token_embeddings(None, em_vector, alpha,
                                                     model=model, layer_idx=layer_to_use)
        enc_2 = tokenizer(prompt_emotional, return_tensors='pt', truncation=True).to(device_t)
        with torch.no_grad():
            # inputs_emb = model.get_input_embeddings()(enc_2['input_ids'])  # [1, L, d]
            # inputs_emb_injected = add_emotion_to_token_embeddings(inputs_emb, em_vector, alpha)
            # attention_mask = torch.ones(inputs_emb_injected.shape[:2], dtype=torch.long, device=device_t)
            # gen_ids2 = model.generate(inputs_embeds=inputs_emb_injected,
            #                              attention_mask=attention_mask,
            #                              max_new_tokens=1024,
            #                              pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id))[0]
            gen_ids2 = model.generate(**enc_2, max_new_tokens=512,
                                      do_sample=True,
                                     temperature=0.6,
                                     top_p=0.9,
                                     repetition_penalty=1.2,
                                     no_repeat_ngram_size=3,
                                     pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id))[0]
        prompt_len2 = enc_2['input_ids'].shape[1]
        gen_inj = tokenizer.decode(gen_ids2[prompt_len2:], skip_special_tokens=True).strip()
        hook_handle.remove()
        print("[Generated - injected (layer hook)]:\n", gen_inj)
        full_inj = prompt_emotional + "\n\n" + gen_inj
        emb_inj = compute_response_embedding_mean(full_inj, prompt_emotional, extractor, layer_prefer=layer_prefer)

        # 3) 记录
        records_out.append({
            "idx": row["idx"] if "idx" in row else idx,
            "gold_emotion": emo,
            "prompt_emotional": prompt_emotional,
            "generated_ordinary": gen_ord,
            "generated_injected": gen_inj
        })
        embed_before.append(emb_ord)
        embed_after.append(emb_inj)
        # 4) 计算投影（cosine similarity）
        cos_before = cosine_similarity(emb_ord.reshape(1, -1), em_vector.reshape(1, -1))[0][0]
        cos_after = cosine_similarity(emb_inj.reshape(1, -1), em_vector.reshape(1, -1))[0][0]
        proj_before.append(cos_before)
        proj_after.append(cos_after)
        print(f"[Info] Cosine similarity to '{emo}' vector: before={cos_before:.4f}, after={cos_after:.4f}")

        return
        


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_jsonl", default="/data/home/xixian_yong/EmoRep/embedding/qwen2.5-7b-outputs/sc-records-raw.jsonl")
    parser.add_argument("--emotion_vecs", default="/data/home/xixian_yong/EmoRep/embedding/emotion_probing_outputs/emotion_vectors_centered_pca.npz", help="npz file with keys like 'anger__layer23'")
    parser.add_argument("--hf_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--scorer_model", default="Qwen/Qwen2.5-3B-Instruct", help="optional: model used for scoring (if different from hf_model)")
    parser.add_argument("--output_dir", default="injection_viz_out_qwen2.5-7b-instruct")
    parser.add_argument("--alpha", type=float, default=50, help="injection strength scalar")
    parser.add_argument("--pref_len", type=int, default=6, help="prefix length (virtual tokens)")
    parser.add_argument("--num_prompts", type=int, default=1000, help="number of prompts to evaluate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer_prefer", type=int, default=None, help="prefer emotion vector from this layer if multiple")
    args = parser.parse_args()

    run_injection_and_viz(args.records_jsonl, args.emotion_vecs, args.hf_model,
                          args.scorer_model, args.output_dir, device=args.device, alpha=args.alpha,
                          pref_len=args.pref_len, num_prompts=args.num_prompts,
                          layer_prefer=args.layer_prefer)