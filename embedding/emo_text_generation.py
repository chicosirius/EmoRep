#!/usr/bin/env python3
"""
emotion_scenario_pipeline.py

Refactored single-model version:
- Only one HF model is loaded and shared for generation, logits, and hidden_states probing.
- EmotionEmbeddingExtractor accepts an existing tokenizer+model instead of loading its own copy.
- Functionality preserved from original pipeline.

Usage (example):
python emotion_scenario_pipeline.py --xlsx_path emotion_scenarios_output.xlsx \
    --hf_model Qwen/Qwen2.5-7B-Instruct --device cuda --max_samples 500

Requirements (recommended):
pip install torch transformers sentence-transformers scikit-learn pandas openpyxl tqdm

Notes:
- This script assumes the chosen HF model supports generation and can return hidden_states.
- If the HF model is seq2seq-only, some logits-based candidate scoring may need alternative handling.
"""

import os
import argparse
import json
import math
import time
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# classifier dependencies
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Automatically select GPUs based on available devices
gpu_count = torch.cuda.device_count()
if gpu_count > 0:
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    print("No GPU found. Defaulting to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# -------------------------
# EmotionEmbeddingExtractor (shared-model version)
# -------------------------
class EmotionEmbeddingExtractor(torch.nn.Module):
    """
    Extractor that reuses an existing tokenizer + model (same as generation model).
    It expects the provided model to support output_hidden_states=True in forward.
    """
    def __init__(self, tokenizer, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        # ensure model is eval and on device
        self.model.eval()

    def forward(self, text: str, max_length: int = 2048) -> torch.Tensor:
        """
        Given text, return per-layer mean token embedding.
        Returns: tensor shape [num_layers, hidden_size] (CPU float32)
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple: (layer0, layer1, ..., layerN)
        layer_embeddings = []
        for hidden_state in hidden_states:
            # hidden_state shape [batch, seq_len, hidden_dim]
            token_embeddings = hidden_state.squeeze(0)  # [seq_len, hidden_dim]
            avg_embedding = token_embeddings.mean(dim=0)  # [hidden_dim]
            layer_embeddings.append(avg_embedding.cpu())
        return torch.stack(layer_embeddings)  # [num_layers, hidden_dim]

    def forward_response_tokens_mean(self, full_text: str, prompt_text: str, max_length: int = 2048) -> Dict[int, np.ndarray]:
        """
        Return mean activations per layer for response tokens only.
        full_text = prompt + response
        prompt_text = text of prompt used (so we can locate response token positions)
        Returns dict layer_index -> numpy array (hidden_dim,)
        """
        inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=max_length)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt')['input_ids']
        prompt_len = prompt_ids.shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple: (layer0, layer1, ..., layerN)
        seq_len = inputs['input_ids'].shape[1]
        resp_positions = list(range(prompt_len, seq_len))
        layer_means = {}
        for li, h in enumerate(hidden_states):
            if len(resp_positions) == 0:
                mean_vec = np.zeros(h.shape[-1], dtype=np.float32)
            else:
                resp_act = h[0, resp_positions, :].cpu().numpy()  # [resp_len, hidden_dim]
                mean_vec = resp_act.mean(axis=0)
            layer_means[li] = mean_vec
        return layer_means

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_jsonl(path: str, records: List[Dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# -------------------------
# Next-token / candidate phrase probability (HF)
# -------------------------
def compute_candidate_logprobs_causal(hf_tokenizer, hf_model, prompt: str, candidates: List[str], device='cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    """
    For causal/auto-regressive HF model:
    For each candidate phrase, compute log P(candidate | prompt) by concatenating prompt + candidate and summing log-softmax probabilities of candidate tokens.
    Return softmax-normalized probabilities across candidates.
    Note: This function expects hf_model to return logits (i.e., is a causal LM or otherwise exposes logits).
    """
    hf_model.eval()
    scores = []
    enc_prompt = hf_tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    base_input_ids = enc_prompt['input_ids']  # [1, L]
    base_len = base_input_ids.shape[1]
    for cand in candidates:
        cand_ids = hf_tokenizer(cand, return_tensors='pt', add_special_tokens=False).to(device)['input_ids']  # [1, M]
        full_ids = torch.cat([base_input_ids, cand_ids], dim=1).to(device)  # [1, L+M]
        # forward once to get logits
        with torch.no_grad():
            out = hf_model(full_ids, output_hidden_states=False, return_dict=True)
            logits = getattr(out, "logits", None)
            if logits is None:
                # If model doesn't provide logits (e.g., some AutoModel wrappers), fall back to uniform score
                scores.append(float("-inf"))
                continue
        # compute logprob of candidate tokens: for token t in candidate, use logits at position base_len + t - 1 to predict token t
        logprob = 0.0
        for t in range(cand_ids.shape[1]):
            pos = base_len + t - 1
            # For the very first predicted token in candidate, the model's logits position should be base_len - 1
            if pos < 0:
                pos = base_len - 1
            logits_at_pos = logits[0, pos, :]
            token_id = cand_ids[0, t].item()
            lp = F.log_softmax(logits_at_pos, dim=0)[token_id].item()
            logprob += lp
        scores.append(logprob)
    # handle -inf scores
    scores_np = np.array([s if np.isfinite(s) else -1e9 for s in scores], dtype=np.float64)
    # stable softmax
    mx = scores_np.max()
    probs = np.exp(scores_np - mx)
    # Avoid division by zero
    if probs.sum() == 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()
    return {cand: float(p) for cand, p in zip(candidates, probs)}


def choose_candidate_by_prompt(hf_tokenizer, hf_model, prompt_base: str, candidates: List[str], device='cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    """
    Prompt the generation model with the prompt_base and a numbered candidate list.
    Expect the model to output either the candidate text or the candidate number.
    Returns a probability dict (one-hot for the chosen candidate when detected; uniform fallback).
    """
    import re
    hf_model.eval()
    cand_list_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    selector_prompt = (
        f"{prompt_base}\n\nCandidates:\n{cand_list_text}\n\n"
        "Please choose the single best emotion from the Candidates above and output only the exact choice (either the full emotion text or the number)."
    )
    try:
        inputs = hf_tokenizer(selector_prompt, return_tensors='pt', truncation=True).to(device)
        with torch.no_grad():
            if hasattr(hf_model, "generate"):
                out_ids = hf_model.generate(**inputs, max_new_tokens=16, do_sample=True)
                gen_ids = out_ids[0][inputs['input_ids'].shape[1]:]
                txt = hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            else:
                out = hf_model(**inputs)
                logits = getattr(out, "logits", None)
                if logits is not None:
                    next_id = logits[:, -1, :].argmax(dim=-1)
                    txt = hf_tokenizer.decode(next_id)
                else:
                    txt = ""
    except Exception:
        txt = ""

    txt_first = txt.splitlines()[0].strip() if txt else ""
    # try exact or contained match (case-insensitive)
    for c in candidates:
        if txt_first.lower() == c.lower() or c.lower() in txt_first.lower():
            probs = {cand: 0.0 for cand in candidates}
            probs[c] = 1.0
            return probs
    # try numeric selection like "1" or "1."
    m = re.search(r"(\d+)", txt_first)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            chosen = candidates[idx]
            probs = {cand: 0.0 for cand in candidates}
            probs[chosen] = 1.0
            return probs
    # fallback: uniform
    return {c: 1.0 / len(candidates) for c in candidates}


# -------------------------
# Generation: neutral vs emotional prompts (English)
# -------------------------
NEUTRAL_PROMPT_TEMPLATE = """System: You are a completely neutral assistant. 
When answering the user's question below, respond in **first-person ("I")**, but do NOT include any emotional language, tone, or expressive phrasing. 
Use factual, objective, and concise statements only, avoiding any subjective impressions or feelings.

User question:
{question}

Provide a strictly neutral, emotion-free answer in first-person.
"""

EMOTIONAL_PROMPT_TEMPLATE = """System: You are an assistant tasked to respond in a clearly expressed emotional style. 
Answer the user's question below in **first-person ("I")** using the {emotion} emotion vividly. 
Use emotion words, descriptive phrases, figurative language, and a tone that unmistakably conveys {emotion}. 
Ensure the content is safe, appropriate, and not offensive.

User question:
{question}

Provide an emotionally expressive answer in first-person that clearly conveys {emotion}.
"""

TASK_PROMPT_TEMPLATE = """You will be given a scenario, assuming you are the main character, and complete the corresponding answer.

Scenario:
{question}

Provide an answer in first-person (answer in one paragraph, no more than 150 words).
"""

# -------------------------
# LLM-based judge prompt (English) — optimized for strict scoring
# -------------------------
JUDGE_PROMPT_TEMPLATE = """You are an emotion evaluator. Your task is to rate how strongly a given answer expresses the target emotion "{emotion}" on a scale from 0 to 100.

Scoring guidelines:
0: No expression of the emotion; purely neutral or factual.
1-20: Minimal, weak, or subtle expression; almost neutral.
21-40: Moderate hints of emotion; some emotional words or tone.
41-60: Noticeable emotion; clear words or phrases expressing the target emotion.
61-80: Strong and vivid expression; consistent tone and descriptive language conveying {emotion}.
81-100: Very strong, unmistakable, and intense expression; highly vivid and clear use of emotional language, imagery, or figurative expressions.

Always return a single numeric value between 0 and 100. Do NOT include explanations, text, or commentary.

Examples:
Target emotion: sadness
Answer: "I feel a deep sense of loss and sorrow." → 85
Answer: "It is unfortunate that things turned out this way." → 45
Answer: "The report shows the results." → 0

Now, given the target emotion "{emotion}", question, and answer below, rate the answer strictly according to the scale above.

Question:
{question}

Answer:
{answer}
"""

NEUTRAL_JUDGE_PROMPT_TEMPLATE = """You are an evaluator. Your task is to rate how emotion-free the given answer is on a scale from 0 to 100.

Scoring guidelines:
0: The answer is highly emotional; contains vivid emotional language.
1-20: Slight traces of emotion; mostly factual.
21-40: Some emotional hints, but still largely neutral.
41-60: Mixed; partially neutral, partially emotional.
61-80: Mostly neutral; minimal emotional content.
81-100: Completely neutral; no emotional language, tone, or expressions.

Always return a single numeric value between 0 and 100. Do NOT include explanations, text, or commentary.

Question:
{question}

Answer:
{answer}
"""


def llm_judge_score(hf_tokenizer, hf_model_for_generation, emotion: str, question: str, answer: str, device='cpu', max_new_tokens=8, num_repeats: int = 5) -> float:
    """
    Use HF generation model to generate a numeric score multiple times and return the average.
    num_repeats: number of generations to sample and average (uses sampling to get diversity).
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(emotion=emotion, question=question, answer=answer)
    inputs = hf_tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
    vals = []
    last_txt = ""
    import re
    with torch.no_grad():
        for i in range(max(1, num_repeats)):
            if hasattr(hf_model_for_generation, "generate"):
                try:
                    out_ids = hf_model_for_generation.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        eos_token_id=getattr(hf_tokenizer, "eos_token_id", None)
                    )
                    gen_ids = out_ids[0][inputs['input_ids'].shape[1]:]
                    txt = hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                except Exception:
                    out = hf_model_for_generation(**inputs)
                    logits = getattr(out, "logits", None)
                    if logits is not None:
                        next_id = logits[:, -1, :].argmax(dim=-1)
                        txt = hf_tokenizer.decode(next_id)
                    else:
                        txt = ""
            else:
                out = hf_model_for_generation(**inputs)
                logits = getattr(out, "logits", None)
                if logits is not None:
                    next_id = logits[:, -1, :].argmax(dim=-1)
                    txt = hf_tokenizer.decode(next_id)
                else:
                    txt = ""
            last_txt = txt
            m = re.search(r"(\d{1,3}(?:\.\d+)?)", txt)
            if m:
                v = float(m.group(1))
                v = max(0.0, min(100.0, v))
                vals.append(v)
            else:
                txt_low = txt.lower()
                if any(w in txt_low for w in ["no emotion", "none", "no emotional", "neutral"]):
                    vals.append(0.0)
                # else: skip this sample (do not append)
    if len(vals) > 0:
        return float(np.mean(vals))
    # fallback to heuristic on last generation if no numeric parsed
    txt_low = last_txt.lower()
    if any(w in txt_low for w in ["no emotion", "none", "no emotional", "neutral"]):
        return 0.0
    return 50.0  # fallback mid score

def llm_judge_neutral_score(hf_tokenizer, hf_model_for_generation, question: str, answer: str, device='cpu', max_new_tokens=8, num_repeats: int = 5) -> float:
    """
    Use HF generation model to generate a numeric score for neutral-ness multiple times and return the average.
    100 = completely neutral, 0 = very emotional
    """
    prompt = NEUTRAL_JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
    inputs = hf_tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
    vals = []
    last_txt = ""
    import re
    with torch.no_grad():
        for i in range(max(1, num_repeats)):
            if hasattr(hf_model_for_generation, "generate"):
                try:
                    out_ids = hf_model_for_generation.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        eos_token_id=getattr(hf_tokenizer, "eos_token_id", None)
                    )
                    gen_ids = out_ids[0][inputs['input_ids'].shape[1]:]
                    txt = hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                except Exception:
                    out = hf_model_for_generation(**inputs)
                    logits = getattr(out, "logits", None)
                    if logits is not None:
                        next_id = logits[:, -1, :].argmax(dim=-1)
                        txt = hf_tokenizer.decode(next_id)
                    else:
                        txt = ""
            else:
                out = hf_model_for_generation(**inputs)
                logits = getattr(out, "logits", None)
                if logits is not None:
                    next_id = logits[:, -1, :].argmax(dim=-1)
                    txt = hf_tokenizer.decode(next_id)
                else:
                    txt = ""
            last_txt = txt
            m = re.search(r"(\d{1,3}(?:\.\d+)?)", txt)
            if m:
                v = float(m.group(1))
                v = max(0.0, min(100.0, v))
                vals.append(v)
            else:
                txt_low = txt.lower()
                if any(w in txt_low for w in ["completely neutral", "no emotion", "none", "neutral"]):
                    vals.append(100.0)
                # else: skip
    if len(vals) > 0:
        return float(np.mean(vals))
    # fallback heuristic
    txt_low = last_txt.lower()
    if any(w in txt_low for w in ["completely neutral", "no emotion", "none", "neutral"]):
        return 100.0
    return 50.0


# -------------------------
# Lightweight classifier judge (sentence-transformer embeddings + logistic regression)
# -------------------------
class ClassifierJudge:
    def __init__(self, embed_model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        print(f"[ClassifierJudge] Loading embedding model: {embed_model_name}")
        self.embedder = SentenceTransformer(embed_model_name, device=device)
        self.models = {}  # emotion -> trained LogisticRegression
        self.label_encoders = {}  # per emotion: encoder for binary labels

    def train_for_emotion(self, texts: List[str], labels_binary: List[int], emotion_name: str):
        """
        Train a simple logistic regression for emotion_name with texts & binary labels (1=emotionful, 0=neutral).
        """
        X = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        y = np.array(labels_binary, dtype=int)
        # small train/test split
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.12, random_state=42, stratify=y if len(set(y))>1 else None)
        except Exception:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.12, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        self.models[emotion_name] = clf
        print(f"[ClassifierJudge] Trained classifier for {emotion_name} (train size {len(y_train)})")

    def score(self, texts: List[str], emotion_name: str) -> np.ndarray:
        """
        Return predicted probability of being 'emotionful' (class 1) for given texts.
        """
        if emotion_name not in self.models:
            raise ValueError(f"No classifier trained for emotion {emotion_name}")
        emb = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        probs = self.models[emotion_name].predict_proba(emb)[:, 1]  # probability for class 1
        # scale to 0-100
        return probs * 100.0

# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(xlsx_path: str,
                 hf_model_name: str,
                 extractor_model_name: Optional[str],
                 device: str = 'cuda',  # Default to GPU
                 max_samples: Optional[int] = None,
                 output_dir: str = 'emotion_pipeline_outputs',
                 use_classifier_judge: bool = False,
                 classifier_prep_samples: int = 1000):
    ensure_dir(output_dir)
    # 1) load data
    print("[Stage] Loading dataset:", xlsx_path)
    df = pd.read_excel(xlsx_path)
    expected_cols = ['scenario_text', 'question', 'emotion']
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Input Excel missing required column: {c}")
    if max_samples:
        df = df.iloc[:max_samples].reset_index(drop=True)
    emotions = sorted(df['emotion'].dropna().unique().tolist())
    print(f"[Data] Loaded {len(df)} samples; emotions detected: {emotions}")

    # prepare candidates (we will use raw emotion labels as candidate phrases)
    candidates = [str(e) for e in emotions]

    # 2) load HF tokenizer & model for logits & generation (single load)
    print("[Stage] Loading HF tokenizer and model for generation / scoring (single shared model):", hf_model_name)
    gen_tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True, use_fast=True)
    # Prefer AutoModelForCausalLM; fall back to seq2seq if necessary
    gen_model = None
    used_seq2seq = False
    try:
        gen_model = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True, device_map="auto")
        print("[Stage] Loaded AutoModelForCausalLM.")

        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_model.config.pad_token_id = gen_tokenizer.eos_token_id
    except Exception as e:
        raise RuntimeError(f"Failed to load model {hf_model_name} as causal: {e}")
        print("[Warning] Could not load AutoModelForCausalLM; trying AutoModelForSeq2SeqLM:", e)
        try:
            gen_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name, trust_remote_code=True, device_map="auto")
            used_seq2seq = True
            print("[Stage] Loaded AutoModelForSeq2SeqLM (seq2seq). Some logits-based scoring may be less accurate.")
        except Exception as e2:
            raise RuntimeError(f"Failed to load model {hf_model_name} as causal or seq2seq: {e2}")

    gen_model.eval()

    # 3) create extractor that reuses tokenizer + model
    # extractor_model_name parameter is ignored; we reuse gen_model to avoid a second load
    extractor = EmotionEmbeddingExtractor(gen_tokenizer, gen_model, device=device)

    # 4) optionally prepare classifier judge (if requested)
    classifier_judge = None
    if use_classifier_judge:
        classifier_judge = ClassifierJudge(device=device)
        # Create a small synthetic dataset by generating neutral and emotional answers
        print("[ClassifierJudge] Preparing synthetic dataset for classifier training...")
        synthetic_texts = []
        synthetic_labels = []
        prep_df = df.sample(min(len(df), classifier_prep_samples), random_state=42).reset_index(drop=True)
        for i, row in tqdm(prep_df.iterrows(), total=len(prep_df)):
            q = str(row['question'])
            emo = str(row['emotion'])
            prompt_neutral = NEUTRAL_PROMPT_TEMPLATE.format(question=q)
            prompt_emotional = EMOTIONAL_PROMPT_TEMPLATE.format(emotion=emo, question=q)
            # generate both outputs
            neutral_text = ""
            emotional_text = ""
            try:
                ni = gen_tokenizer(prompt_neutral, return_tensors='pt', truncation=True).to(device)
                with torch.no_grad():
                    outn = gen_model.generate(**ni, max_new_tokens=120)
                neutral_text = gen_tokenizer.decode(outn[0], skip_special_tokens=True).strip()
            except Exception as e:
                neutral_text = ""
            try:
                ei = gen_tokenizer(prompt_emotional, return_tensors='pt', truncation=True).to(device)
                with torch.no_grad():
                    oute = gen_model.generate(**ei, max_new_tokens=180)
                emotional_text = gen_tokenizer.decode(oute[0], skip_special_tokens=True).strip()
            except Exception as e:
                emotional_text = ""
            if neutral_text:
                synthetic_texts.append(neutral_text)
                synthetic_labels.append(0)
            if emotional_text:
                synthetic_texts.append(emotional_text)
                synthetic_labels.append(1)
        if len(synthetic_texts) < 10:
            print("[ClassifierJudge] Warning: Not enough synthetic texts to train classifier. Falling back to LLM judge.")
            use_classifier_judge = False
            classifier_judge = None
        else:
            classifier_judge.train_for_emotion(synthetic_texts, synthetic_labels, emotion_name='global_emotionful')

    # 5) iterate over dataset: stage 1 + stage 2; collect records
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        scenario = str(row['scenario_text'])
        question = str(row['question'])
        gold_emotion = str(row['emotion'])
        prompt_base = scenario + "\n\n" + question + "\n\n"

        # Stage 1: compute candidate probs (only works well with causal LM)
        # try:
        #     if used_seq2seq:
        #         # Seq2seq might not expose proper next-token causal logits; fallback to uniform
        #         probs = {c: 1.0/len(candidates) for c in candidates}
        #     else:
        #         probs = compute_candidate_logprobs_causal(gen_tokenizer, gen_model, prompt_base, candidates, device=device)
        # except Exception as e:
        #     print(f"[Warning] candidate prob computation failed for idx {idx}: {e}")
        #     probs = {c: 1.0/len(candidates) for c in candidates}
        # pred = max(probs.items(), key=lambda x: x[1])[0]
        # pred_correct = (pred == gold_emotion)

        # Stage 1: 使用 prompt 让模型直接从候选情绪中选择（替换基于 token 概率的做法）
        try:
            probs = choose_candidate_by_prompt(gen_tokenizer, gen_model, prompt_base, candidates, device=device)
        except Exception as e:
            print(f"[Warning] candidate selection prompt failed for idx {idx}: {e}")
            probs = {c: 1.0/len(candidates) for c in candidates}
        pred = max(probs.items(), key=lambda x: x[1])[0]
        pred_correct = (pred == gold_emotion)

        # Stage 2: generation neutral & emotional
        prompt_neutral = NEUTRAL_PROMPT_TEMPLATE.format(question=scenario + "\n" + question)
        prompt_emotional = EMOTIONAL_PROMPT_TEMPLATE.format(emotion=gold_emotion, question=scenario + "\n" + question)
        # Generate neutral answer
        neutral_ans = ""
        emotional_ans = ""
        try:
            ni = gen_tokenizer(prompt_neutral, return_tensors='pt', truncation=True).to(device)
            with torch.no_grad():
                outn = gen_model.generate(**ni, max_new_tokens=1024, do_sample=True)
            neutral_ans_ids = outn[0][ni['input_ids'].shape[1]:]  # 从 prompt 后开始
            neutral_ans = gen_tokenizer.decode(neutral_ans_ids, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[Warning] generation neutral failed for idx {idx}: {e}")
            neutral_ans = ""
        # Generate emotional answer
        try:
            ei = gen_tokenizer(prompt_emotional, return_tensors='pt', truncation=True).to(device)
            with torch.no_grad():
                oute = gen_model.generate(**ei, max_new_tokens=1024, do_sample=True)
            emotional_ans_ids = oute[0][ei['input_ids'].shape[1]:]
            emotional_ans = gen_tokenizer.decode(emotional_ans_ids, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[Warning] generation emotional failed for idx {idx}: {e}")
            emotional_ans = ""

        # Judge scoring
        if use_classifier_judge and classifier_judge is not None:
            try:
                score_emotional = float(classifier_judge.score([emotional_ans], 'global_emotionful')[0])
                score_neutral = float(classifier_judge.score([neutral_ans], 'global_emotionful')[0])
            except Exception as e:
                print("[Warning] classifier judge failed:", e)
                score_emotional = 0.0
                score_neutral = 0.0
        else:
            try:
                score_emotional = llm_judge_score(gen_tokenizer, gen_model, gold_emotion, scenario + "\n" + question, emotional_ans, device=device)
            except Exception as e:
                print("[Warning] llm_judge (emotional) failed:", e)
                score_emotional = 0.0
            try:
                score_neutral = llm_judge_neutral_score(gen_tokenizer, gen_model, scenario + "\n" + question, neutral_ans, device=device)
            except Exception as e:
                print("[Warning] llm_judge (neutral) failed:", e)
                score_neutral = 0.0

        rec = {
            "idx": int(idx),
            "scenario": scenario,
            "question": question,
            "gold_emotion": gold_emotion,
            "candidate_probs": probs,
            "pred_candidate": pred,
            "pred_candidate_correct": pred_correct,
            "neutral_answer": neutral_ans,
            "emotional_answer": emotional_ans,
            "score_emotional_answer": float(score_emotional),
            "score_neutral_answer": float(score_neutral),
        }
        records.append(rec)


    # Save per-sample records
    records_path = os.path.join(output_dir, "sc-records-raw.jsonl")
    save_jsonl(records_path, records)
    print("[Output] Saved per-sample records to", records_path)

    # Stage 1 accuracy summary
    total = len(records)
    correct = sum(1 for r in records if r["pred_candidate_correct"])
    overall_acc = correct / total if total > 0 else 0.0
    counts = {e: {"total": 0, "correct": 0} for e in emotions}
    for r in records:
        g = r["gold_emotion"]
        if g in counts:
            counts[g]["total"] += 1
            if r["pred_candidate_correct"]:
                counts[g]["correct"] += 1
    per_emotion_acc = {e: (counts[e]["correct"] / counts[e]["total"] if counts[e]["total"]>0 else None) for e in emotions}
    acc_summary = {
        "overall_accuracy": overall_acc,
        "per_emotion_accuracy": per_emotion_acc,
        "total_samples": total
    }
    with open(os.path.join(output_dir, "sc-accuracy_summary-raw.json"), 'w', encoding='utf-8') as f:
        json.dump(acc_summary, f, ensure_ascii=False, indent=2)
    print("[Output] Saved accuracy summary to", os.path.join(output_dir, "sc-accuracy_summary-raw.json"))
    print("[Stats] overall_acc:", overall_acc)

    return None

    # Stage 3: Probing: select samples that are good for probing:
    # selection criteria: emotional answer score >= 70 (high emotion) AND neutral answer score <= 10 (neutral)
    print("[Stage] Selecting examples for probing (high emotional vs neutral)")
    probe_pairs_by_emotion = {e: [] for e in emotions}
    for r in records:
        emo = r["gold_emotion"]
        if r["score_emotional_answer"] >= 70.0 and r["score_neutral_answer"] <= 10.0:
            prompt_neutral = NEUTRAL_PROMPT_TEMPLATE.format(question=r["question"])
            prompt_emotional = EMOTIONAL_PROMPT_TEMPLATE.format(emotion=emo, question=r["question"])
            probe_pairs_by_emotion[emo].append((prompt_emotional, r["emotional_answer"], prompt_neutral, r["neutral_answer"], r["idx"]))

    # For each emotion, for each pair, extract layer-wise means and compute diff (emotional - neutral).
    print("[Stage] Extracting layer activations and computing emotion vectors (this may be slow)")
    persona_vectors = {}  # {emotion: {layer: vector}}
    for emo in emotions:
        pairs = probe_pairs_by_emotion.get(emo, [])
        if not pairs:
            print(f"[Probe] No good probe pairs found for emotion '{emo}'")
            continue
        per_pair_diffs = []  # list of dict layer->vector
        for (p_em, resp_em, p_ne, resp_ne, idx_) in pairs:
            full_em = p_em + "\n\n" + resp_em
            full_ne = p_ne + "\n\n" + resp_ne
            try:
                em_acts = extractor.forward_response_tokens_mean(full_em, p_em)
                ne_acts = extractor.forward_response_tokens_mean(full_ne, p_ne)
            except Exception as e:
                print(f"[Warning] activation extraction failed for sample {idx_}: {e}")
                continue
            # compute diff per layer
            diff = {}
            for li in em_acts.keys():
                diff_vec = em_acts[li] - ne_acts.get(li, np.zeros_like(em_acts[li]))
                diff[li] = diff_vec
            per_pair_diffs.append(diff)
        if len(per_pair_diffs) == 0:
            print(f"[Probe] No valid activations for emotion '{emo}'")
            continue
        # average across pairs per layer
        layer_indices = sorted(per_pair_diffs[0].keys())
        layer_avg = {}
        for li in layer_indices:
            mats = np.stack([d[li] for d in per_pair_diffs if li in d], axis=0)
            layer_avg[li] = mats.mean(axis=0)
        persona_vectors[emo] = layer_avg
        print(f"[Probe] Computed persona vector for {emo} with {len(per_pair_diffs)} pairs")

    # Save persona vectors to npz
    save_dict = {}
    for emo, layer_dict in persona_vectors.items():
        for li, vec in layer_dict.items():
            key = f"{emo}__layer{li}"
            save_dict[key] = np.asarray(vec)
    npz_path = os.path.join(output_dir, "emotion_vectors.npz")
    if save_dict:
        np.savez_compressed(npz_path, **save_dict)
        print("[Output] Saved persona vectors to", npz_path)
    else:
        print("[Output] No persona vectors to save (no probe pairs found)")

    # Save a CSV summarizing judge scores and top candidate prediction correctness
    df_out = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "records_summary.csv")
    df_out.to_csv(csv_path, index=False)
    print("[Output] Saved tabular summary to", csv_path)

    return {
        "records_path": records_path,
        "accuracy_summary": acc_summary,
        "persona_vectors_npz": npz_path if save_dict else None,
        "csv_summary": csv_path
    }

# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Emotion scenario probing pipeline (single-model optimized)")
    parser.add_argument("--xlsx_path", type=str, default="/data/home/xixian_yong/EmoRep/data/probing_task/emosc-sc-raw.xlsx", help="Input Excel file path")
    parser.add_argument("--hf_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HF model name or path for generation & scoring")
    parser.add_argument("--extractor_model", type=str, default=None, help="Model name/path for extractor (ignored; extractor will reuse hf_model)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device: cuda or cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="max number of samples to process")
    parser.add_argument("--output_dir", type=str, default="llama3.1-8b-outputs", help="directory to save outputs")
    parser.add_argument("--use_classifier_judge", action="store_true", help="use lightweight classifier judge instead of LLM judge")
    parser.add_argument("--classifier_prep_samples", type=int, default=300, help="samples to synthesize for classifier training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    results = run_pipeline(
        xlsx_path=args.xlsx_path,
        hf_model_name=args.hf_model,
        extractor_model_name=args.extractor_model,
        device=args.device,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        use_classifier_judge=args.use_classifier_judge,
        classifier_prep_samples=args.classifier_prep_samples
    )
    t1 = time.time()
    print("Finished. Outputs:", results)
    print(f"Elapsed time: {t1-t0:.1f}s")