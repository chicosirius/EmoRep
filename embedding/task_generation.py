import re
import requests
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Configuration
API_URL = "http://localhost:3000/v1/chat/completions"
OUTPUT_XLSX = "emotion_scenarios_output-sc.xlsx"
MAX_ROWS = 200

# List of Plutchik’s basic emotions
PLUTCHIK_EMOTIONS = [
    "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"
]

# Dataset selection
AVAILABLE_DATASETS = {
    "social_chemistry": ("leondz/trustgpt_social_chemistry", "train", "action"),
    "norm_bank": ("SALT-NLP/NormBank", "train", "behavior")
}

# 1级标签
LEVEL1_LABELS = [
    "Work & Productivity",
    "Intimate Relationships",
    "Public & Societal",
    "Personal Feelings"
]

# 2级标签
LEVEL2_LABELS = [
    "With Authority Figures", "With Collaborators", "With Subordinates",
    "With Family", "With Lover", "With Friends",
    "With Service Providers", "With Fellow Individuals", "With Governing Bodies",
    "Learning & Working", "Entertainment by oneself", "Body & Spirits"
]

# 3级标签
LEVEL3_LABELS = [
    "Task Acceptance & Execution", "Stating Opinions & Disagreements", "Accepting Evaluation & Feedback",
    "Goal Alignment & Communication", "Responsibility Division & Competition", "Social Maintenance & Activities",
    "Task Assignment & Guidance", "Capability Development & Motivation", "Giving Evaluation & Feedback",
    "Traditional Constraints & Obligations", "Emotional Support & Care", "Clash of Values & Communication",
    "Daily Sharing & Companionship", "Future Planning & Decision-making", "Intimate Expression & Conflict",
    "Spending Leisure Time & Entertainment", "Confiding & Trust", "Boundary Exploration & Maintenance",
    "Making Requests & Waiting", "Complaining & Protecting Rights", "Result Acceptance & Evaluation",
    "Competition for Space & Resources", "Rule Compliance & Violation", "Sudden Assistance or Friction",
    "Rule Compliance & Supervision", "Rights Advocacy & Appeal", "Obligation Fulfillment & Undertaking",
    "Knowledge Acquisition & Delving", "Progress Halted & Problem-Solving", "Achievement & Hobbies",
    "Exploring Hobbies", "Health Management & Discomfort", "Growing Pains & Reflection", "Goal Setting & Motivation"
]

def load_seed_data(dataset_key: str):
    """
    Load dataset from Hugging Face and extract the seed column.
    """
    if dataset_key not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset '{dataset_key}' not found. Available options: {list(AVAILABLE_DATASETS.keys())}")

    dataset_name, split_name, text_column = AVAILABLE_DATASETS[dataset_key]
    print(f"Loading dataset: {dataset_name} (split='{split_name}') ...")
    dataset = load_dataset(dataset_name, split=split_name)
    print(f"Loaded {len(dataset)} rows. Using column '{text_column}' as seed topic.\n")
    return dataset, text_column

def parse_llm_json_response(response):
    """
    解析LLM返回的带有```json\n{...}\n```包裹的字符串，返回dict
    """
    # 如果response是带外层引号的字符串，先反序列化
    if response.startswith('"') and response.endswith('"'):
        response = json.loads(response)
    # print("LLM Response:", response)
    match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 尝试直接解析
        json_str = response
    try:
        return json.loads(json_str)
    except Exception:
        return None

def fuzzy_label_match(label, valid_list):
    """
    标签模糊匹配，忽略大小写和空格
    """
    label_clean = label.strip().lower()
    for valid in valid_list:
        if label_clean in valid.lower():
            return valid
    return f"Others: {label}"

def call_llm(messages, max_tokens=2048, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
    """
    Send a chat completion request to the local FastAPI LLM service.
    """
    payload = {
        "model": "local-llm",  # Placeholder
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ LLM request failed: {e}")
        return None


def classify_emotion(seed_text: str) -> str:
    """
    Use the LLM to classify a short text into one of Plutchik's 8 basic emotions.
    """
    messages = [
        {"role": "system", "content": "You are an expert emotion classifier."},
        {"role": "user", "content": f"""
        Given the following short text: "{seed_text}",
        classify which one of Plutchik's 8 basic emotions it most closely evokes:
        {", ".join(PLUTCHIK_EMOTIONS)}.
        If none applies, return "None".
        Return only the emotion label or 'None'.
        """}
    ]
    label = call_llm(messages, temperature=0.2)
    if not label:
        return None

    label = label.lower().strip()
    for emo in PLUTCHIK_EMOTIONS:
        if emo in label:
            return emo
    return None


def generate_scenario_with_labels(seed_text: str, emotion: str) -> dict:
    """
    Generate a scenario with second-person immersion, a fixed question,
    and hierarchical labels (level 1/2/3) based on the provided classification framework.
    """
    messages = [
        {"role": "system", "content": "You are an expert at creating emotionally evocative social scenarios and classifying them into a hierarchical taxonomy."},
        {"role": "user", "content": f"""
        Create a short, vivid scenario in English based on the seed topic: "{seed_text}".
        Use second-person perspective ("you") so the reader feels immersed.
        The scenario should naturally evoke the emotion: "{emotion}".
        
        At the end of the scenario, include the following fixed question:
        "Facing this situation, how do you feel inside?"
        
        Assign hierarchical labels for the scenario ONLY from the following lists.
        If the scenario does not clearly fit, assign: Others: <predicted label>.

        Level 1 options: {LEVEL1_LABELS}
        Level 2 options: {LEVEL2_LABELS}
        Level 3 options: {LEVEL3_LABELS}

        Return a JSON object with the following keys:
        - "scenario_text": scenario description
        - "question": the fixed question
        - "label_l1": level 1 label
        - "label_l2": level 2 label
        - "label_l3": level 3 label
        """}
    ]
    
    response = call_llm(messages, temperature=0.8, max_tokens=300)
    
    # Try to parse JSON from response
    try:
        scenario_data = parse_llm_json_response(response)
        # 检查标签是否在列表中，否则标为 Others
        def check_label(label, valid_list):
            if label in valid_list:
                return label
            if label.startswith("Others:"):
                return label
            return f"Others: {label}"
        scenario_data["label_l1"] = check_label(scenario_data.get("label_l1", ""), LEVEL1_LABELS)
        scenario_data["label_l2"] = check_label(scenario_data.get("label_l2", ""), LEVEL2_LABELS)
        scenario_data["label_l3"] = check_label(scenario_data.get("label_l3", ""), LEVEL3_LABELS)
        return scenario_data
    except:
        return {
            "scenario_text": response,
            "question": "Facing this situation, how do you feel inside?",
            "label_l1": "Others",
            "label_l2": "Others",
            "label_l3": "Others"
        }


# ===============================================
# Main processing pipeline
# ===============================================

def build_emotion_dataset(dataset_key: str, max_rows: int = MAX_ROWS, output_xlsx: str = OUTPUT_XLSX):
    """
    Main pipeline:
    1. Load the selected dataset.
    2. Classify emotion for each seed topic.
    3. Generate corresponding scenario task with labels.
    4. Save results to Excel.
    """
    dataset, text_column = load_seed_data(dataset_key)
    results = []

    for i, example in tqdm(enumerate(dataset), total=min(max_rows, len(dataset)), desc="Processing"):
        if i >= max_rows:
            break

        seed_text = str(example[text_column]).strip()
        if not seed_text:
            continue

        emotion = classify_emotion(seed_text)
        if emotion:
            scenario_data = generate_scenario_with_labels(seed_text, emotion)
            # 只保存非空 scenario_text 的行
            if scenario_data.get("scenario_text", "").strip():
                results.append({
                    "scenario_text": scenario_data.get("scenario_text", ""),
                    "question": scenario_data.get("question", ""),
                    "emotion": emotion,
                    "label_l1": scenario_data.get("label_l1", ""),
                    "label_l2": scenario_data.get("label_l2", ""),
                    "label_l3": scenario_data.get("label_l3", "")
                })

    df = pd.DataFrame(results, columns=["scenario_text", "question", "emotion", "label_l1", "label_l2", "label_l3"])
    df.to_excel(output_xlsx, index=False)
    print(f"\nDone! Generated {len(df)} entries → saved to {output_xlsx}")

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts(dropna=True))



# ===============================================
# Example usage
# ===============================================
if __name__ == "__main__":
    build_emotion_dataset(dataset_key="social_chemistry", max_rows=5000)