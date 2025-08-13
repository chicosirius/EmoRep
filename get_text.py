import json
import tqdm


DEP_FILE_PATH = "/data/home/xixian_yong/MentalRep/data/SWDD/depressed.jsonl"
CON_FILE_PATH = "/data/home/xixian_yong/MentalRep/data/SWDD/control.jsonl"

# read text from a file
def get_text(file_path):
    texts = []
    with open(file_path, "r") as f:
        for line in tqdm.tqdm(f, desc=f"Reading {file_path}"):
            data = json.loads(line)
            texts.append(data)
    return texts

# Example usage
if __name__ == "__main__":
    depressed_texts = get_text(DEP_FILE_PATH)
    control_texts = get_text(CON_FILE_PATH)
    
    print(f"Number of depressed texts: {len(depressed_texts)}")
    print(f"Number of control texts: {len(control_texts)}")
    
    # Print first 5 texts from each category
    print("\nFirst 5 depressed texts:")
    for text in depressed_texts[:1]:
        # 按json格式打印
        print(json.dumps(text, ensure_ascii=False, indent=2))
    
    print("\nFirst 5 control texts:")
    for text in control_texts[:1]:
        print(json.dumps(text, ensure_ascii=False, indent=2))