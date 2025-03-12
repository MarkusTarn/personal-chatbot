import json
import glob
import argparse
import os

def extract_pairs(json_file, sender_name):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = []
    messages = data.get("messages", [])

    # Check if sender exists in the file
    senders = {msg.get("sender_name") for msg in messages if msg.get("sender_name")}
    if sender_name not in senders:
        raise ValueError(f"Sender '{sender_name}' not found in file {json_file}")

    # Simple pairing: use the previous message as context and current message as reply
    for i in range(1, len(messages)):
        prev = messages[i - 1]
        curr = messages[i]
        if curr.get("sender_name") == sender_name and prev.get("content"):
            prompt = prev.get("content").strip()
            response = curr.get("content", "").strip()
            if prompt and response:
                pairs.append({"prompt": prompt, "completion": response})
    return pairs

def main(sender_name, input_dir, output_file):
    if not sender_name:
        raise ValueError("Sender name can't be empty")

    all_pairs = []
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory {input_dir}")

    for file in glob.glob(os.path.join(input_dir, "*.json")):
        all_pairs.extend(extract_pairs(file, sender_name))
    print(f"Extracted {len(all_pairs)} prompt-response pairs.")

    with open(output_file, "w", encoding="utf-8") as f_out:
        for pair in all_pairs:
            f_out.write(json.dumps(pair) + "\n")
    print(f"Saved dataset to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract chat pairs from JSON files.")
    parser.add_argument("--sender_name", type=str, required=True, help="Name of the sender to extract responses from")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing chat JSON files")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()
    main(args.sender_name, args.input_dir, args.output_file)