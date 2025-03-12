from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main(model_dir="./finetuned_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "User: What is your name?\Bot: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated response:")
    print(response)

if __name__ == "__main__":
    main()
