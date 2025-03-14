from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def main(model_dir="./finetuned_model"):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    
    # Load the base model in 8-bit mode with automatic device mapping
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        load_in_8bit=True,          # Load in 8-bit mode to reduce memory usage
        # device_map="auto"           # Automatically place layers on the GPU
        device_map={"": "cuda"},
    )
    
    # Load the fine-tuned PEFT adapter on top of the base model.
    model = PeftModel.from_pretrained(base_model, model_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Adjust prompt formatting as needed.
    prompt = "User: What is your name?\nBot: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated response:")
    print(response)

if __name__ == "__main__":
    main()
