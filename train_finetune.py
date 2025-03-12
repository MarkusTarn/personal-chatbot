import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

def main(data_file, model_name="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", output_dir="./finetuned_model"):
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Move model to GPU if available
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset("json", data_files={"train": data_file})["train"]

    # Tokenize function
    def tokenize_function(examples):
        # Handle batched inputs
        texts = [prompt + "\n" + completion 
                for prompt, completion in zip(examples["prompt"], examples["completion"])]
        return tokenizer(texts, truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=200,
        fp16=True,
        evaluation_strategy="no",
        label_names=["labels"],
        no_cuda=False,
        use_cpu=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a conversational model using LoRA.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the training data JSONL file")
    parser.add_argument("--model_name", type=str, default="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default="./finetuned_model", help="Directory to save the fine-tuned model")
    args = parser.parse_args()
    main(args.data_file, args.model_name, args.output_dir)
