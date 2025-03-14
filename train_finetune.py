import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

def main(data_file, model_name="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", output_dir="./finetuned_model"):
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create BitsAndBytesConfig for 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Set to True for 8-bit quantization
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cuda"},
        # device_map="auto",  # Automatically places layers on the GPU
        quantization_config=quantization_config  # Use BitsAndBytesConfig for quantization
    )

    # Prepare the model for int8 training
    model = prepare_model_for_kbit_training(model)

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
    model.gradient_checkpointing_enable()

    # Move model to GPU if available
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=200,
        fp16=True,
        evaluation_strategy="no",
        label_names=["labels"],
        optim="adamw_8bit",
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
