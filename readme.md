# Personal Fine-tuned Chatbot

Aim of this project is to extract personal chat data from Messenger JSON files and use it to fine-tune an open source conversational model (e.g., OpenAssistant) using LoRA.

## Files

- **extract_data.py**: Parses JSON chat files to extract prompt-response pairs (where `messages.content` is the text and `sender_name` equals given name) and writes them to a JSONL file.
- **train_finetune.py**: Fine-tunes a chosen conversational model using the dataset and PEFT (LoRA) for parameter-efficient fine-tuning.
- **test_generate.py**: Generates sample responses from your fine-tuned model.
- **requirements.txt**: Contains all required Python packages.

## Setup

1. **Create Virtual Environment:**
```bash
python -m venv venv
```
2. **Activate Virtual Environment:**
```bash
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
4. **Log In to Hugging Face to access the models:**
```bash
huggingface-cli login
```
> If you don't have a token, you can create one by following the [instructions](#generating-huggingface-token) below.


## Usage

1. **Add raw training data:**  
Include all your Meta Messenger chat-thread JSON files in a directory called `data/`.
2. **Extract Training Data:**
```bash
python extract_data.py --sender_name "Markus Tarn" --input_dir data/ --output_file training_data.jsonl
```
3. **Fine-tune Model with Extracted Data:**
```bash
python train_finetune.py --data_file training_data.jsonl
```
4. **Test Generated Responses:**
```bash
python test_generate.py
```



## Generating HuggingFace Token
1. Either [create a new account](https://huggingface.co/join) or [use an existing one](https://huggingface.co/login).
2. Once your account is created and you’re logged in, got to Settings.
3. In the settings menu on the left, click on Access Tokens.
4. Create a New Token:
   * Name: Enter a descriptive name (e.g., “local-read-token”).
   * Role: Select Read (this is enough to download models and datasets).
   * Note: If you plan to upload your models or write to Hugging Face, you would need a token with Write permissions.
5. You will see your new token displayed. Copy or store it because you might not see it again.

