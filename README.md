# 🇪🇹 EthioBBPE: Byte-Level BPE Tokenizer for Ethiopian Languages

A robust, production-ready Byte-Level BPE (BBPE) tokenizer training framework specifically optimized for Ethiopian languages (Amharic, Oromo, Tigrinya, Somali, etc.) using Hugging Face's `tokenizers` library.

## ✨ Features

- **Optimized for Ethiopic Script**: Handles complex Ge'ez script characters and Latin-based orthographies seamlessly.
- **Byte-Level Encoding**: Robust against unknown characters and emojis, ensuring no `<UNK>` tokens.
- **End-to-End Pipeline**: From raw text corpus to a ready-to-use `tokenizer.json`.
- **Hugging Face Compatible**: Directly usable with `transformers` models.
- **Flexible Configuration**: Customize vocabulary size, minimum frequency, and special tokens.
- **Multi-Format Support**: Train on `.txt`, `.json`, or `.jsonl` datasets.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Prepare Your Data
Place your training corpus (raw text files) in the `data/` directory.
```text
data/
├── amharic_corpus.txt
├── oromo_corpus.txt
└── mixed_ethio_text.txt
```

### 2. Train the Tokenizer

**Using CLI:**
```bash
python scripts/train_tokenizer.py \
    --data_dir ./data \
    --model_name EthioBBPE \
    --vocab_size 32000 \
    --min_frequency 2 \
    --special_tokens "[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"
```

**Using Python API:**
```python
from scripts.bbpe_trainer import BBPETrainer, BBPEConfig

# Configure for Ethiopian Languages
config = BBPEConfig(
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

trainer = BBPETrainer(config=config, model_name="EthioBBPE")
trainer.train_from_directory("./data")
trainer.save("./models/EthioBBPE")

# Test it
text = "ሰላም ልጄ እንዴት ነሽ? (Hello daughter, how are you?)"
tokens = trainer.tokenize(text)
print(f"Tokens: {tokens}")
```

### 3. Load and Use

```python
from tokenizers import Tokenizer

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("models/EthioBBPE/tokenizer.json")

# Encode
encoded = tokenizer.encode("የኢትዮጵያ ህዝብ")
print(encoded.ids)
print(encoded.tokens)

# Decode
decoded = tokenizer.decode(encoded.ids)
print(decoded)
```

## 🏗️ Architecture

The `EthioBBPE` architecture follows these steps:
1. **Pre-tokenization**: Splits text into words while preserving byte-level integrity.
2. **Byte Conversion**: Converts all characters (including Ge'ez script) into their byte representations.
3. **BPE Training**: Learns merge operations based on frequency in the corpus.
4. **Vocabulary Creation**: Generates a fixed-size vocabulary of byte-level tokens.

## 📂 Project Structure

```text
Ethio_BBPE/
├── data/                   # Raw training data
├── models/                 # Output directory for trained models
├── scripts/
│   ├── bbpe_trainer.py     # Core logic (BBPEConfig, BBPETrainer)
│   ├── train_tokenizer.py  # CLI entry point
│   └── example_usage.py    # Usage examples
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🤗 Hugging Face Hub Integration

The trained tokenizer can be easily shared and loaded via the Hugging Face Hub.

### Loading from Hub
```python
from tokenizers import Tokenizer

# Load directly from the Hub
tokenizer = Tokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")

# Encode text
output = tokenizer.encode("ሰላም ልዑል እንዴት ነህ?")
print(output.tokens)
```

### Uploading Your Own Trained Model
If you have trained a custom version, you can upload it using the `huggingface_hub` library:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./models/your_model_name",
    repo_id="your-username/your-repo-name",
    repo_type="model",
    token="YOUR_HF_TOKEN"
)
```

## 📄 License

MIT License

## 🙏 Acknowledgments

Built for the Ethiopian NLP community to foster better language understanding and generation capabilities.
