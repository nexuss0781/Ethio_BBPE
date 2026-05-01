# EthioBBPE

A robust, production-ready Byte-Level BPE (BBPE) tokenizer training environment built with Hugging Face's `tokenizers` library. EthioBBPE provides a flexible framework for training high-quality tokenizers on any text corpus with advanced features like checkpointing and compressed storage.

## ✨ Features

- **Byte-Level Encoding**: Handles any Unicode character seamlessly, eliminating unknown token (`<unk>`) issues.
- **End-to-End Pipeline**: From raw text corpus to a ready-to-use `tokenizer.json`.
- **Hugging Face Compatible**: Directly usable with `transformers` models.
- **Flexible Configuration**: Customize vocabulary size, minimum frequency, and special tokens.
- **Multi-Format Support**: Train on `.txt`, `.json`, `.jsonl`, or `.parquet` datasets.
- **Production Ready**: 
  - ✅ Automatic checkpointing for fault-tolerant training
  - ✅ Gzip compression for efficient storage (~60% space savings)
  - ✅ Structured logging with progress tracking
  - ✅ Auto-generated model cards for Hugging Face Hub

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Prepare Your Data

Place your training corpus in the `data/` directory. Supported formats: `.txt`, `.json`, `.jsonl`, `.parquet`

For parquet files, use the data preparation script:
```bash
python scripts/prepare_data.py --data_dir ./data --output ./data/training_corpus.txt
```

### 2. Train the Tokenizer

**Using CLI:**
```bash
python scripts/train_tokenizer.py \
    --data_dir ./data \
    --model_name EthioBBPE \
    --vocab_size 30000 \
    --min_frequency 2
```

**Advanced Options:**
```bash
python scripts/train_tokenizer.py \
    --data_dir ./data \
    --model_name EthioBBPE \
    --vocab_size 32000 \
    --min_frequency 2 \
    --special_tokens "<pad>" "<unk>" "<s>" "</s>" "<mask>" \
    --use_checkpoint \
    --checkpoint_dir ./models/checkpoints \
    --save_compressed
```

**Using Python API:**
```python
from scripts.bbpe_trainer import BBPEConfig, EthioBBPETrainer

# Configure
config = BBPEConfig(
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
    use_checkpoint=True,
    save_compressed=True
)

trainer = EthioBBPETrainer(config=config)
trainer.train()  # Uses config.data_dir automatically
trainer.save()   # Uses config.model_name automatically

# Test it
text = "Hello world! This is a test."
tokens = trainer.tokenize(text)
print(f"Tokens: {tokens}")
```

### 3. Load and Use

```python
from tokenizers import Tokenizer

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("models/EthioBBPE/tokenizer.json")

# Encode
encoded = tokenizer.encode("Hello world this is a test")
print(encoded.ids)
print(encoded.tokens)

# Decode
decoded = tokenizer.decode(encoded.ids)
print(decoded)
```

## 🏗️ Architecture

The `EthioBBPE` architecture follows these steps:
1. **Pre-tokenization**: Splits text into words while preserving byte-level integrity.
2. **Byte Conversion**: Converts all characters into their byte representations.
3. **BPE Training**: Learns merge operations based on frequency in the corpus.
4. **Vocabulary Creation**: Generates a fixed-size vocabulary of byte-level tokens.
5. **Compression** (optional): Applies gzip compression to vocabulary for efficient storage.

## 📂 Project Structure

```text
Ethio_BBPE/
├── data/                       # Raw training data
│   ├── *.parquet              # Parquet datasets
│   └── training_corpus.txt    # Prepared training corpus
├── models/                     # Output directory for trained models
│   ├── EthioBBPE/             # Trained tokenizer
│   │   ├── tokenizer.json     # Main tokenizer file
│   │   ├── vocab.json.gz      # Compressed vocabulary
│   │   ├── config.json        # Training configuration
│   │   └── README.md          # Model card
│   └── checkpoints/           # Training checkpoints
├── scripts/
│   ├── bbpe_trainer.py        # Core logic (BBPEConfig, EthioBBPETrainer)
│   ├── train_tokenizer.py     # CLI entry point
│   ├── prepare_data.py        # Data preparation from parquet
│   └── example_usage.py       # Usage examples
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🔧 Advanced Features

### Checkpointing
Automatically saves training progress and can resume from the latest checkpoint:
```bash
python scripts/train_tokenizer.py --use_checkpoint --checkpoint_dir ./checkpoints
```

### Compression
Saves vocabulary in gzip format, reducing storage by ~60%:
```bash
python scripts/train_tokenizer.py --save_compressed
```

Output includes both standard `tokenizer.json` and compressed `vocab.json.gz`.

### Custom Special Tokens
Define custom special tokens for your use case:
```bash
python scripts/train_tokenizer.py --special_tokens "<pad>" "<unk>" "<s>" "</s>"
```

## 🤗 Hugging Face Hub Integration

### Loading from Hub
```python
from transformers import AutoTokenizer

# Load directly from the Hub
tokenizer = AutoTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")

# Encode text
output = tokenizer.encode("Hello world this is a test")
print(output.tokens)
```

### Uploading Your Own Trained Model
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./models/EthioBBPE",
    repo_id="your-username/your-repo-name",
    repo_type="model",
    token="YOUR_HF_TOKEN"
)
```

## 📊 Training Statistics

Example training on biblical and synaxarium datasets:
- **Training samples**: 61,576 texts
- **Total characters**: 6,789,143
- **Vocabulary size**: 30,000
- **Training time**: ~20 seconds
- **Storage saved**: ~2.3 MB with compression

## 📄 License

MIT License
