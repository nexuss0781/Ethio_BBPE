# 🇪🇹 EthioBBPE - Amharic Biblical Tokenizer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging_Face-yellow)](https://huggingface.co/Nexuss0781/Ethio-BBPE)
[![Amharic](https://img.shields.io/badge/Language-Amharic-green.svg)](https://en.wikipedia.org/wiki/Amharic)
[![Tokenizer Type](https://img.shields.io/badge/Type-Byte--level_BPE-orange.svg)](https://huggingface.co/docs/tokenizers/index)

A **production-ready Byte-level BPE tokenizer** specifically trained on **Amharic biblical and religious texts**, achieving **perfect reconstruction** of complex Ge'ez script, ancient punctuation, and liturgical content.

## ✨ Features

- ✅ **Perfect Reconstruction**: 100% accuracy on all test samples including ancient Ge'ez punctuation
- ✅ **Specialized Vocabulary**: Trained on 61,769 lines of Amharic biblical texts (Synaxarium + Canon Bible)
- ✅ **Compressed Storage**: Gzip compression (level 9) reduces model size by **89.8%** (1.3MB → 136KB)
- ✅ **Production Ready**: Checkpointing, metrics tracking, and comprehensive error handling
- ✅ **Ge'ez Script Support**: Full support for Ethiopic characters, numerals, and liturgical punctuation marks
- ✅ **Hugging Face Compatible**: Directly usable with `transformers` models

## 📊 Training Data

| Dataset | Source | Texts | Description |
|---------|--------|-------|-------------|
| **Synaxarium** | [Nexuss0781/synaxarium](https://huggingface.co/datasets/Nexuss0781/synaxarium) | 366 | Daily synaxarium readings in Amharic |
| **Canon Biblical** | [Nexuss0781/conon-biblical-am-en](https://huggingface.co/datasets/Nexuss0781/conon-biblical-am-en) | 61,403 | Amharic-English biblical texts |
| **Total** | - | **61,769** | **15.43 MB** combined corpus |

### Training Configuration

```json
{
  "vocab_size": 16000,
  "min_frequency": 2,
  "special_tokens": ["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
  "lowercase": false,
  "compression": "gzip (level 9)",
  "checkpointing": true
}
```

## 🎯 Performance Metrics

| Metric | Result |
|--------|--------|
| **Perfect Reconstruction** | ✅ **100%** |
| **Ge'ez Punctuation** | ✅ Perfect (1 token for `፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠`) |
| **Synaxarium Text** | ✅ Perfect (66 tokens) |
| **Biblical Text** | ✅ Perfect (82 tokens) |
| **Compression Ratio** | **89.8%** (1.3MB → 136KB) |
| **Training Time** | ~17 seconds |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Load from Hugging Face Hub

```python
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# Download and load tokenizer
tokenizer_path = hf_hub_download("Nexuss0781/Ethio-BBPE", "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)

# Encode Amharic text
text = "ሰላም ለኢዮብ ዘኢነበበ ከንቶ ።"
encoded = tokenizer.encode(text)

print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {tokenizer.decode(encoded.ids)}")
```

### Train Your Own Tokenizer

```bash
# Basic training
python scripts/train_tokenizer.py --data_dir ./data --model_name EthioBBPE --vocab_size 16000

# Advanced training with compression and checkpointing
python scripts/train_tokenizer.py \
    --data_dir ./data \
    --model_name EthioBBPE \
    --vocab_size 16000 \
    --use_checkpoint \
    --save_compressed
```

## 📁 Model Files

| File | Size | Description |
|------|------|-------------|
| `tokenizer.json` | 1.3 MB | Standard tokenizer format |
| `vocab.json.gz` | 136 KB | Compressed vocabulary (89.8% smaller) |
| `config.json` | 431 B | Training configuration |
| `training_metrics.json` | 1.2 KB | Comprehensive training metrics |
| `README.md` | - | Model documentation |

## 🔬 Technical Details

### Architecture
- **Type**: Byte-level BPE (BBPE)
- **Vocabulary Size**: 16,000 tokens
- **Special Tokens**: `<pad>`, `<unk>`, `<s>`, `</s>`, `<mask>`
- **Minimum Frequency**: 2 occurrences

### Preprocessing
- No lowercasing (preserves Ge'ez case distinctions)
- No prefix space (optimal for Amharic morphology)
- Unicode normalization enabled

### Compression
- **Algorithm**: Gzip (level 9)
- **Original Size**: 1.3 MB
- **Compressed Size**: 136 KB
- **Space Saved**: 89.8%

## 🧪 Testing & Validation

All test cases achieve **perfect reconstruction**:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("models/EthioBBPE/tokenizer.json")

test_cases = [
    ("Ge'ez Punctuation", "፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠"),
    ("Synaxarium", "ሰላም ለኢዮብ ዘኢነበበ ከንቶ ።"),
    ("Biblical", "ወደ ቍስጥንጥንያ አገርም በደረሰች ጊዜ")
]

for name, text in test_cases:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)
    assert text == decoded, f"{name} failed!"
    print(f"✅ {name}: Perfect ({len(encoded.ids)} tokens)")
```

## 📂 Project Structure

```text
Ethio_BBPE/
├── data/                       # Raw training data
│   ├── synaxarium_dataset.parquet
│   ├── canon_biblical_am_en.parquet
│   └── combined_corpus.txt     # Prepared training corpus
├── models/                     # Output directory for trained models
│   ├── EthioBBPE/             # Trained tokenizer
│   │   ├── tokenizer.json     # Main tokenizer file
│   │   ├── vocab.json.gz      # Compressed vocabulary
│   │   ├── config.json        # Training configuration
│   │   ├── training_metrics.json
│   │   └── README.md          # Model card
│   └── checkpoints/           # Training checkpoints
├── scripts/
│   ├── bbpe_trainer.py        # Core logic (BBPEConfig, EthioBBPETrainer)
│   ├── train_tokenizer.py     # CLI entry point
│   └── prepare_data.py        # Data preparation from parquet
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🛠️ Advanced Features

### Checkpointing
Automatic checkpointing during training allows resumption from interruptions:
```bash
python scripts/train_tokenizer.py --data_dir ./data --use_checkpoint
```

### Custom Vocabulary Size
```bash
python scripts/train_tokenizer.py --data_dir ./data --vocab_size 32000
```

### Alternative Compression
```bash
python scripts/train_tokenizer.py --data_dir ./data --save_compressed
# Supports: gzip, bz2, lzma
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

### Uploading to Hub
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./models/EthioBBPE",
    repo_id="nexuss0781/Ethio-BBPE",
    repo_type="model",
    token="YOUR_HF_TOKEN"
)
```

## 📚 Datasets

This tokenizer was trained on two specialized Amharic biblical datasets:

1. **Synaxarium Dataset**: Daily readings from the Ethiopian Orthodox Synaxarium containing lives of saints and biblical narratives
2. **Canon Biblical Dataset**: Comprehensive Amharic-English parallel biblical texts

Both datasets are available on Hugging Face under the `Nexuss0781` organization.

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Datasets**: [Nexuss0781/synaxarium](https://huggingface.co/datasets/Nexuss0781/synaxarium) and [Nexuss0781/conon-biblical-am-en](https://huggingface.co/datasets/Nexuss0781/conon-biblical-am-en)
- **Library**: [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
- **Script**: Ethiopic (Ge'ez) Unicode block U+1200–U+137F

## 📬 Contact & Support

- **GitHub**: [nexuss0781/Ethio_BBPE](https://github.com/nexuss0781/Ethio_BBPE)
- **Hugging Face**: [Nexuss0781/Ethio-BBPE](https://huggingface.co/Nexuss0781/Ethio-BBPE)
- **Issues**: Please open an issue on GitHub for bugs or feature requests

---

**Made with ❤️ for the Amharic NLP Community**

*Last Updated: May 2026*
