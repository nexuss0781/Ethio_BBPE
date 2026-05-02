# EthioBBPE

<div align="center">

![PyPI - Version](https://img.shields.io/pypi/v/EthioBBPE.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/EthioBBPE.svg)
![PyPI - License](https://img.shields.io/pypi/l/EthioBBPE.svg)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/Nexuss0781/Ethio-BBPE)
[![GitHub](https://img.shields.io/badge/GitHub-repo-blue)](https://github.com/nexuss0781/Ethio_BBPE)

</div>

## 📖 Overview

**EthioBBPE** is a professional, production-ready Byte Pair Encoding (BPE) tokenizer specifically designed for **Amharic**, **Ge'ez**, and **biblical texts**. It provides state-of-the-art tokenization for Ethiopian languages with support for ancient scripts and special characters.

### ✨ Key Features

- 🚀 **One-line Installation**: `pip install EthioBBPE`
- 🤖 **Auto-download**: Models automatically downloaded from Hugging Face Hub
- 📦 **Compressed Storage**: Gzip compression for efficient storage (65%+ size reduction)
- ⚡ **High Performance**: Optimized for speed with batch processing
- 🎯 **Perfect Reconstruction**: 100% accuracy on Amharic biblical texts
- 🔧 **Professional API**: Clean, intuitive interface inspired by Hugging Face
- 📊 **Production Ready**: Checkpointing, quantization, and comprehensive metrics

## 🚀 Quick Start

### Installation

```bash
pip install EthioBBPE
```

### Basic Usage

```python
from ethiobbpe import EthioBBPETokenizer

# Load pretrained tokenizer (auto-downloads from Hugging Face)
tokenizer = EthioBBPETokenizer.from_pretrained()

# Encode text
text = "ሰላም ለኢዮብ ዘኢነበበ ከንቶ ።"
encoded = tokenizer.encode(text)

print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")

# Decode back to text
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: {decoded}")
# Output: ሰላም ለኢዮብ ዘኢነበበ ከንቶ ። (100% accurate!)
```

### Advanced Usage

```python
from ethiobbpe import AutoTokenizer

# Alternative: Use AutoTokenizer factory
tokenizer = AutoTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")

# Batch encoding
texts = [
    "በመዠመሪያ፡እግዚአብሔር፡ሰማይንና፡ምድርን፡ፈጠረ።",
    "ወደ ቍስጥንጥንያ አገርም በደረሰች ጊዜ",
    "ሐዋርያ መንፈስ ይቤ እንዘ ያነክር ሕይወቶ"
]

encodings = tokenizer.encode_batch(texts)
for i, enc in enumerate(encodings):
    print(f"Text {i}: {len(enc.tokens)} tokens")

# Callable interface
result = tokenizer("ሰላም ዓለም")
print(result.tokens)

# Get vocabulary
vocab = tokenizer.get_vocab()
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
```

## 📊 Model Details

| Property | Value |
|----------|-------|
| **Model Name** | EthioBBPE_AmharicBible |
| **Vocabulary Size** | 16,000 tokens |
| **Training Data** | 61,769 lines (Synaxarium + Canon Biblical) |
| **Compression** | Gzip Level 9 (65%+ reduction) |
| **Reconstruction Accuracy** | 100% |
| **License** | Apache 2.0 |

### Training Datasets

1. **Synaxarium Dataset**: 366 religious texts
2. **Canon Biblical Dataset**: 61,403 Amharic-English parallel texts

Total corpus: ~27.5 MB of high-quality Amharic biblical text.

## 🛠️ API Reference

### `EthioBBPETokenizer`

Main tokenizer class with full encoding/decoding capabilities.

#### Methods

- `from_pretrained(model_name, cache_dir, force_download)`: Load pretrained model
- `from_file(file_path)`: Load from local file
- `encode(text, add_special_tokens, truncation, max_length)`: Encode single text
- `encode_batch(texts, ...)`: Encode multiple texts
- `decode(ids, skip_special_tokens)`: Decode token IDs
- `decode_batch(batch_ids, ...)`: Decode multiple sequences
- `get_vocab()`: Get vocabulary dictionary
- `get_vocab_size()`: Get vocabulary size
- `save(path)`: Save tokenizer to file

### `AutoTokenizer`

Factory class for automatic tokenizer loading.

```python
from ethiobbpe import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")
```

### `Encoding`

Wrapper object for encoding results with properties:
- `ids`: Token IDs (List[int])
- `tokens`: Token strings (List[str])
- `attention_mask`: Attention mask (List[int])
- `type_ids`: Token type IDs (List[int])
- `offsets`: Character offsets (List[tuple])
- `special_tokens_mask`: Special tokens mask (List[int])

## 📈 Performance

### Compression Benefits

| Format | Size | Compression |
|--------|------|-------------|
| tokenizer.json | 1.3 MB | Baseline |
| vocab.json.gz | 136 KB | **89.8%** |
| tokenizer_quantized_8bit.json.gz | 56 KB | **95.7%** |

### Speed

- Single encoding: < 1ms
- Batch encoding (32 texts): < 10ms
- Model download: Automatic caching for subsequent uses

## 🔬 Technical Details

### Architecture

- **Algorithm**: Byte Pair Encoding (BPE)
- **Pre-tokenization**: Whitespace splitting
- **Special Tokens**: `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`, `[UNK]`
- **Vocabulary**: 16,000 tokens optimized for Amharic/Ge'ez

### Production Features

1. **Checkpointing**: SHA256 integrity verification for fault-tolerant training
2. **Quantization**: 8-bit and 4-bit options for deployment optimization
3. **Metrics Tracking**: Comprehensive training statistics in JSON format
4. **Multiple Export Formats**: tokenizer.json, vocab.json.gz, quantized versions

## 📝 Examples

### Ge'ez Punctuation Handling

```python
tokenizer = EthioBBPETokenizer.from_pretrained()

# Ancient Ge'ez punctuation marks
geez_text = "፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠"
encoded = tokenizer.encode(geez_text)
print(f"Ge'ez punctuation: {encoded.tokens}")
# Single token for repeated marks!

decoded = tokenizer.decode(encoded.ids)
assert decoded == geez_text  # Perfect reconstruction
```

### Biblical Text Processing

```python
biblical_text = """
ሰላም ለኢዮብ ዘኢነበበ ከንቶ ። 
አመ አኀዞ አበቅ ወአመ አህጎለ ጥሪቶ ። 
ሐዋርያ መንፈስ ይቤ እንዘ ያነክር ሕይወቶ ።
"""

encoded = tokenizer.encode(biblical_text)
print(f"Token count: {len(encoded.tokens)}")
print(f"Perfect reconstruction: {tokenizer.decode(encoded.ids) == biblical_text}")
```

## 🤝 Integration

### With Hugging Face Transformers

```python
from transformers import AutoModel, AutoTokenizer
from ethiobbpe import EthioBBPETokenizer

# Use EthioBBPE as the tokenizer
ethio_tokenizer = EthioBBPETokenizer.from_pretrained()

# Integrate with your pipeline
def tokenize_for_model(text):
    encoding = ethio_tokenizer.encode(text)
    return {
        "input_ids": encoding.ids,
        "attention_mask": encoding.attention_mask
    }
```

### With PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset
from ethiobbpe import EthioBBPETokenizer

class AmharicDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length
        )
        return {
            "input_ids": torch.tensor(encoding.ids),
            "attention_mask": torch.tensor(encoding.attention_mask)
        }
```

## 📦 Installation Options

### Basic Installation

```bash
pip install EthioBBPE
```

### Development Installation

```bash
pip install EthioBBPE[dev]
```

### From Source

```bash
git clone https://github.com/nexuss0781/Ethio_BBPE.git
cd Ethio_BBPE
pip install -e .
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=ethiobbpe
```

## 📚 Resources

- [Hugging Face Model Page](https://huggingface.co/Nexuss0781/Ethio-BBPE)
- [GitHub Repository](https://github.com/nexuss0781/Ethio_BBPE)
- [Issue Tracker](https://github.com/nexuss0781/Ethio_BBPE/issues)

## 🙏 Acknowledgments

- Training data from [Synaxarium Dataset](https://huggingface.co/datasets/Nexuss0781/synaxarium_dataset)
- Training data from [Canon Biblical Dataset](https://huggingface.co/datasets/Nexuss0781/conon-biblical-am-en)
- Built with [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## 👤 Author

**Nexuss0781**  
Email: nexuss0781@gmail.com

---

<div align="center">

Made with ❤️ for Ethiopian Language NLP

[![Hugging Face](https://img.shields.io/badge/🤗-Download%20Model-yellow)](https://huggingface.co/Nexuss0781/Ethio-BBPE)
[![GitHub Stars](https://img.shields.io/github/stars/nexuss0781/Ethio_BBPE)](https://github.com/nexuss0781/Ethio_BBPE)

</div>
