# 🎉 EthioBBPE v1.0.0 - Production Release

## ✅ Successfully Published to PyPI

**Package URL**: https://pypi.org/project/EthioBBPE/1.0.0/

### Installation

```bash
pip install EthioBBPE
```

### Quick Start

```python
from ethiobbpe import EthioBBPETokenizer

# Auto-downloads model from Hugging Face
tokenizer = EthioBBPETokenizer.from_pretrained()

# Encode Amharic text
text = "ሰላም ለኢዮብ ዘኢነበበ"
encoded = tokenizer.encode(text)
print(f"Tokens: {encoded.tokens}")

# Perfect reconstruction
decoded = tokenizer.decode(encoded.ids)
assert decoded == text  # ✓ 100% accuracy!
```

## 📦 What's Included

### Core Features
- ✅ Professional PyPI package with clean API
- ✅ Automatic model download from Hugging Face Hub
- ✅ Support for Amharic, Ge'ez, and biblical texts
- ✅ Perfect reconstruction of ancient scripts
- ✅ Batch processing capabilities
- ✅ Truncation and special tokens handling

### Technical Specifications
- **Vocabulary Size**: 16,000 tokens
- **Training Data**: 61,769 lines (Synaxarium + Canon Biblical)
- **Compression**: Gzip level 9 (65%+ size reduction)
- **Reconstruction Accuracy**: 100%
- **License**: Apache 2.0

### Files Structure
```
EthioBBPE/
├── src/ethiobbpe/
│   ├── __init__.py          # Package exports
│   └── tokenizer.py         # Core tokenizer implementation
├── tests/
│   └── test_tokenizer.py    # Comprehensive test suite
├── pyproject.toml           # Professional build configuration
├── README.md                # Beautiful documentation
└── LICENSE                  # Apache 2.0 license
```

## 🔗 Links

- **PyPI**: https://pypi.org/project/EthioBBPE/
- **GitHub**: https://github.com/nexuss0781/Ethio_BBPE
- **Hugging Face**: https://huggingface.co/Nexuss0781/Ethio-BBPE
- **Issues**: https://github.com/nexuss0781/Ethio_BBPE/issues

## 👤 Author

**Nexuss0781**  
Email: nexuss0781@gmail.com

---

Made with ❤️ for Ethiopian Language NLP
