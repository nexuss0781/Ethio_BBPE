---
language:
  - am  # Amharic
  - om  # Oromo
  - ti  # Tigrinya
  - so  # Somali
license: mit
tags:
  - byte-level-bpe
  - tokenizer
  - ethiopian-languages
  - ge'ez-script
  - amharic
  - oromo
  - tigrinya
  - somali
pipeline_tag: token-classification
library_name: tokenizers
datasets:
  - nexuss0781/ethio-corpus
metrics:
  - vocabulary-size
---

# 🇪🇹 EthioBBPE: Byte-Level BPE Tokenizer for Ethiopian Languages

This is a Byte-Level BPE (BBPE) tokenizer specifically trained for Ethiopian languages including Amharic, Oromo, Tigrinya, and Somali. It handles the complex Ge'ez script and Latin-based orthographies seamlessly.

## Features

- **Byte-Level Encoding**: Robust against unknown characters, ensuring no `<UNK>` tokens
- **Optimized for Ethiopic Script**: Handles complex Ge'ez script characters efficiently
- **Multi-language Support**: Works across Amharic, Oromo, Tigrinya, Somali, and more
- **Hugging Face Compatible**: Directly usable with `transformers` models

## Installation

```bash
pip install tokenizers
```

## Usage

### Load the Tokenizer

```python
from tokenizers import Tokenizer

# Load from Hugging Face Hub
tokenizer = Tokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")

# Encode text
text = "ሰላም ልዑል እንዴት ነህ? (Hello, how are you?)"
encoded = tokenizer.encode(text)

print(f"Token IDs: {encoded.ids}")
print(f"Tokens: {encoded.tokens}")

# Decode back
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: {decoded}")
```

### Using with Transformers

```python
from transformers import AutoTokenizer

# Load as a fast tokenizer
tokenizer = AutoTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE", use_fast=True)

# Tokenize
inputs = tokenizer("የኢትዮጵያ ህዝብ በጣም ብዙ ነው።")
print(inputs)
```

## Training Details

- **Model Type**: Byte-Level BPE
- **Vocabulary Size**: 32,000 tokens
- **Minimum Frequency**: 2
- **Special Tokens**: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- **Training Data**: Ethiopian language corpora (Amharic, Oromo, Tigrinya, Somali)

## Supported Languages

| Language | Code | Script |
|----------|------|--------|
| Amharic | am | Ge'ez (Ethiopic) |
| Oromo | om | Latin |
| Tigrinya | ti | Ge'ez (Ethiopic) |
| Somali | so | Latin |
| Afar | aa | Latin |
| Sidamo | sid | Latin |

## Examples

### Amharic (Ge'ez Script)
```python
text = "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ሀገር ናት።"
tokens = tokenizer.encode(text).tokens
```

### Oromo (Latin Script)
```python
text = "Itoophiyaan biyya gaafa afriikaa argamtuudha."
tokens = tokenizer.encode(text).tokens
```

### Mixed Language
```python
text = "ሰላም! Hello! Akkam! How are you?"
tokens = tokenizer.encode(text).tokens
```

## Repository Structure

The full training codebase is available at:
- **GitHub**: [nexuss0781/Ethio_BBPE](https://github.com/nexuss0781/Ethio_BBPE)

## License

MIT License

## Acknowledgments

Built for the Ethiopian NLP community to foster better language understanding and generation capabilities.
