---
language:
  - code
license: mit
tags:
  - byte-level-bpe
  - tokenizer
  - bbpe
  - tokenizers
pipeline_tag: token-classification
library_name: tokenizers
datasets: []
metrics:
  - vocabulary-size
---

# EthioBBPE: Byte-Level BPE Tokenizer

This is a Byte-Level BPE (BBPE) tokenizer trained using Hugging Face's `tokenizers` library. It handles diverse Unicode scripts and complex morphological structures seamlessly.

## Features

- **Byte-Level Encoding**: Robust against unknown characters, ensuring no `<UNK>` tokens
- **Universal Script Support**: Handles any Unicode character efficiently
- **Hugging Face Compatible**: Directly usable with `transformers` models
- **Efficient**: Fast encoding/decoding with optimized C++ backend

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
text = "Hello world! This is a test."
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
inputs = tokenizer("The quick brown fox jumps over the lazy dog.")
print(inputs)
```

## Training Details

- **Model Type**: Byte-Level BPE
- **Vocabulary Size**: 30,000 tokens
- **Minimum Frequency**: 2
- **Special Tokens**: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`

## Repository Structure

The full training codebase is available at:
- **GitHub**: [nexuss0781/Ethio_BBPE](https://github.com/nexuss0781/Ethio_BBPE)

## License

MIT License
