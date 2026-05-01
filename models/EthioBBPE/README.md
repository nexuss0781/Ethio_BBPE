---
language:
- multilingual
tags:
- ethiobbpe
- bpe
- tokenizer
- byte-level
license: apache-2.0
datasets:
- user-provided
---

# EthioBBPE Tokenizer

This is a production-ready Byte-Level BPE tokenizer trained for robust text processing.

## Features
- **Byte-Level**: Handles any Unicode character without <UNK>.
- **Compressed Storage**: Supports gzip compression for efficient deployment.
- **Checkpointing**: Built-in safety checkpoints during training.

## Usage

### Transformers
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EthioBBPE")
```

### Tokenizers Library
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
```

## Training Configuration
```json
{
  "vocab_size": 32000,
  "min_frequency": 2,
  "show_progress": true,
  "special_tokens": [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]"
  ],
  "lowercase": false,
  "dropout": null,
  "data_dir": "./data",
  "model_save_dir": "models",
  "model_name": "EthioBBPE",
  "use_checkpoint": true,
  "checkpoint_dir": "./models/checkpoints",
  "save_compressed": true,
  "checkpoint_steps": null,
  "num_threads": -1
}
```
