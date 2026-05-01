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
- **Parallel Processing**: Optimized for multi-threaded training.

## Installation

```bash
pip install tokenizers transformers
```

## Usage

### With Transformers

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")

# Encode
encoded = tokenizer("Hello World!")
print(encoded.input_ids)

# Decode
decoded = tokenizer.decode(encoded.input_ids)
print(decoded)
```

### With Tokenizers Library

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

# Encode
encoded = tokenizer.encode("Hello World!")
print(encoded.ids)

# Decode
decoded = tokenizer.decode(encoded.ids)
print(decoded)
```

## Training Your Own

```python
from scripts.bbpe_trainer import BBPEConfig, EthioBBPETrainer

# Configure
config = BBPEConfig(
    vocab_size=30000,
    min_frequency=2,
    save_compressed=True
)

# Train
trainer = EthioBBPETrainer(config, output_dir="./models")
trainer.train(files="./data", use_checkpoint=True)
trainer.save(model_name="my_tokenizer")
```

## Advanced Features

### Checkpointing
Automatically saves checkpoints during training. Resume with `use_checkpoint=True`.

### Compression
Save models in compressed format (gzip) to reduce storage by ~60%.

### Logging
Built-in structured logging for monitoring training progress.

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 30000 | Size of the vocabulary |
| `min_frequency` | 2 | Minimum frequency for tokens |
| `special_tokens` | ["<pad>", "<unk>", "<s>", "</s>"] | Special tokens |
| `save_compressed` | True | Save in gzip format |
| `lowercase` | False | Lowercase input |

## License

Apache 2.0
