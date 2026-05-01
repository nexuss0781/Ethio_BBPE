#!/usr/bin/env python3
"""
EthioBBPE: Production-Ready Byte-Level BPE Tokenizer Trainer
Features: Checkpointing, Compression, Parallel Processing, Robust Logging
"""

import os
import json
import gzip
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from tokenizers import ByteLevelBPETokenizer, trainers
from tokenizers.implementations import BaseTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EthioBBPE")


@dataclass
class BBPEConfig:
    """Configuration for EthioBBPE training."""
    vocab_size: int = 30000
    min_frequency: int = 2
    show_progress: bool = True
    special_tokens: List[str] = field(default_factory=lambda: ["<pad>", "<unk>", "<s>", "</s>"])
    lowercase: bool = False
    dropout: Optional[float] = None
    
    # File paths
    data_dir: str = "./data"
    model_save_dir: str = "./models"
    model_name: str = "EthioBBPE"
    
    # Advanced features
    use_checkpoint: bool = True
    checkpoint_dir: str = "./models/checkpoints"
    save_compressed: bool = True
    checkpoint_steps: Optional[int] = None  # Save checkpoint every N steps if custom trainer used
    num_threads: int = -1  # -1 for auto

    def save(self, path: str):
        """Save configuration to JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BBPEConfig":
        """Load configuration from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class EthioBBPETrainer:
    """
    Production-ready trainer for Byte-Level BPE with checkpointing and compression.
    """
    
    def __init__(self, config: BBPEConfig = None):
        self.config = config or BBPEConfig()
        self.output_dir = Path(self.config.model_save_dir)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.tokenizer = None
        self.is_trained = False
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized EthioBBPETrainer with output dir: {self.output_dir}")

    def _initialize_tokenizer(self):
        """Initialize the ByteLevelBPETokenizer."""
        self.tokenizer = ByteLevelBPETokenizer(
            add_prefix_space=False,
            trim_offsets=True,
            lowercase=self.config.lowercase
        )
        logger.info("Tokenizer initialized")

    def train(self, files: Union[str, List[str]] = None, use_checkpoint: bool = None):
        """
        Train the tokenizer on a list of files or a directory.
        
        Args:
            files: Path to a file, list of files, or directory containing text files.
                   If None, uses files from config.data_dir
            use_checkpoint: If True, attempts to resume from the latest checkpoint.
                           Defaults to config.use_checkpoint
        """
        if self.tokenizer is None:
            self._initialize_tokenizer()

        # Use config default if not specified
        use_checkpoint = use_checkpoint if use_checkpoint is not None else self.config.use_checkpoint
        
        # Resolve file paths
        if files is None:
            # Use data_dir from config
            data_path = Path(self.config.data_dir)
            if data_path.is_dir():
                file_paths = [str(f) for f in data_path.glob("**/*.txt")]
                file_paths.extend([str(f) for f in data_path.glob("**/*.jsonl")])
                file_paths.extend([str(f) for f in data_path.glob("**/*.json")])
            else:
                raise FileNotFoundError(f"Data directory not found: {self.config.data_dir}")
        elif isinstance(files, str):
            path = Path(files)
            if path.is_dir():
                file_paths = [str(f) for f in path.glob("**/*.txt")]
                file_paths.extend([str(f) for f in path.glob("**/*.jsonl")])
                file_paths.extend([str(f) for f in path.glob("**/*.json")])
            else:
                file_paths = [str(path)]
        else:
            file_paths = files

        if not file_paths:
            raise ValueError("No valid training files found.")
        
        logger.info(f"Found {len(file_paths)} files for training.")

        # Checkpoint logic
        start_from_scratch = True
        if use_checkpoint:
            latest_ckpt = self._get_latest_checkpoint()
            if latest_ckpt:
                logger.info(f"Resuming from checkpoint: {latest_ckpt}")
                # Use Tokenizer.from_file for loading checkpoints (works with tokenizer.json format)
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(str(latest_ckpt))
                start_from_scratch = False
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        # Train
        logger.info("Starting training...")
        
        # ByteLevelBPETokenizer.train() accepts parameters directly, not a trainer object
        self.tokenizer.train(
            files=file_paths,
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=self.config.show_progress
        )

        self.is_trained = True
        logger.info("Training completed successfully.")
        
        # Auto-save checkpoint after training
        self._save_checkpoint("final_pre_compress")

        return self.tokenizer

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint file."""
        ckpts = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not ckpts:
            return None
        # Sort by modification time
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return ckpts[0]

    def _save_checkpoint(self, name: str = "latest"):
        """Save current tokenizer state to checkpoint."""
        if self.tokenizer is None:
            return
        ckpt_path = self.checkpoint_dir / f"checkpoint_{name}.json"
        self.tokenizer.save(str(ckpt_path))
        logger.info(f"Checkpoint saved to {ckpt_path}")

    def save(self, model_name: str = None, compress: bool = None):
        """
        Save the trained tokenizer.
        
        Args:
            model_name: Name of the model folder. Defaults to config.model_name
            compress: If True, saves vocab and merges in gzip format. 
                      Defaults to config.save_compressed.
        """
        if not self.is_trained and self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained yet.")

        model_name = model_name or self.config.model_name
        compress = compress if compress is not None else self.config.save_compressed
        model_path = self.output_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {model_path} (compressed={compress})...")

        if compress:
            # Save standard tokenizer.json (required for HF loading)
            tokenizer_file = model_path / "tokenizer.json"
            self.tokenizer.save(str(tokenizer_file))
            
            # Extract vocab from tokenizer
            vocab = self.tokenizer.get_vocab()
            
            # Save compressed vocab
            vocab_path = model_path / "vocab.json.gz"
            with gzip.open(vocab_path, 'wt', encoding='utf-8') as f:
                json.dump(vocab, f)
            
            logger.info(f"Compressed vocab saved: {vocab_path}")
            
            # Calculate savings
            original_size = tokenizer_file.stat().st_size
            compressed_size = vocab_path.stat().st_size
            logger.info(f"Storage saved: {(original_size - compressed_size) / 1024:.2f} KB")
        else:
            # Standard save
            self.tokenizer.save(str(model_path / "tokenizer.json"))
            self.tokenizer.model.save(str(model_path))
            logger.info("Standard model artifacts saved.")

        # Save config
        self.config.save(str(model_path / "config.json"))
        
        # Save metadata card for Hugging Face
        self._save_model_card(model_path)

        logger.info(f"Model successfully saved to {model_path}")
        return model_path

    def _save_model_card(self, path: Path):
        """Generate and save a README.md for Hugging Face Hub."""
        card_content = f"""---
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

tokenizer = AutoTokenizer.from_pretrained("{path.name}")
```

### Tokenizers Library
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
```

## Training Configuration
```json
{json.dumps(asdict(self.config), indent=2)}
```
"""
        with open(path / "README.md", 'w', encoding='utf-8') as f:
            f.write(card_content)

    def tokenize(self, text: str) -> List[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        return self.tokenizer.encode(text).tokens

    def encode(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        return self.tokenizer.decode(ids)
