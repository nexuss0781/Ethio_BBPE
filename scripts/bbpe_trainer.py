"""
Byte-Level BPE Tokenizer Training Pipeline

This module provides a comprehensive architecture for training Byte-Level BPE (BBPE) tokenizers
using Hugging Face's `tokenizers` library. It includes data preprocessing, training configuration,
and model serialization utilities.
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass, field, asdict
from tokenizers import ByteLevelBPETokenizer, Tokenizer


@dataclass
class BBPEConfig:
    """Configuration class for BBPE tokenizer training."""
    
    # Vocabulary settings
    vocab_size: int = 30000
    min_frequency: int = 2
    
    # Special tokens
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
        "<mask>"
    ])
    
    # Byte-level settings
    lowercase: bool = False
    add_prefix_space: bool = True
    trim_offsets: bool = False
    
    # Training settings
    show_progress: bool = True
    initial_alphabet: List[str] = field(default_factory=list)
    
    # Paths
    data_dir: str = "data"
    model_save_dir: str = "models"
    model_name: str = "bbpe_tokenizer"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'BBPEConfig':
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class BBPETrainer:
    """
    End-to-end trainer for Byte-Level BPE tokenizers.
    
    This class handles the complete training pipeline including:
    - Data loading and preprocessing
    - Tokenizer initialization with byte-level encoding
    - BPE training with configurable parameters
    - Model saving and loading
    """
    
    def __init__(self, config: Optional[BBPEConfig] = None):
        """
        Initialize the BBPE trainer.
        
        Args:
            config: BBPEConfig instance. If None, default config is used.
        """
        self.config = config or BBPEConfig()
        self.tokenizer: Optional[ByteLevelBPETokenizer] = None
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories for data and models."""
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    def initialize_tokenizer(self) -> ByteLevelBPETokenizer:
        """
        Initialize a new ByteLevelBPETokenizer with byte-level encoding.
        
        Returns:
            Initialized ByteLevelBPETokenizer instance
        """
        tokenizer = ByteLevelBPETokenizer(
            add_prefix_space=self.config.add_prefix_space,
            trim_offsets=self.config.trim_offsets,
            lowercase=self.config.lowercase,
        )
        self.tokenizer = tokenizer
        return tokenizer
    
    def get_training_files(self) -> List[str]:
        """
        Get list of text files for training from the data directory.
        
        Returns:
            List of file paths to text files
        """
        data_path = Path(self.config.data_dir)
        text_files = []
        
        # Support multiple text file extensions
        extensions = ['.txt', '.jsonl', '.json']
        
        for ext in extensions:
            text_files.extend(list(data_path.glob(f'*{ext}')))
        
        if not text_files:
            raise FileNotFoundError(
                f"No training files found in {data_path}. "
                f"Please add .txt, .json, or .jsonl files to this directory."
            )
        
        return [str(f) for f in text_files]
    
    def train(self, 
              files: Optional[List[str]] = None,
              config_override: Optional[dict] = None) -> ByteLevelBPETokenizer:
        """
        Train the BBPE tokenizer on the provided files.
        
        Args:
            files: List of file paths to train on. If None, uses files from data_dir.
            config_override: Optional dictionary to override config parameters.
            
        Returns:
            Trained ByteLevelBPETokenizer instance
        """
        # Apply config overrides if provided
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            self.initialize_tokenizer()
        
        # Get training files
        if files is None:
            files = self.get_training_files()
        
        print(f"Training on {len(files)} file(s)...")
        for f in files:
            print(f"  - {f}")
        
        # Train the tokenizer using the new API (tokenizers >= 0.15)
        print("\nStarting training...")
        self.tokenizer.train(
            files=files,
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=self.config.show_progress,
        )
        print("Training completed!")
        
        # Print vocabulary statistics
        vocab_size = self.tokenizer.get_vocab_size()
        print(f"\nVocabulary size: {vocab_size}")
        print(f"Special tokens: {self.config.special_tokens}")
        
        return self.tokenizer
    
    def save(self, model_name: Optional[str] = None) -> str:
        """
        Save the trained tokenizer to disk.
        
        Args:
            model_name: Name for the saved model. If None, uses config.model_name.
            
        Returns:
            Path to the saved model directory
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save. Please train first.")
        
        name = model_name or self.config.model_name
        save_path = Path(self.config.model_save_dir) / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer files
        self.tokenizer.save_model(str(save_path))
        
        # Save configuration
        config_path = save_path / "config.json"
        self.config.save(str(config_path))
        
        # Save tokenizer.json (full tokenizer state)
        tokenizer_json_path = save_path / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_json_path))
        
        print(f"\nTokenizer saved to: {save_path}")
        print(f"  - vocab.json")
        print(f"  - merges.txt")
        print(f"  - config.json")
        print(f"  - tokenizer.json")
        
        return str(save_path)
    
    def load(self, model_path: str) -> ByteLevelBPETokenizer:
        """
        Load a pre-trained tokenizer from disk.
        
        Args:
            model_path: Path to the directory containing tokenizer files.
            
        Returns:
            Loaded ByteLevelBPETokenizer instance
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Try to load tokenizer.json first (preferred method for tokenizers >= 0.15)
        tokenizer_json = model_path / "tokenizer.json"
        if tokenizer_json.exists():
            # Use the generic Tokenizer class to load the full tokenizer state
            base_tokenizer = Tokenizer.from_file(str(tokenizer_json))
            # Wrap it as ByteLevelBPETokenizer for consistent API
            self.tokenizer = ByteLevelBPETokenizer(
                add_prefix_space=self.config.add_prefix_space,
                trim_offsets=self.config.trim_offsets,
                lowercase=self.config.lowercase,
            )
            # Copy the vocabulary and merges from the loaded tokenizer
            self.tokenizer = base_tokenizer
        else:
            # Fall back to loading vocab.json and merges.txt
            vocab_file = model_path / "vocab.json"
            merges_file = model_path / "merges.txt"
            
            if not vocab_file.exists() or not merges_file.exists():
                raise FileNotFoundError(
                    f"Required files not found in {model_path}. "
                    f"Need either tokenizer.json or both vocab.json and merges.txt"
                )
            
            self.tokenizer = ByteLevelBPETokenizer.from_file(
                str(vocab_file), str(merges_file)
            )
        
        # Load config if exists
        config_file = model_path / "config.json"
        if config_file.exists():
            self.config = BBPEConfig.load(str(config_file))
        
        print(f"Tokenizer loaded from: {model_path}")
        return self.tokenizer
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded. Please train or load first.")
        return self.tokenizer.encode(text, **kwargs).ids
    
    def decode(self, ids: List[int], **kwargs) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded. Please train or load first.")
        return self.tokenizer.decode(ids, **kwargs)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to token strings."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded. Please train or load first.")
        return self.tokenizer.encode(text).tokens


def main():
    """Example usage of the BBPE trainer."""
    # Create configuration
    config = BBPEConfig(
        vocab_size=30000,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
        data_dir="data",
        model_save_dir="models",
        model_name="my_bbpe_tokenizer",
    )
    
    # Initialize trainer
    trainer = BBPETrainer(config)
    
    # Train the tokenizer
    trainer.train()
    
    # Save the tokenizer
    save_path = trainer.save()
    
    # Test encoding/decoding
    test_text = "Hello, world! This is a test of the BBPE tokenizer."
    encoded = trainer.encode(test_text)
    decoded = trainer.decode(encoded)
    tokens = trainer.tokenize(test_text)
    
    print(f"\nTest encoding:")
    print(f"  Input: {test_text}")
    print(f"  Tokens: {tokens}")
    print(f"  IDs: {encoded}")
    print(f"  Decoded: {decoded}")


if __name__ == "__main__":
    main()
