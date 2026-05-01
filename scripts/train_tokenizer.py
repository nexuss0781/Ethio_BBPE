#!/usr/bin/env python3
"""
Command-line interface for training BBPE tokenizers.

Usage:
    python train_tokenizer.py --data_dir ./data --vocab_size 30000 --model_name my_tokenizer
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bbpe_trainer import BBPETrainer, BBPEConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Byte-Level BPE (BBPE) tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing training text files (.txt, .json, .jsonl)",
    )
    
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="Specific files to train on (overrides data_dir)",
    )
    
    # Model arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30000,
        help="Target vocabulary size",
    )
    
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens to be included in vocabulary",
    )
    
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
        help="Special tokens to add to the vocabulary",
    )
    
    # Training options
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert text to lowercase before tokenization",
    )
    
    parser.add_argument(
        "--no_prefix_space",
        action="store_true",
        help="Disable adding prefix space (default: add prefix space)",
    )
    
    parser.add_argument(
        "--show_progress",
        action="store_true",
        default=True,
        help="Show training progress bar",
    )
    
    parser.add_argument(
        "--no_progress",
        action="store_false",
        dest="show_progress",
        help="Hide training progress bar",
    )
    
    # Output arguments
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models",
        help="Directory to save the trained tokenizer",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="bbpe_tokenizer",
        help="Name for the saved tokenizer model",
    )
    
    # Config file arguments
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to JSON config file (overrides other arguments)",
    )
    
    parser.add_argument(
        "--save_config",
        type=str,
        default=None,
        help="Path to save the configuration JSON file",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI training."""
    args = parse_args()
    
    # Load config from file if provided
    if args.config_file:
        print(f"Loading configuration from {args.config_file}")
        config = BBPEConfig.load(args.config_file)
    else:
        # Create config from arguments
        config = BBPEConfig(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=args.special_tokens,
            lowercase=args.lowercase,
            add_prefix_space=not args.no_prefix_space,
            show_progress=args.show_progress,
            data_dir=args.data_dir,
            model_save_dir=args.model_save_dir,
            model_name=args.model_name,
        )
    
    # Save config if requested
    if args.save_config:
        config.save(args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    # Initialize trainer
    trainer = BBPETrainer(config)
    
    # Get training files
    if args.files:
        print(f"Using specified files: {args.files}")
        files = args.files
    else:
        files = None  # Will use files from data_dir
    
    # Train the tokenizer
    try:
        trainer.train(files=files)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo fix this:")
        print(f"  1. Add your training data to the '{args.data_dir}' directory")
        print("  2. Supported formats: .txt, .json, .jsonl")
        print("  3. Or specify files directly with --files flag")
        sys.exit(1)
    
    # Save the tokenizer
    save_path = trainer.save()
    
    # Test the tokenizer
    print("\n" + "="*60)
    print("TESTING TOKENIZER")
    print("="*60)
    
    test_texts = [
        "Hello, world!",
        "This is a test of the BBPE tokenizer.",
        "Special characters: @#$%^&*()",
        "Numbers: 12345 and words mixed together.",
    ]
    
    for text in test_texts:
        encoded = trainer.encode(text)
        tokens = trainer.tokenize(text)
        decoded = trainer.decode(encoded)
        
        print(f"\nInput:    {text}")
        print(f"Tokens:   {tokens}")
        print(f"IDs:      {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"Decoded:  {decoded}")
    
    print("\n" + "="*60)
    print(f"Tokenizer training complete!")
    print(f"Model saved to: {save_path}")
    print("="*60)


if __name__ == "__main__":
    main()
