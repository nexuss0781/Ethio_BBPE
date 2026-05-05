"""
EthioBBPE: Enterprise-grade Amharic & Ge'ez Biblical Text Tokenizer

A high-performance Byte Pair Encoding (BPE) tokenizer optimized for 
Amharic, Ge'ez, and biblical texts with support for ancient scripts.

This package provides production-ready tokenization capabilities with:
- Automatic model downloading from Hugging Face Hub
- Batch processing for efficient inference
- Full type hints for IDE support
- Comprehensive error handling

Example:
    >>> from ethiobbpe import EthioBBPETokenizer
    >>> tokenizer = EthioBBPETokenizer.from_pretrained()
    >>> encoded = tokenizer.encode("ሰላም ለኢዮብ")
    >>> print(encoded.tokens)
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Nexuss0781"
__email__ = "nexuss0781@gmail.com"
__license__ = "Apache-2.0"

from .tokenizer import EthioBBPETokenizer, AutoTokenizer, Encoding

__all__ = [
    "EthioBBPETokenizer",
    "AutoTokenizer",
    "Encoding",
    "__version__",
]
