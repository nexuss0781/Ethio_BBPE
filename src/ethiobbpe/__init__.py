"""
EthioBBPE: Professional Amharic Biblical Text Tokenizer

A high-performance Byte Pair Encoding tokenizer optimized for 
Amharic, Ge'ez, and biblical texts with support for ancient scripts.
"""

__version__ = "1.0.0"
__author__ = "Nexuss0781"
__email__ = "nexuss0781@gmail.com"
__license__ = "Apache-2.0"

from .tokenizer import EthioBBPETokenizer, AutoTokenizer

__all__ = [
    "EthioBBPETokenizer",
    "AutoTokenizer",
    "__version__",
]
