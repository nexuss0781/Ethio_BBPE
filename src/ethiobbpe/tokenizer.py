"""
Core tokenizer implementation for EthioBBPE.

Provides a clean, professional API for tokenizing Amharic and Ge'ez texts.
Enterprise-grade tokenizer with support for biblical texts and ancient scripts.
"""

from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, TYPE_CHECKING
from functools import lru_cache
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import Encoding as HFEncoding

try:
    from tokenizers import Tokenizer as HFTokenizer  # pragma: no cover
    from tokenizers.models import BPE  # pragma: no cover
    from tokenizers.trainers import BpeTrainer  # pragma: no cover
    from tokenizers.pre_tokenizers import Whitespace  # pragma: no cover
    from tokenizers import Encoding as HFEncoding  # pragma: no cover
    HAS_TOKENIZERS = True  # pragma: no cover
except ImportError:  # pragma: no cover
    HAS_TOKENIZERS = False  # pragma: no cover
    HFEncoding = Any  # type: ignore  # pragma: no cover

from huggingface_hub import hf_hub_download


class EthioBBPETokenizer:
    """
    Professional Amharic Biblical Text Tokenizer.
    
    Provides encoding, decoding, and batch processing capabilities
    with automatic model downloading from Hugging Face Hub.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        model_name (str): Name of the pretrained model
        
    Example:
        >>> from ethiobbpe import EthioBBPETokenizer
        >>> tokenizer = EthioBBPETokenizer.from_pretrained()
        >>> encoded = tokenizer.encode("ሰላም ለኢዮብ")
        >>> print(encoded.tokens)
        >>> decoded = tokenizer.decode(encoded.ids)
        >>> print(decoded)
    """
    
    MODEL_NAME = "Nexuss0781/Ethio-BBPE"
    DEFAULT_VOCAB_SIZE = 16000
    
    def __init__(self, tokenizer_obj: Any, config: Optional[Dict] = None):
        """
        Initialize the tokenizer.
        
        Args:
            tokenizer_obj: Underlying tokenizer object (HuggingFace or custom)
            config: Optional configuration dictionary
        """
        self._tokenizer = tokenizer_obj
        self._config = config or {}
        self.vocab_size = self._config.get("vocab_size", self.DEFAULT_VOCAB_SIZE)
        self.model_name = self._config.get("model_name", self.MODEL_NAME)
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False
    ) -> "EthioBBPETokenizer":
        """
        Load a pretrained tokenizer from Hugging Face Hub.
        
        Args:
            model_name: Model identifier on Hugging Face Hub.
                       Defaults to "Nexuss0781/Ethio-BBPE"
            cache_dir: Directory to cache downloaded models
            force_download: Force re-download even if cached
            
        Returns:
            EthioBBPETokenizer instance
            
        Example:
            >>> tokenizer = EthioBBPETokenizer.from_pretrained()
            >>> # Or specify custom model
            >>> tokenizer = EthioBBPETokenizer.from_pretrained("my-custom-model")
        """
        model_id = model_name or cls.MODEL_NAME
        
        try:
            # Download tokenizer.json from Hugging Face
            tokenizer_path = hf_hub_download(
                repo_id=model_id,
                filename="tokenizer.json",
                cache_dir=cache_dir,
                force_download=force_download
            )
            
            if not HAS_TOKENIZERS:  # pragma: no cover
                raise ImportError(
                    "tokenizers library required. Install with: pip install tokenizers"
                )
            
            tokenizer_obj = HFTokenizer.from_file(tokenizer_path)
            
            # Try to load config
            try:
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    cache_dir=cache_dir
                )
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Ensure model_name is set correctly in config
                config["model_name"] = model_id
            except Exception:
                config = {"vocab_size": cls.DEFAULT_VOCAB_SIZE, "model_name": model_id}
            
            return cls(tokenizer_obj, config)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer from {model_id}: {str(e)}"
            )
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "EthioBBPETokenizer":
        """
        Load tokenizer from a local file.
        
        Args:
            file_path: Path to tokenizer.json or vocab.json.gz file
            
        Returns:
            EthioBBPETokenizer instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")
        
        if not HAS_TOKENIZERS:  # pragma: no cover
            raise ImportError(
                "tokenizers library required. Install with: pip install tokenizers"
            )
        
        if file_path.suffix == '.json':
            tokenizer_obj = HFTokenizer.from_file(str(file_path))
            config = {}
        elif file_path.name.endswith('.json.gz'):
            # Load compressed vocab and create tokenizer
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                vocab = json.load(f)
            
            tokenizer_obj = HFTokenizer(BPE(unk_token="[UNK]"))
            tokenizer_obj.add_special_tokens(["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
            config = {"vocab_size": len(vocab)}
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls(tokenizer_obj, config)
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None
    ) -> Encoding:
        """
        Encode a single text into tokens.
        
        Args:
            text: Input text string
            add_special_tokens: Add special tokens (CLS, SEP, etc.)
            truncation: Truncate to max_length
            max_length: Maximum sequence length
            
        Returns:
            Encoding object with ids, tokens, and attention mask

        Raises:
            RuntimeError: If encoding fails
        """
        try:
            # Configure truncation if requested using the tokenizer's API
            if truncation and max_length is not None:
                self._tokenizer.enable_truncation(max_length=max_length)
            elif not truncation and max_length is not None:
                # Disable truncation if not requested (only when max_length was previously set)
                try:
                    self._tokenizer.no_truncation()
                except Exception:
                    pass  # Ignore if already disabled
            
            encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
            
            return Encoding(encoding)
        except Exception as e:
            raise RuntimeError(f"Failed to encode text: {str(e)}") from e
    
    def encode_batch(
        self, 
        texts: List[str],
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None
    ) -> List[Encoding]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input text strings
            add_special_tokens: Add special tokens
            truncation: Truncate sequences
            max_length: Maximum sequence length
            
        Returns:
            List of Encoding objects

        Raises:
            RuntimeError: If batch encoding fails
        """
        try:
            # Configure truncation if requested using the tokenizer's API
            if truncation and max_length is not None:
                self._tokenizer.enable_truncation(max_length=max_length)
            elif not truncation and max_length is not None:
                # Disable truncation if not requested (only when max_length was previously set)
                try:
                    self._tokenizer.no_truncation()
                except Exception:
                    pass  # Ignore if already disabled
            
            encodings = self._tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
            return [Encoding(enc) for enc in encodings]
        except Exception as e:
            raise RuntimeError(f"Failed to encode batch: {str(e)}") from e
    
    def decode(
        self, 
        ids: Union[List[int], Encoding],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs or Encoding object
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text string

        Raises:
            RuntimeError: If decoding fails
        """
        try:
            if hasattr(ids, 'ids'):
                ids = ids.ids
            
            return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            raise RuntimeError(f"Failed to decode: {str(e)}") from e
    
    def decode_batch(
        self, 
        batch_ids: List[Union[List[int], Encoding]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token IDs.
        
        Args:
            batch_ids: List of token ID lists or Encoding objects
            skip_special_tokens: Skip special tokens
            
        Returns:
            List of decoded text strings

        Raises:
            RuntimeError: If batch decoding fails
        """
        try:
            ids_list = []
            for ids in batch_ids:
                if hasattr(ids, 'ids'):
                    ids_list.append(ids.ids)
                else:
                    ids_list.append(ids)
            
            return self._tokenizer.decode_batch(ids_list, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            raise RuntimeError(f"Failed to decode batch: {str(e)}") from e
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary as a dictionary."""
        return self._tokenizer.get_vocab()
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer to a file.
        
        Args:
            path: Output file path

        Raises:
            RuntimeError: If saving fails
        """
        try:
            self._tokenizer.save(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to save tokenizer: {str(e)}") from e
    
    def __call__(
        self, 
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None
    ) -> Union[Encoding, List[Encoding]]:
        """
        Tokenize text(s) - callable interface.
        
        Args:
            text: Single text string or list of strings
            add_special_tokens: Add special tokens
            truncation: Truncate sequences
            max_length: Maximum sequence length
            
        Returns:
            Encoding object or list of Encoding objects
        """
        if isinstance(text, str):
            return self.encode(text, add_special_tokens, truncation, max_length)
        else:
            return self.encode_batch(text, add_special_tokens, truncation, max_length)


class Encoding:
    """
    Wrapper for encoding results with convenient properties.
    
    Provides a consistent interface for accessing tokenization results
    including token IDs, tokens, attention masks, and offsets.
    """
    
    def __init__(self, encoding_obj: Any):
        """
        Initialize Encoding wrapper.
        
        Args:
            encoding_obj: Underlying encoding object from tokenizers library
        """
        self._encoding = encoding_obj
    
    @property
    def ids(self) -> List[int]:
        """Token IDs."""
        return self._encoding.ids
    
    @property
    def tokens(self) -> List[str]:
        """Token strings."""
        return self._encoding.tokens
    
    @property
    def attention_mask(self) -> List[int]:
        """Attention mask."""
        return self._encoding.attention_mask
    
    @property
    def type_ids(self) -> List[int]:
        """Token type IDs."""
        return self._encoding.type_ids
    
    @property
    def offsets(self) -> List[tuple]:
        """Character offsets for each token."""
        return self._encoding.offsets
    
    @property
    def special_tokens_mask(self) -> List[int]:
        """Mask indicating special tokens."""
        return self._encoding.special_tokens_mask
    
    def __len__(self) -> int:
        """Return the number of tokens."""
        return len(self.ids)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"Encoding(num_tokens={len(self.ids)}, num_special_tokens={sum(self.special_tokens_mask)})"
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another Encoding object."""
        if not isinstance(other, Encoding):
            return NotImplemented
        return self.ids == other.ids and self.tokens == other.tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert encoding to a dictionary.
        
        Returns:
            Dictionary with all encoding properties
        """
        return {
            "ids": self.ids,
            "tokens": self.tokens,
            "attention_mask": self.attention_mask,
            "type_ids": self.type_ids,
            "offsets": self.offsets,
            "special_tokens_mask": self.special_tokens_mask,
        }


class AutoTokenizer:
    """
    Factory class for automatically loading the appropriate tokenizer.
    
    Example:
        >>> from ethiobbpe import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")
    """
    
    @staticmethod
    def from_pretrained(
        model_name: str,
        **kwargs
    ) -> EthioBBPETokenizer:
        """
        Load a pretrained tokenizer by model name.
        
        Args:
            model_name: Model identifier on Hugging Face Hub
            **kwargs: Additional arguments passed to EthioBBPETokenizer
            
        Returns:
            EthioBBPETokenizer instance
        """
        return EthioBBPETokenizer.from_pretrained(model_name, **kwargs)
