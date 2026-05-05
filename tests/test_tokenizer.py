import pytest
import json
import gzip
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from ethiobbpe import EthioBBPETokenizer, AutoTokenizer, Encoding


class TestEthioBBPETokenizer:
    """Test suite for EthioBBPE tokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        """Load pretrained tokenizer for tests."""
        return EthioBBPETokenizer.from_pretrained()
    
    @pytest.fixture
    def sample_tokenizer_json(self):
        """Create a sample tokenizer.json file for testing."""
        tokenizer_config = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": 0, "content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 1, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 2, "content": "[CLS]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 3, "content": "[SEP]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 4, "content": "[MASK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            ],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": {"type": "BPEDecoder", "suffix": "</w>", "fuse": False},
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": False,
                "vocab": {"ሰ": 5, "ላ": 6, "ም": 7, " ": 8, "ለ": 9, "ኢ": 10, "ዮ": 11, "ብ": 12},
                "merges": []
            }
        }
        return tokenizer_config
    
    def test_initialization(self, tokenizer):
        """Test tokenizer initialization."""
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0
        assert tokenizer.model_name == "Nexuss0781/Ethio-BBPE"
    
    def test_encode_single_text(self, tokenizer):
        """Test encoding a single Amharic text."""
        text = "ሰላም ለኢዮብ"
        encoded = tokenizer.encode(text)
        
        assert encoded is not None
        assert len(encoded.ids) > 0
        assert len(encoded.tokens) > 0
        assert len(encoded.ids) == len(encoded.tokens)
    
    def test_decode_single_text(self, tokenizer):
        """Test decoding token IDs back to text."""
        text = "ሰላም ዓለም"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        
        assert decoded == text
    
    def test_round_trip_perfect_reconstruction(self, tokenizer):
        """Test perfect reconstruction of biblical texts."""
        test_cases = [
            "ሰላም ለኢዮብ ዘኢነበበ ከንቶ ።",
            "በመዠመሪያ፡እግዚአብሔር፡ሰማይንና፡ምድርን፡ፈጠረ።",
            "፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠",
        ]
        
        for text in test_cases:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            assert decoded == text, f"Failed for: {text}"
    
    def test_batch_encoding(self, tokenizer):
        """Test batch encoding of multiple texts."""
        texts = [
            "ሰላም ለኢዮብ",
            "ወደ ቍስጥንጥንያ አገርም",
            "ሐዋርያ መንፈስ ይቤ"
        ]
        
        encodings = tokenizer.encode_batch(texts)
        
        assert len(encodings) == len(texts)
        for enc in encodings:
            assert len(enc.ids) > 0
            assert len(enc.tokens) > 0
    
    def test_batch_decoding(self, tokenizer):
        """Test batch decoding of token IDs."""
        texts = ["ሰላም ዓለም", "እግዚአብሔር", "ቍስጥንጥንያ"]
        encodings = [tokenizer.encode(t) for t in texts]
        batch_ids = [enc.ids for enc in encodings]
        
        decoded_texts = tokenizer.decode_batch(batch_ids)
        
        assert len(decoded_texts) == len(texts)
        for original, decoded in zip(texts, decoded_texts):
            assert decoded == original
    
    def test_callable_interface_single_text(self, tokenizer):
        """Test callable interface with single text."""
        text = "ሰላም"
        result = tokenizer(text)
        
        assert result is not None
        assert len(result.ids) > 0
    
    def test_callable_interface_batch_texts(self, tokenizer):
        """Test callable interface with batch of texts."""
        texts = ["ሰላም", "ዓለም", "እንዴት"]
        results = tokenizer(texts)
        
        assert isinstance(results, list)
        assert len(results) == len(texts)
        for result in results:
            assert len(result.ids) > 0
    
    def test_special_tokens(self, tokenizer):
        """Test special tokens handling."""
        text = "ሰላም ዓለም"
        encoded_with_special = tokenizer.encode(text, add_special_tokens=True)
        encoded_without_special = tokenizer.encode(text, add_special_tokens=False)
        
        assert len(encoded_with_special.ids) >= len(encoded_without_special.ids)
    
    def test_truncation(self, tokenizer):
        """Test truncation functionality."""
        text = "ሰላም ዓለም እንዴት ነህ ዛሬ ምን አዲስ ነገር አለ"
        encoded = tokenizer.encode(text, truncation=True, max_length=5)
        
        assert len(encoded.ids) <= 5
    
    def test_attention_mask(self, tokenizer):
        """Test attention mask generation."""
        text = "ሰላም ዓለም"
        encoded = tokenizer.encode(text)
        
        assert len(encoded.attention_mask) == len(encoded.ids)
        assert all(mask == 1 for mask in encoded.attention_mask)
    
    def test_vocabulary_access(self, tokenizer):
        """Test vocabulary access methods."""
        vocab = tokenizer.get_vocab()
        
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
        assert tokenizer.get_vocab_size() > 0
    
    def test_geez_punctuation(self, tokenizer):
        """Test Ge'ez punctuation handling."""
        geez_marks = "፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠፠"
        encoded = tokenizer.encode(geez_marks)
        decoded = tokenizer.decode(encoded.ids)
        
        assert decoded == geez_marks
    
    def test_auto_tokenizer_factory(self):
        """Test AutoTokenizer factory class."""
        tokenizer = AutoTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")
        
        assert tokenizer is not None
        assert isinstance(tokenizer, EthioBBPETokenizer)
    
    def test_encoding_properties(self, tokenizer):
        """Test Encoding object properties."""
        text = "ሰላም ዓለም"
        encoded = tokenizer.encode(text)
        
        # Test all properties exist and have correct types
        assert isinstance(encoded.ids, list)
        assert isinstance(encoded.tokens, list)
        assert isinstance(encoded.attention_mask, list)
        assert isinstance(encoded.type_ids, list)
        assert isinstance(encoded.offsets, list)
        assert isinstance(encoded.special_tokens_mask, list)
        
        # Test length property
        assert len(encoded) == len(encoded.ids)
        
        # Test string representation
        assert "Encoding" in repr(encoded)


class TestTokenizerFromFile:
    """Test suite for loading tokenizer from file."""
    
    @pytest.fixture
    def sample_tokenizer_json(self):
        """Create a sample tokenizer.json file for testing."""
        return {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": 0, "content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 1, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 2, "content": "[CLS]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 3, "content": "[SEP]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 4, "content": "[MASK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            ],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": {"type": "BPEDecoder", "suffix": "</w>", "fuse": False},
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": False,
                "vocab": {"ሰ": 5, "ላ": 6, "ም": 7, " ": 8, "ለ": 9, "ኢ": 10, "ዮ": 11, "ብ": 12},
                "merges": []
            }
        }
    
    def test_from_file_json(self, sample_tokenizer_json):
        """Test loading tokenizer from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_tokenizer_json, f)
            temp_path = f.name
        
        try:
            tokenizer = EthioBBPETokenizer.from_file(temp_path)
            assert tokenizer is not None
            # vocab_size comes from config which defaults to DEFAULT_VOCAB_SIZE when not provided in file
            assert tokenizer.vocab_size > 0
        finally:
            Path(temp_path).unlink()
    
    def test_from_file_gzipped_vocab(self, sample_tokenizer_json):
        """Test loading tokenizer from gzipped vocab file."""
        vocab = sample_tokenizer_json["model"]["vocab"]
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.json.gz', delete=False) as f:
            with gzip.open(f, 'wt', encoding='utf-8') as gz_f:
                json.dump(vocab, gz_f)
            temp_path = f.name
        
        try:
            tokenizer = EthioBBPETokenizer.from_file(temp_path)
            assert tokenizer is not None
            assert tokenizer.vocab_size == len(vocab)
        finally:
            Path(temp_path).unlink()
    
    def test_from_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            EthioBBPETokenizer.from_file("/nonexistent/path/tokenizer.json")
    
    def test_from_file_unsupported_format(self):
        """Test ValueError for unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                EthioBBPETokenizer.from_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestTokenizerErrors:
    """Test suite for error handling in tokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        """Load pretrained tokenizer for tests."""
        return EthioBBPETokenizer.from_pretrained()
    
    def test_from_pretrained_import_error(self):
        """Test ImportError when tokenizers library is not available."""
        # This test verifies that proper error is raised when tokenizers is not installed
        # Since we can't easily mock the module-level HAS_TOKENIZERS after import,
        # we verify the error message contains the expected text
        with patch('ethiobbpe.tokenizer.hf_hub_download', return_value="/fake/path.json"):
            with patch('ethiobbpe.tokenizer.HFTokenizer') as mock_hf:
                # Simulate the case where tokenizers is not available by raising ImportError in from_file
                mock_hf.from_file.side_effect = ImportError("tokenizers library required")
                
                with pytest.raises(RuntimeError, match="Failed to load tokenizer"):
                    EthioBBPETokenizer.from_pretrained()
    
    def test_from_pretrained_download_failure(self):
        """Test RuntimeError when download fails."""
        with patch('ethiobbpe.tokenizer.hf_hub_download', side_effect=Exception("Download failed")):
            with pytest.raises(RuntimeError, match="Failed to load tokenizer"):
                EthioBBPETokenizer.from_pretrained()
    
    def test_from_pretrained_config_load_failure(self):
        """Test that tokenizer loads even if config fails to load."""
        with patch('ethiobbpe.tokenizer.hf_hub_download') as mock_download:
            # First call succeeds (tokenizer.json), second fails (config.json)
            mock_download.side_effect = ["/fake/tokenizer.json", Exception("Config not found")]
            
            with patch('ethiobbpe.tokenizer.HFTokenizer') as mock_hf:
                mock_tokenizer_obj = MagicMock()
                mock_hf.from_file.return_value = mock_tokenizer_obj
                
                tokenizer = EthioBBPETokenizer.from_pretrained()
                assert tokenizer is not None
                assert tokenizer.model_name == "Nexuss0781/Ethio-BBPE"
    
    def test_encode_exception_handling(self, tokenizer):
        """Test exception handling in encode method."""
        with patch.object(tokenizer._tokenizer, 'encode', side_effect=Exception("Encode failed")):
            with pytest.raises(RuntimeError, match="Failed to encode text"):
                tokenizer.encode("test")
    
    def test_encode_batch_exception_handling(self, tokenizer):
        """Test exception handling in encode_batch method."""
        with patch.object(tokenizer._tokenizer, 'encode_batch', side_effect=Exception("Batch encode failed")):
            with pytest.raises(RuntimeError, match="Failed to encode batch"):
                tokenizer.encode_batch(["test1", "test2"])
    
    def test_decode_exception_handling(self, tokenizer):
        """Test exception handling in decode method."""
        with patch.object(tokenizer._tokenizer, 'decode', side_effect=Exception("Decode failed")):
            with pytest.raises(RuntimeError, match="Failed to decode"):
                tokenizer.decode([1, 2, 3])
    
    def test_decode_batch_exception_handling(self, tokenizer):
        """Test exception handling in decode_batch method."""
        with patch.object(tokenizer._tokenizer, 'decode_batch', side_effect=Exception("Batch decode failed")):
            with pytest.raises(RuntimeError, match="Failed to decode batch"):
                tokenizer.decode_batch([[1, 2], [3, 4]])
    
    def test_save_exception_handling(self, tokenizer):
        """Test exception handling in save method."""
        with patch.object(tokenizer._tokenizer, 'save', side_effect=Exception("Save failed")):
            with pytest.raises(RuntimeError, match="Failed to save tokenizer"):
                tokenizer.save("/nonexistent/path/tokenizer.json")
    
    def test_save_to_valid_path(self, tokenizer):
        """Test saving tokenizer to valid path."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            tokenizer.save(temp_path)
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink()


class TestEncodingClass:
    """Test suite for Encoding wrapper class."""
    
    @pytest.fixture
    def tokenizer(self):
        """Load pretrained tokenizer for tests."""
        return EthioBBPETokenizer.from_pretrained()
    
    def test_encoding_eq_with_non_encoding(self, tokenizer):
        """Test Encoding equality comparison with non-Encoding object."""
        encoded = tokenizer.encode("ሰላም")
        
        # Should return NotImplemented (which becomes False in practice)
        result = encoded.__eq__("not an encoding")
        assert result is NotImplemented
    
    def test_encoding_to_dict(self, tokenizer):
        """Test converting Encoding to dictionary."""
        text = "ሰላም ዓለም"
        encoded = tokenizer.encode(text)
        
        result_dict = encoded.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "ids" in result_dict
        assert "tokens" in result_dict
        assert "attention_mask" in result_dict
        assert "type_ids" in result_dict
        assert "offsets" in result_dict
        assert "special_tokens_mask" in result_dict
        
        assert result_dict["ids"] == encoded.ids
        assert result_dict["tokens"] == encoded.tokens
    
    def test_encoding_repr(self, tokenizer):
        """Test Encoding string representation."""
        text = "ሰላም ዓለም"
        encoded = tokenizer.encode(text)
        
        repr_str = repr(encoded)
        assert "Encoding" in repr_str
        assert "num_tokens" in repr_str
    
    def test_encoding_len(self, tokenizer):
        """Test Encoding length."""
        text = "ሰላም ዓለም"
        encoded = tokenizer.encode(text)
        
        assert len(encoded) == len(encoded.ids)
    
    def test_encoding_equality_same(self, tokenizer):
        """Test Encoding equality with same content."""
        text = "ሰላም"
        encoded1 = tokenizer.encode(text)
        encoded2 = tokenizer.encode(text)
        
        assert encoded1 == encoded2
    
    def test_encoding_inequality_different(self, tokenizer):
        """Test Encoding inequality with different content."""
        encoded1 = tokenizer.encode("ሰላም")
        encoded2 = tokenizer.encode("ዓለም")
        
        assert encoded1 != encoded2


class TestTruncationModes:
    """Test suite for truncation configuration."""
    
    @pytest.fixture
    def tokenizer(self):
        """Load pretrained tokenizer for tests."""
        return EthioBBPETokenizer.from_pretrained()

    def test_truncation_enable_disable(self, tokenizer):
        """Test enabling and disabling truncation."""
        text = "ሰላም ዓለም እንዴት ነህ"
        
        # Enable truncation
        encoded_truncated = tokenizer.encode(text, truncation=True, max_length=3)
        assert len(encoded_truncated.ids) <= 3
        
        # Disable truncation - encode without max_length to get full encoding
        encoded_no_trunc = tokenizer.encode(text, truncation=False, max_length=None)
        # The full encoding should have more tokens than truncated
        assert len(encoded_no_trunc.ids) >= len(encoded_truncated.ids)
    
    def test_encode_batch_truncation(self, tokenizer):
        """Test batch encoding with truncation."""
        texts = ["ሰላም ዓለም እንዴት ነህ", "የተሻለ ቀን"]
        
        encodings = tokenizer.encode_batch(texts, truncation=True, max_length=3)
        
        assert len(encodings) == 2
        for enc in encodings:
            assert len(enc.ids) <= 3


class TestDecodeWithEncoding:
    """Test decoding with Encoding objects."""
    
    @pytest.fixture
    def tokenizer(self):
        """Load pretrained tokenizer for tests."""
        return EthioBBPETokenizer.from_pretrained()

    def test_decode_from_encoding_object(self, tokenizer):
        """Test decoding directly from Encoding object."""
        text = "ሰላም ዓለም"
        encoded = tokenizer.encode(text)
        
        decoded = tokenizer.decode(encoded)
        assert decoded == text
    
    def test_decode_batch_from_encoding_objects(self, tokenizer):
        """Test batch decoding from Encoding objects."""
        texts = ["ሰላም", "ዓለም"]
        encodings = [tokenizer.encode(t) for t in texts]
        
        decoded_texts = tokenizer.decode_batch(encodings)
        
        assert len(decoded_texts) == len(texts)
        for original, decoded in zip(texts, decoded_texts):
            assert decoded == original


class TestTruncationDisableBranches:
    """Test suite for covering truncation disable branches."""
    
    @pytest.fixture
    def tokenizer(self):
        """Load pretrained tokenizer for tests."""
        return EthioBBPETokenizer.from_pretrained()

    def test_encode_disable_truncation_branch(self, tokenizer):
        """Test encode method with truncation=False to cover elif branch."""
        text = "ሰላም ዓለም እንዴት ነህ"
        
        # First enable truncation with max_length
        _ = tokenizer.encode(text, truncation=True, max_length=3)
        # Then disable truncation with max_length to trigger the elif branch
        encoded = tokenizer.encode(text, truncation=False, max_length=10)
        
        assert encoded is not None
        assert len(encoded.ids) > 0
    
    def test_encode_batch_disable_truncation_branch(self, tokenizer):
        """Test encode_batch method with truncation=False to cover elif branch."""
        texts = ["ሰላም ዓለም", "እንዴት ነህ"]
        
        # First enable truncation with max_length
        _ = tokenizer.encode_batch(texts, truncation=True, max_length=3)
        # Then disable truncation with max_length to trigger the elif branch
        encodings = tokenizer.encode_batch(texts, truncation=False, max_length=10)
        
        assert len(encodings) == 2
        for enc in encodings:
            assert len(enc.ids) > 0
    
    def test_encode_no_truncation_exception_handling(self, tokenizer):
        """Test encode handles exception when no_truncation() fails."""
        text = "ሰላም ዓለም"
        
        # First enable truncation to set up state
        _ = tokenizer.encode(text, truncation=True, max_length=5)
        
        # Mock no_truncation to raise an exception
        with patch.object(tokenizer._tokenizer, 'no_truncation', side_effect=Exception("Already disabled")):
            # Should not raise, just ignore the exception
            encoded = tokenizer.encode(text, truncation=False, max_length=10)
            assert encoded is not None
            assert len(encoded.ids) > 0
    
    def test_encode_batch_no_truncation_exception_handling(self, tokenizer):
        """Test encode_batch handles exception when no_truncation() fails."""
        texts = ["ሰላም", "ዓለም"]
        
        # First enable truncation to set up state
        _ = tokenizer.encode_batch(texts, truncation=True, max_length=3)
        
        # Mock no_truncation to raise an exception
        with patch.object(tokenizer._tokenizer, 'no_truncation', side_effect=Exception("Already disabled")):
            # Should not raise, just ignore the exception
            encodings = tokenizer.encode_batch(texts, truncation=False, max_length=10)
            assert len(encodings) == 2
            for enc in encodings:
                assert len(enc.ids) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
