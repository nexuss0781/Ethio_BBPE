import pytest
from ethiobbpe import EthioBBPETokenizer, AutoTokenizer


class TestEthioBBPETokenizer:
    """Test suite for EthioBBPE tokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        """Load pretrained tokenizer for tests."""
        return EthioBBPETokenizer.from_pretrained()
    
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
    
    def test_callable_interface(self, tokenizer):
        """Test callable interface."""
        text = "ሰላም"
        result = tokenizer(text)
        
        assert result is not None
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
