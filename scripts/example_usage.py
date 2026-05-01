"""
EthioBBPE Example Usage

Demonstrates how to use the trained EthioBBPE tokenizer for Ethiopian languages.
"""

from tokenizers import Tokenizer
import os

def load_tokenizer(model_path="models/EthioBBPE/tokenizer.json"):
    """Load the trained EthioBBPE tokenizer."""
    if not os.path.exists(model_path):
        # Try demo model
        demo_path = "models/demo_tokenizer/tokenizer.json"
        if os.path.exists(demo_path):
            print(f"⚠️  EthioBBPE model not found, using demo model instead.")
            model_path = demo_path
        else:
            raise FileNotFoundError(
                f"No tokenizer found at {model_path}. Please train a model first."
            )
    
    return Tokenizer.from_file(model_path)


def main():
    print("🇪🇹 EthioBBPE Tokenizer - Example Usage\n")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    print(f"✅ Loaded tokenizer from: {tokenizer.model_filename}\n")
    
    # Test texts in multiple Ethiopian languages
    test_texts = [
        ("Amharic", "ሰላም! እንዴት ነህ? የኢትዮጵያ ህዝብ በጣም ተቀራራቢ ነው።"),
        ("Oromo", "Akkam! Akkam jirta? Ummanni Itoophiyaa baay'ee wal-qabaataa dha."),
        ("Tigrinya", "ሰላም! ከመይ ኣለኻ? ህዝቢ ኢትዮጵያ ኣዝዩ ሓደ እዩ።"),
        ("English", "Hello! How are you? The people of Ethiopia are very united."),
        ("Mixed", "ሰላም Hello! እንዴት ነህ? How are you? 🇪🇹"),
    ]
    
    for lang_name, text in test_texts:
        print(f"\n--- {lang_name} ---")
        print(f"Original: {text}")
        
        # Encode
        encoded = tokenizer.encode(text)
        print(f"Tokens ({len(encoded.tokens)}): {encoded.tokens[:20]}{'...' if len(encoded.tokens) > 20 else ''}")
        print(f"IDs ({len(encoded.ids)}): {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''}")
        
        # Decode
        decoded = tokenizer.decode(encoded.ids)
        print(f"Decoded: {decoded}")
        
        # Verify round-trip
        match = "✅" if decoded == text else "⚠️"
        print(f"Round-trip: {match} {'Perfect match!' if decoded == text else 'Minor differences'}")
    
    print("\n" + "=" * 50)
    print("✨ Example usage complete!")
    print("\nTo train your own EthioBBPE tokenizer:")
    print("  python scripts/train_tokenizer.py --data_dir ./data --model_name EthioBBPE")


if __name__ == "__main__":
    main()
