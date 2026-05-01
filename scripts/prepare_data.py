#!/usr/bin/env python3
"""
Data Preparation Script for EthioBBPE
Extracts text from parquet datasets and prepares training corpus
"""

import os
import glob
import pandas as pd
from pathlib import Path


def extract_text_from_parquet(parquet_path, text_columns=None):
    """
    Extract text from specified columns in a parquet file.
    
    Args:
        parquet_path: Path to parquet file
        text_columns: List of column names to extract. If None, extracts all string columns.
    
    Returns:
        List of text strings
    """
    df = pd.read_parquet(parquet_path)
    
    if text_columns is None:
        # Auto-detect string columns
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    texts = []
    for col in text_columns:
        if col in df.columns:
            texts.extend(df[col].dropna().astype(str).tolist())
    
    return texts


def prepare_training_corpus(
    data_dir="./data",
    output_file="./data/training_corpus.txt",
    min_length=10,
    max_length=5000
):
    """
    Prepare training corpus from all parquet files in data directory.
    
    Args:
        data_dir: Directory containing parquet files
        output_file: Output file path for combined corpus
        min_length: Minimum text length to include
        max_length: Maximum text length to include
    
    Returns:
        dict with statistics
    """
    data_path = Path(data_dir)
    parquet_files = list(data_path.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet file(s)")
    
    all_texts = []
    stats = {"total_files": 0, "total_texts": 0, "filtered_texts": 0}
    
    for pq_file in parquet_files:
        print(f"\nProcessing: {pq_file.name}")
        
        # Determine which columns to extract based on filename
        if "synaxarium" in pq_file.name.lower():
            text_cols = ["መጽሃፍ"]  # Book/content column
        elif "canon" in pq_file.name.lower() or "biblical" in pq_file.name.lower():
            text_cols = ["ጥቅስ", "verse"]  # Verse columns
        else:
            text_cols = None  # Auto-detect
        
        try:
            texts = extract_text_from_parquet(str(pq_file), text_cols)
            stats["total_files"] += 1
            stats["total_texts"] += len(texts)
            
            # Filter by length
            filtered = [
                t for t in texts 
                if min_length <= len(t) <= max_length
            ]
            stats["filtered_texts"] += len(filtered)
            all_texts.extend(filtered)
            
            print(f"  - Extracted: {len(texts)} texts")
            print(f"  - After filtering: {len(filtered)} texts")
            
        except Exception as e:
            print(f"  ⚠️ Error processing {pq_file.name}: {e}")
    
    if not all_texts:
        raise ValueError("No valid texts extracted from datasets")
    
    # Write to output file
    print(f"\n✍️ Writing {len(all_texts)} texts to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text.strip() + "\n")
    
    # Calculate statistics
    total_chars = sum(len(t) for t in all_texts)
    avg_length = total_chars / len(all_texts) if all_texts else 0
    
    stats.update({
        "output_file": output_file,
        "total_characters": total_chars,
        "average_length": round(avg_length, 2),
        "unique_texts": len(set(all_texts))
    })
    
    print("\n" + "="*60)
    print("📊 PREPARATION STATISTICS")
    print("="*60)
    print(f"Files processed:     {stats['total_files']}")
    print(f"Total texts:         {stats['total_texts']}")
    print(f"After filtering:     {stats['filtered_texts']}")
    print(f"Unique texts:        {stats['unique_texts']}")
    print(f"Total characters:    {stats['total_characters']:,}")
    print(f"Average length:      {stats['average_length']} chars")
    print(f"Output file:         {stats['output_file']}")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training corpus from parquet files")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory with parquet files")
    parser.add_argument("--output", type=str, default="./data/training_corpus.txt", help="Output corpus file")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum text length")
    parser.add_argument("--max_length", type=int, default=5000, help="Maximum text length")
    
    args = parser.parse_args()
    
    stats = prepare_training_corpus(
        data_dir=args.data_dir,
        output_file=args.output,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    print("\n✅ Corpus preparation complete!")
    print(f"Ready to train tokenizer with: {args.output}")
