#!/usr/bin/env python3
"""
Command-line interface for the Plagiarism Checker.
"""

import argparse
import sys
from pathlib import Path
from plagiarism_checker import PlagiarismChecker


def read_file(filepath: str) -> str:
    """
    Read text content from a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File content as string
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def format_similarity_score(score: float) -> str:
    """
    Format similarity score as percentage.
    
    Args:
        score: Similarity score (0 to 1)
        
    Returns:
        Formatted string
    """
    percentage = score * 100
    return f"{percentage:.2f}%"


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Plagiarism Checker - Detect plagiarism using TF-IDF and cosine similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two files
  python cli.py file1.txt file2.txt
  
  # Compare multiple files (all pairwise comparisons)
  python cli.py file1.txt file2.txt file3.txt
  
  # Compare with a specific threshold
  python cli.py file1.txt file2.txt --threshold 0.8
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='Text files to compare (at least 2 files required)'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.0,
        help='Similarity threshold (0.0 to 1.0). Only show results above this threshold. Default: 0.0'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output with detailed information'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.files) < 2:
        parser.error("At least 2 files are required for comparison")
    
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("Threshold must be between 0.0 and 1.0")
    
    # Verify all files exist
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)
    
    # Read all files
    if args.verbose:
        print(f"Reading {len(args.files)} file(s)...")
    
    texts = []
    for filepath in args.files:
        content = read_file(filepath)
        texts.append(content)
        if args.verbose:
            word_count = len(content.split())
            print(f"  {filepath}: {word_count} words")
    
    # Create checker and compare documents
    checker = PlagiarismChecker()
    
    if len(args.files) == 2:
        # Compare two files
        if args.verbose:
            print("\nComparing documents...")
        
        similarity = checker.compare_documents(texts[0], texts[1])
        
        print(f"\nSimilarity between '{args.files[0]}' and '{args.files[1]}':")
        print(f"  Score: {format_similarity_score(similarity)}")
        
        if similarity >= args.threshold:
            if similarity >= 0.8:
                print("  Status: HIGH similarity - Potential plagiarism detected!")
            elif similarity >= 0.5:
                print("  Status: MODERATE similarity - Review recommended")
            else:
                print("  Status: LOW similarity")
        else:
            print(f"  Status: Below threshold ({format_similarity_score(args.threshold)})")
    
    else:
        # Compare multiple files
        if args.verbose:
            print(f"\nComparing {len(args.files)} documents (pairwise)...")
        
        results = checker.compare_multiple_documents(texts)
        
        print(f"\nPairwise similarity results:")
        print("-" * 70)
        
        found_matches = False
        for i, j, similarity in results:
            if similarity >= args.threshold:
                found_matches = True
                status = ""
                if similarity >= 0.8:
                    status = " [HIGH - Potential plagiarism!]"
                elif similarity >= 0.5:
                    status = " [MODERATE]"
                
                print(f"{args.files[i]} <-> {args.files[j]}")
                print(f"  Similarity: {format_similarity_score(similarity)}{status}")
                print()
        
        if not found_matches:
            print(f"No document pairs found with similarity >= {format_similarity_score(args.threshold)}")
        
        # Summary statistics
        if args.verbose and results:
            similarities = [score for _, _, score in results]
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            min_similarity = min(similarities)
            
            print("-" * 70)
            print("Summary Statistics:")
            print(f"  Average similarity: {format_similarity_score(avg_similarity)}")
            print(f"  Maximum similarity: {format_similarity_score(max_similarity)}")
            print(f"  Minimum similarity: {format_similarity_score(min_similarity)}")


if __name__ == '__main__':
    main()
