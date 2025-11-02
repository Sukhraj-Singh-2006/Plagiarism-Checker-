#!/usr/bin/env python3
"""
Command-line interface for the Plagiarism Checker.
"""

import argparse
import sys
from plagiarism_checker import PlagiarismChecker, read_file


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Plagiarism Checker - Compare text documents using TF-IDF and cosine similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two files
  python cli.py file1.txt file2.txt
  
  # Compare multiple files (all pairs)
  python cli.py doc1.txt doc2.txt doc3.txt
  
  # Compare with threshold
  python cli.py file1.txt file2.txt --threshold 0.7
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='Text files to compare for plagiarism'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.0,
        help='Similarity threshold (0.0 to 1.0). Only show results above this threshold (default: 0.0)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    parser.add_argument(
        '--no-emoji',
        action='store_true',
        help='Disable emoji indicators for better accessibility'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Validate number of files
    if len(args.files) < 2:
        print("Error: At least two files are required for comparison", file=sys.stderr)
        sys.exit(1)
    
    # Read files
    checker = PlagiarismChecker()
    
    if args.verbose:
        print(f"Reading {len(args.files)} files...")
    
    for filepath in args.files:
        try:
            content = read_file(filepath)
            checker.add_document(content, filepath)
            if args.verbose:
                print(f"  - {filepath}: {len(content)} characters")
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file {filepath}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Perform comparisons
    if args.verbose:
        print("\nComparing documents...")
    
    results = checker.check_all_pairs()
    
    # Filter by threshold and sort by similarity (descending)
    filtered_results = [r for r in results if r[2] >= args.threshold]
    filtered_results.sort(key=lambda x: x[2], reverse=True)
    
    # Display results
    print("\nPlagiarism Detection Results")
    print("=" * 70)
    
    if not filtered_results:
        print(f"No similarities found above threshold {args.threshold:.2f}")
    else:
        for doc1, doc2, similarity in filtered_results:
            percentage = similarity * 100
            
            # Visual indicator
            if similarity >= 0.9:
                indicator = "[HIGH]" if args.no_emoji else "🔴 HIGH"
            elif similarity >= 0.7:
                indicator = "[MEDIUM]" if args.no_emoji else "🟡 MEDIUM"
            elif similarity >= 0.5:
                indicator = "[LOW]" if args.no_emoji else "🟢 LOW"
            else:
                indicator = "[MINIMAL]" if args.no_emoji else "⚪ MINIMAL"
            
            print(f"\n{indicator}")
            print(f"Document 1: {doc1}")
            print(f"Document 2: {doc2}")
            print(f"Similarity: {percentage:.2f}%")
            print("-" * 70)
    
    # Summary
    print(f"\nTotal comparisons: {len(results)}")
    print(f"Above threshold ({args.threshold:.2f}): {len(filtered_results)}")


if __name__ == '__main__':
    main()
