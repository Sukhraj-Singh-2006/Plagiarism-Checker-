# Plagiarism Checker

A command-line tool and Python library to detect plagiarism by comparing text documents using **TF-IDF** (Term Frequency-Inverse Document Frequency) and **Cosine Similarity**.

## Features

- 📊 **TF-IDF Analysis**: Uses Term Frequency-Inverse Document Frequency for intelligent text comparison
- 📐 **Cosine Similarity**: Measures document similarity with mathematical precision
- 🖥️ **Command-Line Interface**: Easy-to-use CLI for quick comparisons
- 📚 **Python Library**: Import and use in your own projects
- 🎯 **Threshold Filtering**: Filter results by similarity threshold
- 📁 **Multiple File Support**: Compare multiple documents at once
- 🔍 **Visual Indicators**: Color-coded similarity levels (HIGH/MEDIUM/LOW/MINIMAL)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/Sukhraj-Singh-2006/Plagiarism-Checker-.git
cd Plagiarism-Checker-

# Install the package
pip install -e .
```

### Requirements

- Python 3.7 or higher
- No external dependencies (uses only Python standard library)

## Usage

### Command-Line Interface

#### Compare Two Files

```bash
python cli.py examples/doc1.txt examples/doc2.txt
```

#### Compare Multiple Files

```bash
python cli.py examples/doc1.txt examples/doc2.txt examples/doc3.txt
```

#### Filter by Threshold

```bash
# Only show similarities above 70%
python cli.py examples/doc1.txt examples/doc2.txt --threshold 0.7
```

#### Verbose Output

```bash
python cli.py examples/doc1.txt examples/doc2.txt --verbose
```

#### After Installation

If you installed the package, you can use the `plagiarism-checker` command:

```bash
plagiarism-checker file1.txt file2.txt
plagiarism-checker file1.txt file2.txt file3.txt --threshold 0.5
```

### Python Library

You can also use the plagiarism checker as a library in your Python code:

```python
from plagiarism_checker import PlagiarismChecker

# Create a checker instance
checker = PlagiarismChecker()

# Compare two texts directly
text1 = "Python is a programming language."
text2 = "Python is a popular programming language."
similarity = checker.compare_documents(text1, text2)
print(f"Similarity: {similarity * 100:.2f}%")

# Or add multiple documents and compare all pairs
checker.add_document(text1, "Document 1")
checker.add_document(text2, "Document 2")
checker.add_document("JavaScript is a scripting language.", "Document 3")

results = checker.check_all_pairs()
for doc1, doc2, score in results:
    print(f"{doc1} vs {doc2}: {score * 100:.2f}%")
```

#### Reading from Files

```python
from plagiarism_checker import PlagiarismChecker, read_file

checker = PlagiarismChecker()

# Read files and compare
text1 = read_file("file1.txt")
text2 = read_file("file2.txt")
similarity = checker.compare_documents(text1, text2)
print(f"Similarity: {similarity * 100:.2f}%")
```

## How It Works

### 1. Text Preprocessing
- Converts text to lowercase
- Extracts words using regular expressions
- Removes punctuation and special characters

### 2. TF-IDF Calculation
- **Term Frequency (TF)**: Measures how frequently a word appears in a document
  ```
  TF(word) = (Number of times word appears in document) / (Total words in document)
  ```

- **Inverse Document Frequency (IDF)**: Measures how important a word is across all documents
  ```
  IDF(word) = log(Total number of documents / Number of documents containing word)
  ```

- **TF-IDF**: Combines both metrics
  ```
  TF-IDF(word) = TF(word) × IDF(word)
  ```

### 3. Cosine Similarity
Calculates the cosine of the angle between two TF-IDF vectors:

```
Similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A · B is the dot product of vectors A and B
- ||A|| and ||B|| are the magnitudes of the vectors

Result ranges from 0 (completely different) to 1 (identical).

## Similarity Interpretation

| Score Range | Indicator | Interpretation |
|------------|-----------|----------------|
| 0.9 - 1.0  | 🔴 HIGH    | Very high similarity - potential plagiarism |
| 0.7 - 0.9  | 🟡 MEDIUM  | Moderate similarity - needs review |
| 0.5 - 0.7  | 🟢 LOW     | Some similarity - likely coincidental |
| 0.0 - 0.5  | ⚪ MINIMAL | Low similarity - documents are different |

## Examples

The `examples/` directory contains sample documents for testing:

- `doc1.txt`: Text about Python programming language
- `doc2.txt`: Similar text about Python (high similarity expected)
- `doc3.txt`: Text about JavaScript (low similarity expected)

Try running:

```bash
python cli.py examples/doc1.txt examples/doc2.txt examples/doc3.txt
```

## Running Tests

Run the unit tests to verify functionality:

```bash
python test_plagiarism_checker.py
```

Or with verbose output:

```bash
python test_plagiarism_checker.py -v
```

## API Reference

### PlagiarismChecker Class

#### Methods

- `preprocess_text(text: str) -> List[str]`
  - Preprocesses text and returns list of words

- `calculate_tf(words: List[str]) -> Dict[str, float]`
  - Calculates Term Frequency

- `calculate_idf(documents_words: List[List[str]]) -> Dict[str, float]`
  - Calculates Inverse Document Frequency

- `calculate_tfidf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]`
  - Calculates TF-IDF scores

- `cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float`
  - Calculates cosine similarity between two vectors

- `compare_documents(text1: str, text2: str) -> float`
  - Compares two documents and returns similarity score (0-1)

- `add_document(text: str, name: str = None)`
  - Adds a document to the checker

- `check_all_pairs() -> List[Tuple[str, str, float]]`
  - Compares all added documents and returns results

- `clear_documents()`
  - Clears all added documents

### Helper Functions

- `read_file(filepath: str) -> str`
  - Reads and returns content from a file

## CLI Options

```
usage: cli.py [-h] [-t THRESHOLD] [-v] files [files ...]

positional arguments:
  files                 Text files to compare for plagiarism

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        Similarity threshold (0.0 to 1.0). Only show results
                        above this threshold (default: 0.0)
  -v, --verbose         Show detailed output
```

## Limitations

- Works best with documents that have substantial text content
- Case-insensitive comparison
- Does not account for word order or sentence structure
- Based on word frequency, not semantic meaning
- May produce high similarity scores for documents with the same topic but no actual plagiarism

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Sukhraj Singh
