# Python Plagiarism Checker

A command-line tool and Python library to detect plagiarism by comparing text documents using **TF-IDF** (Term Frequency-Inverse Document Frequency) and **cosine similarity**.

## Features

- 🔍 **Accurate Detection**: Uses TF-IDF and cosine similarity for reliable plagiarism detection
- 📊 **Multiple Comparison Modes**: Compare two documents or multiple documents (pairwise)
- 🛠️ **Dual Interface**: Works as both a command-line tool and a Python library
- 📈 **Similarity Scoring**: Returns similarity scores from 0% (completely different) to 100% (identical)
- 🎯 **Threshold Filtering**: Set custom thresholds to filter results
- 📝 **Detailed Reports**: Verbose mode for detailed analysis

## Installation

No external dependencies required! This tool uses only Python standard library.

```bash
# Clone the repository
git clone https://github.com/Sukhraj-Singh-2006/Plagiarism-Checker-.git
cd Plagiarism-Checker-

# Optional: Install development dependencies
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

#### Compare Two Files
```bash
python cli.py file1.txt file2.txt
```

Output:
```
Similarity between 'file1.txt' and 'file2.txt':
  Score: 56.02%
  Status: MODERATE similarity - Review recommended
```

#### Compare Multiple Files
```bash
python cli.py file1.txt file2.txt file3.txt
```

#### Verbose Mode
```bash
python cli.py file1.txt file2.txt -v
```

#### Set Similarity Threshold
```bash
python cli.py file1.txt file2.txt --threshold 0.8
```
This will only show results with similarity >= 80%

#### Help
```bash
python cli.py --help
```

### Python Library

#### Basic Usage

```python
from plagiarism_checker import check_plagiarism

text1 = "Machine learning is a subset of artificial intelligence."
text2 = "Machine learning represents a branch of artificial intelligence."

similarity = check_plagiarism(text1, text2)
print(f"Similarity: {similarity * 100:.2f}%")
```

#### Advanced Usage

```python
from plagiarism_checker import PlagiarismChecker

# Create checker instance
checker = PlagiarismChecker()

# Compare two documents
doc1 = "Python is a programming language."
doc2 = "Python is a programming language famous for simplicity."

similarity = checker.compare_documents(doc1, doc2)
print(f"Similarity: {similarity * 100:.2f}%")

# Compare multiple documents
documents = [
    "First document text here.",
    "Second document text here.",
    "Third document text here."
]

results = checker.compare_multiple_documents(documents)
for i, j, score in results:
    print(f"Doc{i} vs Doc{j}: {score * 100:.2f}%")
```

## How It Works

### TF-IDF (Term Frequency-Inverse Document Frequency)

1. **Term Frequency (TF)**: Measures how frequently a term appears in a document
   - `TF = (Number of times term appears) / (Total terms in document)`

2. **Inverse Document Frequency (IDF)**: Measures how important a term is across all documents
   - `IDF = log((1 + Total documents) / (1 + Documents containing term)) + 1`

3. **TF-IDF**: Combines both metrics
   - `TF-IDF = TF × IDF`

### Cosine Similarity

Measures the cosine of the angle between two TF-IDF vectors:
- **1.0 (100%)**: Identical documents
- **0.0 (0%)**: Completely different documents
- **0.5-0.8 (50-80%)**: Moderate to high similarity (potential plagiarism)

## Similarity Interpretation

| Score Range | Classification | Description |
|------------|----------------|-------------|
| 80-100% | HIGH | Strong indication of plagiarism |
| 50-79% | MODERATE | Review recommended |
| 0-49% | LOW | Minimal similarity |

## Testing

Run the test suite:

```bash
python -m unittest test_plagiarism_checker.py -v
```

## Examples

Sample text files are included:
- `sample1.txt`: Machine learning overview
- `sample2.txt`: Similar to sample1 (paraphrased)
- `sample3.txt`: Python programming overview

Try them:
```bash
python cli.py sample1.txt sample2.txt sample3.txt -v
```

## API Reference

### `PlagiarismChecker` Class

#### Methods

- `preprocess_text(text: str) -> List[str]`
  - Preprocesses text by converting to lowercase and extracting words

- `calculate_term_frequency(words: List[str]) -> Dict[str, float]`
  - Calculates term frequency for each word

- `calculate_inverse_document_frequency(documents: List[List[str]]) -> Dict[str, float]`
  - Calculates IDF values across documents

- `calculate_tf_idf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]`
  - Calculates TF-IDF scores

- `cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float`
  - Calculates cosine similarity between two vectors

- `compare_documents(text1: str, text2: str) -> float`
  - Compares two documents and returns similarity score (0 to 1)

- `compare_multiple_documents(texts: List[str]) -> List[Tuple[int, int, float]]`
  - Compares multiple documents pairwise
  - Returns list of tuples: (doc1_index, doc2_index, similarity_score)

### Convenience Function

- `check_plagiarism(text1: str, text2: str) -> float`
  - Quick way to compare two texts
  - Returns similarity score (0 to 1)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Sukhraj Singh
