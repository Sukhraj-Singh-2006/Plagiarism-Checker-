"""
Plagiarism Checker - A tool to detect similarity between text documents.

This module provides functionality to compare text documents using TF-IDF
(Term Frequency-Inverse Document Frequency) and cosine similarity.
"""

import re
import math
from collections import Counter
from typing import List, Tuple, Dict


class PlagiarismChecker:
    """
    A class to check plagiarism between text documents using TF-IDF and cosine similarity.
    """
    
    def __init__(self):
        """Initialize the PlagiarismChecker."""
        self.documents = []
        self.document_names = []
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by converting to lowercase and extracting words.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed words
        """
        # Convert to lowercase and extract words (alphanumeric sequences)
        text = text.lower()
        words = re.findall(r'\w+', text)
        return words
    
    def calculate_tf(self, words: List[str]) -> Dict[str, float]:
        """
        Calculate Term Frequency for a document.
        
        Args:
            words: List of words in the document
            
        Returns:
            Dictionary mapping words to their TF values
        """
        word_count = Counter(words)
        total_words = len(words)
        
        tf = {}
        for word, count in word_count.items():
            tf[word] = count / total_words if total_words > 0 else 0
        
        return tf
    
    def calculate_idf(self, documents_words: List[List[str]]) -> Dict[str, float]:
        """
        Calculate Inverse Document Frequency across all documents.
        
        Args:
            documents_words: List of documents, each represented as a list of words
            
        Returns:
            Dictionary mapping words to their IDF values
        """
        num_documents = len(documents_words)
        
        # Count in how many documents each word appears
        word_doc_count = Counter()
        for words in documents_words:
            unique_words = set(words)
            for word in unique_words:
                word_doc_count[word] += 1
        
        idf = {}
        for word, doc_count in word_doc_count.items():
            # Use smoothed IDF to avoid zero values: log((N + 1) / (df + 1)) + 1
            idf[word] = math.log((num_documents + 1) / (doc_count + 1)) + 1
        
        return idf
    
    def calculate_tfidf(self, tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for a document.
        
        Args:
            tf: Term Frequency dictionary
            idf: Inverse Document Frequency dictionary
            
        Returns:
            Dictionary mapping words to their TF-IDF values
        """
        tfidf = {}
        for word, tf_value in tf.items():
            tfidf[word] = tf_value * idf.get(word, 0)
        
        return tfidf
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two TF-IDF vectors.
        
        Args:
            vec1: First TF-IDF vector (dictionary)
            vec2: Second TF-IDF vector (dictionary)
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Get all unique words from both vectors
        all_words = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate dot product
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        # Calculate cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return similarity
    
    def add_document(self, text: str, name: str = None):
        """
        Add a document to the checker for comparison.
        
        Args:
            text: Document text content
            name: Optional name/identifier for the document
        """
        words = self.preprocess_text(text)
        self.documents.append(words)
        self.document_names.append(name or f"Document {len(self.documents)}")
    
    def compare_documents(self, text1: str, text2: str) -> float:
        """
        Compare two documents and return their similarity score.
        
        Args:
            text1: First document text
            text2: Second document text
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = self.preprocess_text(text1)
        words2 = self.preprocess_text(text2)
        
        # Calculate TF for both documents
        tf1 = self.calculate_tf(words1)
        tf2 = self.calculate_tf(words2)
        
        # Calculate IDF across both documents
        idf = self.calculate_idf([words1, words2])
        
        # Calculate TF-IDF for both documents
        tfidf1 = self.calculate_tfidf(tf1, idf)
        tfidf2 = self.calculate_tfidf(tf2, idf)
        
        # Calculate cosine similarity
        similarity = self.cosine_similarity(tfidf1, tfidf2)
        
        return similarity
    
    def check_all_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Check plagiarism between all pairs of added documents.
        
        Returns:
            List of tuples containing (doc1_name, doc2_name, similarity_score)
        """
        if len(self.documents) < 2:
            return []
        
        # Calculate IDF across all documents
        idf = self.calculate_idf(self.documents)
        
        # Calculate TF-IDF for all documents
        tfidf_vectors = []
        for words in self.documents:
            tf = self.calculate_tf(words)
            tfidf = self.calculate_tfidf(tf, idf)
            tfidf_vectors.append(tfidf)
        
        # Compare all pairs
        results = []
        for i in range(len(self.documents)):
            for j in range(i + 1, len(self.documents)):
                similarity = self.cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
                results.append((
                    self.document_names[i],
                    self.document_names[j],
                    similarity
                ))
        
        return results
    
    def clear_documents(self):
        """Clear all added documents."""
        self.documents = []
        self.document_names = []


def read_file(filepath: str) -> str:
    """
    Read text content from a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Content of the file as a string
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
