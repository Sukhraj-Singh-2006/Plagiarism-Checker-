"""
Plagiarism Checker Module

A Python library to detect plagiarism by comparing text documents
using cosine similarity and TF-IDF (Term Frequency-Inverse Document Frequency).
"""

import math
import re
from collections import Counter
from typing import List, Dict, Tuple


class PlagiarismChecker:
    """
    A class to check for plagiarism between text documents using TF-IDF and cosine similarity.
    """
    
    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        """
        Preprocess text by converting to lowercase and extracting words.
        
        Args:
            text: Input text string
            
        Returns:
            List of words (tokens) from the text
        """
        # Convert to lowercase and extract words (alphanumeric sequences)
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    @staticmethod
    def calculate_term_frequency(words: List[str]) -> Dict[str, float]:
        """
        Calculate term frequency for each word in the document.
        
        Args:
            words: List of words from a document
            
        Returns:
            Dictionary mapping words to their term frequencies
        """
        word_count = Counter(words)
        total_words = len(words)
        
        tf = {}
        for word, count in word_count.items():
            tf[word] = count / total_words if total_words > 0 else 0
        
        return tf
    
    @staticmethod
    def calculate_inverse_document_frequency(documents: List[List[str]]) -> Dict[str, float]:
        """
        Calculate inverse document frequency for words across all documents.
        Uses smoothed IDF to avoid zero values.
        
        Args:
            documents: List of documents, where each document is a list of words
            
        Returns:
            Dictionary mapping words to their IDF values
        """
        total_documents = len(documents)
        if total_documents == 0:
            return {}
        
        # Count how many documents contain each word
        word_doc_count = {}
        for doc in documents:
            unique_words = set(doc)
            for word in unique_words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        # Calculate IDF with smoothing
        # Using: log((1 + total_docs) / (1 + doc_count)) + 1
        # This ensures IDF is never zero and provides better discrimination
        idf = {}
        for word, doc_count in word_doc_count.items():
            idf[word] = math.log((1 + total_documents) / (1 + doc_count)) + 1
        
        return idf
    
    @staticmethod
    def calculate_tf_idf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores.
        
        Args:
            tf: Term frequency dictionary
            idf: Inverse document frequency dictionary
            
        Returns:
            Dictionary mapping words to their TF-IDF scores
        """
        tf_idf = {}
        for word, tf_value in tf.items():
            tf_idf[word] = tf_value * idf.get(word, 0)
        
        return tf_idf
    
    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two TF-IDF vectors.
        
        Args:
            vec1: First TF-IDF vector
            vec2: Second TF-IDF vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Get all unique words
        all_words = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate dot product
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
        magnitude2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return similarity
    
    def compare_documents(self, text1: str, text2: str) -> float:
        """
        Compare two text documents and return their similarity score.
        
        Args:
            text1: First document text
            text2: Second document text
            
        Returns:
            Similarity score between 0 and 1 (1 means identical)
        """
        # Preprocess documents
        words1 = self.preprocess_text(text1)
        words2 = self.preprocess_text(text2)
        
        # Calculate TF for each document
        tf1 = self.calculate_term_frequency(words1)
        tf2 = self.calculate_term_frequency(words2)
        
        # Calculate IDF across both documents
        idf = self.calculate_inverse_document_frequency([words1, words2])
        
        # Calculate TF-IDF vectors
        tf_idf1 = self.calculate_tf_idf(tf1, idf)
        tf_idf2 = self.calculate_tf_idf(tf2, idf)
        
        # Calculate and return cosine similarity
        similarity = self.cosine_similarity(tf_idf1, tf_idf2)
        
        return similarity
    
    def compare_multiple_documents(self, texts: List[str]) -> List[Tuple[int, int, float]]:
        """
        Compare multiple documents and return pairwise similarity scores.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of tuples (doc1_index, doc2_index, similarity_score)
        """
        if len(texts) < 2:
            return []
        
        # Preprocess all documents
        all_words = [self.preprocess_text(text) for text in texts]
        
        # Calculate TF for each document
        all_tf = [self.calculate_term_frequency(words) for words in all_words]
        
        # Calculate IDF across all documents
        idf = self.calculate_inverse_document_frequency(all_words)
        
        # Calculate TF-IDF vectors for all documents
        all_tf_idf = [self.calculate_tf_idf(tf, idf) for tf in all_tf]
        
        # Compare all pairs of documents
        results = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = self.cosine_similarity(all_tf_idf[i], all_tf_idf[j])
                results.append((i, j, similarity))
        
        return results


def check_plagiarism(text1: str, text2: str) -> float:
    """
    Convenience function to check plagiarism between two documents.
    
    Args:
        text1: First document text
        text2: Second document text
        
    Returns:
        Similarity score between 0 and 1
    """
    checker = PlagiarismChecker()
    return checker.compare_documents(text1, text2)
