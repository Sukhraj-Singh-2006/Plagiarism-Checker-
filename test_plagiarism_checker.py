"""
Unit tests for the Plagiarism Checker.
"""

import unittest
import math
from plagiarism_checker import PlagiarismChecker


class TestPlagiarismChecker(unittest.TestCase):
    """Test cases for the PlagiarismChecker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = PlagiarismChecker()
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "Hello World! This is a TEST."
        result = self.checker.preprocess_text(text)
        self.assertEqual(result, ['hello', 'world', 'this', 'is', 'a', 'test'])
    
    def test_preprocess_text_with_numbers(self):
        """Test preprocessing with numbers."""
        text = "Python 3.11 was released in 2022"
        result = self.checker.preprocess_text(text)
        self.assertEqual(result, ['python', '3', '11', 'was', 'released', 'in', '2022'])
    
    def test_calculate_tf(self):
        """Test Term Frequency calculation."""
        words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
        tf = self.checker.calculate_tf(words)
        
        self.assertAlmostEqual(tf['apple'], 3/6)
        self.assertAlmostEqual(tf['banana'], 2/6)
        self.assertAlmostEqual(tf['cherry'], 1/6)
    
    def test_calculate_tf_empty(self):
        """Test TF calculation with empty list."""
        words = []
        tf = self.checker.calculate_tf(words)
        self.assertEqual(tf, {})
    
    def test_calculate_idf(self):
        """Test Inverse Document Frequency calculation."""
        docs = [
            ['apple', 'banana'],
            ['apple', 'cherry'],
            ['banana', 'cherry']
        ]
        idf = self.checker.calculate_idf(docs)
        
        # All words appear in 2 out of 3 documents
        # Using smoothed IDF: log((N + 1) / (df + 1)) + 1
        expected_idf = math.log((3 + 1) / (2 + 1)) + 1
        self.assertAlmostEqual(idf['apple'], expected_idf)
        self.assertAlmostEqual(idf['banana'], expected_idf)
        self.assertAlmostEqual(idf['cherry'], expected_idf)
    
    def test_calculate_tfidf(self):
        """Test TF-IDF calculation."""
        tf = {'apple': 0.5, 'banana': 0.3}
        idf = {'apple': 1.5, 'banana': 2.0, 'cherry': 1.0}
        
        tfidf = self.checker.calculate_tfidf(tf, idf)
        
        self.assertAlmostEqual(tfidf['apple'], 0.5 * 1.5)
        self.assertAlmostEqual(tfidf['banana'], 0.3 * 2.0)
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        vec = {'apple': 0.5, 'banana': 0.3, 'cherry': 0.2}
        similarity = self.checker.cosine_similarity(vec, vec)
        self.assertAlmostEqual(similarity, 1.0)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = {'apple': 1.0}
        vec2 = {'banana': 1.0}
        similarity = self.checker.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.0)
    
    def test_cosine_similarity_empty(self):
        """Test cosine similarity with empty vectors."""
        vec1 = {}
        vec2 = {'apple': 1.0}
        similarity = self.checker.cosine_similarity(vec1, vec2)
        self.assertEqual(similarity, 0.0)
    
    def test_compare_documents_identical(self):
        """Test comparing identical documents."""
        text = "This is a sample text document."
        similarity = self.checker.compare_documents(text, text)
        self.assertAlmostEqual(similarity, 1.0)
    
    def test_compare_documents_different(self):
        """Test comparing completely different documents."""
        text1 = "Python is a programming language."
        text2 = "The quick brown fox jumps over the lazy dog."
        similarity = self.checker.compare_documents(text1, text2)
        self.assertLess(similarity, 0.5)
    
    def test_compare_documents_similar(self):
        """Test comparing similar documents."""
        text1 = "Python is a high-level programming language used for web development."
        text2 = "Python is a popular programming language utilized in web development."
        similarity = self.checker.compare_documents(text1, text2)
        self.assertGreater(similarity, 0.5)
    
    def test_add_document(self):
        """Test adding documents."""
        self.checker.add_document("First document", "Doc1")
        self.checker.add_document("Second document")
        
        self.assertEqual(len(self.checker.documents), 2)
        self.assertEqual(self.checker.document_names[0], "Doc1")
        self.assertEqual(self.checker.document_names[1], "Document 2")
    
    def test_check_all_pairs_empty(self):
        """Test checking pairs with no documents."""
        results = self.checker.check_all_pairs()
        self.assertEqual(results, [])
    
    def test_check_all_pairs_one_document(self):
        """Test checking pairs with only one document."""
        self.checker.add_document("Single document")
        results = self.checker.check_all_pairs()
        self.assertEqual(results, [])
    
    def test_check_all_pairs_three_documents(self):
        """Test checking all pairs with three documents."""
        self.checker.add_document("Python programming", "Doc1")
        self.checker.add_document("Python programming", "Doc2")  # Identical to Doc1
        self.checker.add_document("Java programming", "Doc3")
        
        results = self.checker.check_all_pairs()
        
        # Should have 3 comparisons: (Doc1, Doc2), (Doc1, Doc3), (Doc2, Doc3)
        self.assertEqual(len(results), 3)
        
        # Find the comparison between Doc1 and Doc2 (should be identical)
        doc1_doc2 = next(r for r in results if r[0] == "Doc1" and r[1] == "Doc2")
        self.assertAlmostEqual(doc1_doc2[2], 1.0, places=5)
    
    def test_clear_documents(self):
        """Test clearing documents."""
        self.checker.add_document("Document 1")
        self.checker.add_document("Document 2")
        
        self.assertEqual(len(self.checker.documents), 2)
        
        self.checker.clear_documents()
        
        self.assertEqual(len(self.checker.documents), 0)
        self.assertEqual(len(self.checker.document_names), 0)
    
    def test_compare_empty_documents(self):
        """Test comparing empty documents."""
        similarity = self.checker.compare_documents("", "")
        self.assertEqual(similarity, 0.0)
    
    def test_compare_one_empty_document(self):
        """Test comparing when one document is empty."""
        text = "This is a document."
        similarity = self.checker.compare_documents(text, "")
        self.assertEqual(similarity, 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = PlagiarismChecker()
    
    def test_special_characters(self):
        """Test handling of special characters."""
        text1 = "Hello! How are you? I'm fine."
        text2 = "Hello!!! How are you??? I'm great."
        
        # Should still find similarity despite different punctuation
        similarity = self.checker.compare_documents(text1, text2)
        self.assertGreater(similarity, 0.5)
    
    def test_case_insensitive(self):
        """Test that comparison is case-insensitive."""
        text1 = "Python Is A Programming Language"
        text2 = "python is a programming language"
        
        similarity = self.checker.compare_documents(text1, text2)
        self.assertAlmostEqual(similarity, 1.0)
    
    def test_word_order_matters(self):
        """Test that word order affects similarity."""
        text1 = "The cat sat on the mat"
        text2 = "The mat sat on the cat"
        
        # Should have high similarity (same words) but not perfect (different context)
        similarity = self.checker.compare_documents(text1, text2)
        self.assertGreater(similarity, 0.9)


if __name__ == '__main__':
    unittest.main()
