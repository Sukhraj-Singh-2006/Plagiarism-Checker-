"""
Tests for the Plagiarism Checker module.
"""

import unittest
from plagiarism_checker import PlagiarismChecker, check_plagiarism


class TestPlagiarismChecker(unittest.TestCase):
    """Test cases for PlagiarismChecker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = PlagiarismChecker()
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "Hello World! This is a TEST."
        words = self.checker.preprocess_text(text)
        self.assertEqual(words, ['hello', 'world', 'this', 'is', 'a', 'test'])
    
    def test_preprocess_text_with_numbers(self):
        """Test preprocessing with numbers."""
        text = "Test 123 and ABC456"
        words = self.checker.preprocess_text(text)
        self.assertEqual(words, ['test', '123', 'and', 'abc456'])
    
    def test_calculate_term_frequency(self):
        """Test term frequency calculation."""
        words = ['hello', 'world', 'hello', 'test']
        tf = self.checker.calculate_term_frequency(words)
        self.assertEqual(tf['hello'], 0.5)
        self.assertEqual(tf['world'], 0.25)
        self.assertEqual(tf['test'], 0.25)
    
    def test_calculate_term_frequency_empty(self):
        """Test term frequency with empty list."""
        words = []
        tf = self.checker.calculate_term_frequency(words)
        self.assertEqual(tf, {})
    
    def test_calculate_inverse_document_frequency(self):
        """Test IDF calculation."""
        docs = [
            ['hello', 'world'],
            ['hello', 'test'],
            ['world', 'test']
        ]
        idf = self.checker.calculate_inverse_document_frequency(docs)
        
        # 'hello' appears in 2 out of 3 documents
        # Smoothed IDF = log((1+3)/(1+2)) + 1 = log(4/3) + 1 ≈ 1.288
        self.assertAlmostEqual(idf['hello'], 1.288, places=2)
        
        # 'world' and 'test' also appear in 2 out of 3 documents
        self.assertAlmostEqual(idf['world'], 1.288, places=2)
        self.assertAlmostEqual(idf['test'], 1.288, places=2)
    
    def test_calculate_inverse_document_frequency_empty(self):
        """Test IDF with empty document list."""
        docs = []
        idf = self.checker.calculate_inverse_document_frequency(docs)
        self.assertEqual(idf, {})
    
    def test_calculate_tf_idf(self):
        """Test TF-IDF calculation."""
        tf = {'hello': 0.5, 'world': 0.5}
        idf = {'hello': 1.0, 'world': 2.0}
        tf_idf = self.checker.calculate_tf_idf(tf, idf)
        
        self.assertEqual(tf_idf['hello'], 0.5)
        self.assertEqual(tf_idf['world'], 1.0)
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        vec1 = {'hello': 1.0, 'world': 1.0}
        vec2 = {'hello': 1.0, 'world': 1.0}
        similarity = self.checker.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_cosine_similarity_different(self):
        """Test cosine similarity with completely different vectors."""
        vec1 = {'hello': 1.0, 'world': 1.0}
        vec2 = {'foo': 1.0, 'bar': 1.0}
        similarity = self.checker.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.0, places=5)
    
    def test_cosine_similarity_empty(self):
        """Test cosine similarity with empty vectors."""
        vec1 = {}
        vec2 = {'hello': 1.0}
        similarity = self.checker.cosine_similarity(vec1, vec2)
        self.assertEqual(similarity, 0.0)
    
    def test_compare_documents_identical(self):
        """Test comparing identical documents."""
        text = "This is a test document with some words."
        similarity = self.checker.compare_documents(text, text)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_compare_documents_similar(self):
        """Test comparing similar documents."""
        text1 = "This is a test document about Python programming."
        text2 = "This is a test document about Python coding."
        similarity = self.checker.compare_documents(text1, text2)
        # Should be high similarity (> 0.7) since most words are the same
        self.assertGreater(similarity, 0.7)
    
    def test_compare_documents_different(self):
        """Test comparing completely different documents."""
        text1 = "Python is a programming language."
        text2 = "The quick brown fox jumps over the lazy dog."
        similarity = self.checker.compare_documents(text1, text2)
        # Should be low similarity (< 0.3)
        self.assertLess(similarity, 0.3)
    
    def test_compare_documents_empty(self):
        """Test comparing with empty documents."""
        text1 = "Some text here"
        text2 = ""
        similarity = self.checker.compare_documents(text1, text2)
        self.assertEqual(similarity, 0.0)
    
    def test_compare_multiple_documents(self):
        """Test comparing multiple documents."""
        texts = [
            "This is document one about programming.",
            "This is document two about programming.",
            "Completely different topic and content."
        ]
        results = self.checker.compare_multiple_documents(texts)
        
        # Should have 3 comparisons: (0,1), (0,2), (1,2)
        self.assertEqual(len(results), 3)
        
        # Check that results are tuples with correct structure
        for i, j, score in results:
            self.assertIsInstance(i, int)
            self.assertIsInstance(j, int)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # First two documents should be more similar than others
        doc0_1_similarity = results[0][2]  # (0, 1, score)
        self.assertGreater(doc0_1_similarity, 0.7)
    
    def test_compare_multiple_documents_insufficient(self):
        """Test comparing with insufficient documents."""
        texts = ["Only one document"]
        results = self.checker.compare_multiple_documents(texts)
        self.assertEqual(results, [])
    
    def test_check_plagiarism_convenience_function(self):
        """Test the convenience function."""
        text1 = "This is a test."
        text2 = "This is a test."
        similarity = check_plagiarism(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=5)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world plagiarism detection scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = PlagiarismChecker()
    
    def test_paraphrased_content(self):
        """Test detection of paraphrased content."""
        original = "Machine learning is a subset of artificial intelligence that focuses on data and algorithms."
        paraphrased = "Machine learning, a part of artificial intelligence, concentrates on data and algorithms."
        
        similarity = self.checker.compare_documents(original, paraphrased)
        # Paraphrased content should have moderate to high similarity
        self.assertGreater(similarity, 0.5)
    
    def test_added_content(self):
        """Test when additional content is added."""
        text1 = "Python is a programming language."
        text2 = "Python is a programming language. It is widely used for web development, data science, and automation."
        
        similarity = self.checker.compare_documents(text1, text2)
        # Should still have some similarity
        self.assertGreater(similarity, 0.3)
    
    def test_word_order_change(self):
        """Test when word order is changed."""
        text1 = "The cat sat on the mat."
        text2 = "The mat had the cat sitting on it."
        
        similarity = self.checker.compare_documents(text1, text2)
        # TF-IDF doesn't consider word order, so similar words = similar documents
        self.assertGreater(similarity, 0.4)


if __name__ == '__main__':
    unittest.main()
