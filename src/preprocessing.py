"""
Text preprocessing utilities for sentiment analysis
This module contains functions for cleaning and preprocessing text data
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """
    A class to handle text preprocessing for sentiment analysis
    """
    
    def __init__(self):
        """Initialize the preprocessor with required NLTK components"""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        # Initialize components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean text by converting to lowercase and removing punctuation
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without stopwords
        """
        if not isinstance(text, str):
            return ""
        
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        return ' '.join(filtered_text)
    
    def stem_text(self, text):
        """
        Apply stemming to text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Stemmed text
        """
        if not isinstance(text, str):
            return ""
        
        word_tokens = word_tokenize(text)
        stemmed_text = [self.stemmer.stem(word) for word in word_tokens]
        return ' '.join(stemmed_text)
    
    def lemmatize_text(self, text):
        """
        Apply lemmatization to text (less aggressive than stemming)
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lemmatized text
        """
        if not isinstance(text, str):
            return ""
        
        word_tokens = word_tokenize(text)
        lemmatized_text = [self.lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_text)
    
    def preprocess_text(self, text):
        """
        Complete text preprocessing pipeline
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Fully preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords
        text = self.remove_stopwords(text)
        
        # Apply lemmatization instead of aggressive stemming
        text = self.lemmatize_text(text)
        
        return text
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts
        
        Args:
            texts (list): List of texts to preprocess
            
        Returns:
            list: List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]

# Convenience functions for backward compatibility
def preprocess_text(text):
    """
    Convenience function to preprocess a single text
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_text(text)

def preprocess_batch(texts):
    """
    Convenience function to preprocess a batch of texts
    
    Args:
        texts (list): List of texts
        
    Returns:
        list: List of preprocessed texts
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_batch(texts)
