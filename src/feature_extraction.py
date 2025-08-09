"""
Feature extraction utilities for sentiment analysis
This module handles TF-IDF and Count vectorization of text data
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

class FeatureExtractor:
    """
    A class to handle feature extraction for sentiment analysis
    """
    
    def __init__(self, max_features=1000, min_df=2, max_df=0.8, ngram_range=(1, 2)):
        """
        Initialize the feature extractor
        
        Args:
            max_features (int): Maximum number of features to extract
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency (as fraction)
            ngram_range (tuple): Range of n-grams to consider
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=True
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=True
        )
        
        # Storage for processed data
        self.X_tfidf = None
        self.X_count = None
        self.y = None
        self.feature_names_tfidf = None
        self.feature_names_count = None
    
    def fit_transform_tfidf(self, texts, labels=None):
        """
        Fit TF-IDF vectorizer and transform texts
        
        Args:
            texts (list or pandas.Series): Text data to transform
            labels (list or pandas.Series): Corresponding labels (optional)
            
        Returns:
            scipy.sparse matrix: TF-IDF transformed features
        """
        print("Applying TF-IDF vectorization...")
        self.X_tfidf = self.tfidf_vectorizer.fit_transform(texts)
        self.feature_names_tfidf = self.tfidf_vectorizer.get_feature_names_out()
        
        if labels is not None:
            self.y = np.array(labels)
        
        print(f"✓ TF-IDF matrix shape: {self.X_tfidf.shape}")
        print(f"✓ Number of features: {len(self.feature_names_tfidf)}")
        print(f"✓ Matrix sparsity: {1 - (self.X_tfidf.nnz / (self.X_tfidf.shape[0] * self.X_tfidf.shape[1])):.3f}")
        
        return self.X_tfidf
    
    def fit_transform_count(self, texts, labels=None):
        """
        Fit Count vectorizer and transform texts
        
        Args:
            texts (list or pandas.Series): Text data to transform
            labels (list or pandas.Series): Corresponding labels (optional)
            
        Returns:
            scipy.sparse matrix: Count transformed features
        """
        print("Applying Count vectorization...")
        self.X_count = self.count_vectorizer.fit_transform(texts)
        self.feature_names_count = self.count_vectorizer.get_feature_names_out()
        
        if labels is not None and self.y is None:
            self.y = np.array(labels)
        
        print(f"✓ Count matrix shape: {self.X_count.shape}")
        print(f"✓ Number of features: {len(self.feature_names_count)}")
        print(f"✓ Matrix sparsity: {1 - (self.X_count.nnz / (self.X_count.shape[0] * self.X_count.shape[1])):.3f}")
        
        return self.X_count
    
    def get_feature_importance(self, method='tfidf', top_n=20):
        """
        Get top features by importance
        
        Args:
            method (str): 'tfidf' or 'count'
            top_n (int): Number of top features to return
            
        Returns:
            pandas.DataFrame: Top features with their scores
        """
        if method == 'tfidf' and self.X_tfidf is not None:
            X = self.X_tfidf
            feature_names = self.feature_names_tfidf
        elif method == 'count' and self.X_count is not None:
            X = self.X_count
            feature_names = self.feature_names_count
        else:
            raise ValueError(f"Method '{method}' not available or data not fitted")
        
        # Calculate mean scores
        mean_scores = np.array(X.mean(axis=0)).flatten()
        
        # Create DataFrame
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'mean_score': mean_scores
        }).sort_values('mean_score', ascending=False)
        
        return feature_df.head(top_n)
    
    def split_data(self, test_size=0.2, random_state=42, stratify=True):
        """
        Split data into training and testing sets
        
        Args:
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducibility
            stratify (bool): Whether to stratify the split
            
        Returns:
            tuple: (X_train_tfidf, X_test_tfidf, X_train_count, X_test_count, y_train, y_test)
        """
        if self.X_tfidf is None or self.y is None:
            raise ValueError("Data must be fitted before splitting")
        
        stratify_param = self.y if stratify else None
        
        # Split TF-IDF data
        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
            self.X_tfidf, self.y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # Split Count data if available
        if self.X_count is not None:
            X_train_count, X_test_count, _, _ = train_test_split(
                self.X_count, self.y, test_size=test_size, random_state=random_state, stratify=stratify_param
            )
        else:
            X_train_count, X_test_count = None, None
        
        print(f"Data split completed:")
        print(f"Training set: {X_train_tfidf.shape[0]} samples")
        print(f"Test set: {X_test_tfidf.shape[0]} samples")
        print(f"Training labels distribution: {np.bincount(y_train)}")
        print(f"Test labels distribution: {np.bincount(y_test)}")
        
        return X_train_tfidf, X_test_tfidf, X_train_count, X_test_count, y_train, y_test
    
    def save_vectorizers(self, output_dir='results'):
        """
        Save fitted vectorizers to disk
        
        Args:
            output_dir (str): Directory to save vectorizers
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save TF-IDF vectorizer
        if self.tfidf_vectorizer:
            with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print(f"✓ TF-IDF vectorizer saved to {output_dir}/")
        
        # Save Count vectorizer
        if self.count_vectorizer:
            with open(os.path.join(output_dir, 'count_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.count_vectorizer, f)
            print(f"✓ Count vectorizer saved to {output_dir}/")
        
        # Save feature information
        feature_info = {
            'tfidf_features': self.feature_names_tfidf.tolist() if self.feature_names_tfidf is not None else None,
            'count_features': self.feature_names_count.tolist() if self.feature_names_count is not None else None,
            'vectorizer_params': {
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': self.ngram_range
            }
        }
        
        with open(os.path.join(output_dir, 'feature_info.pkl'), 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"✓ Feature information saved to {output_dir}/")
    
    @staticmethod
    def load_vectorizers(input_dir='results'):
        """
        Load vectorizers from disk
        
        Args:
            input_dir (str): Directory to load vectorizers from
            
        Returns:
            tuple: (tfidf_vectorizer, count_vectorizer, feature_info)
        """
        tfidf_path = os.path.join(input_dir, 'tfidf_vectorizer.pkl')
        count_path = os.path.join(input_dir, 'count_vectorizer.pkl')
        info_path = os.path.join(input_dir, 'feature_info.pkl')
        
        tfidf_vectorizer = None
        count_vectorizer = None
        feature_info = None
        
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
        
        if os.path.exists(count_path):
            with open(count_path, 'rb') as f:
                count_vectorizer = pickle.load(f)
        
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                feature_info = pickle.load(f)
        
        return tfidf_vectorizer, count_vectorizer, feature_info

def extract_features_from_csv(csv_path, text_column='cleaned_review', label_column='sentiment', 
                             max_features=1000, test_size=0.2, output_dir='results'):
    """
    Convenience function to extract features from a CSV file
    
    Args:
        csv_path (str): Path to CSV file with text data
        text_column (str): Name of column containing text
        label_column (str): Name of column containing labels
        max_features (int): Maximum number of features
        test_size (float): Test set size
        output_dir (str): Directory to save results
        
    Returns:
        tuple: (extractor, train_test_data)
    """
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    # Initialize extractor
    extractor = FeatureExtractor(max_features=max_features)
    
    # Extract features
    extractor.fit_transform_tfidf(df[text_column], df[label_column])
    extractor.fit_transform_count(df[text_column], df[label_column])
    
    # Split data
    splits = extractor.split_data(test_size=test_size)
    
    # Save everything
    extractor.save_vectorizers(output_dir)
    
    return extractor, splits
