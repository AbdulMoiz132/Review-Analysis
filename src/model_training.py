"""
Model training utilities for sentiment analysis
This module handles training and evaluation of machine learning models
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, train_test_split

class SentimentModelTrainer:
    """
    A class to handle sentiment analysis model training and evaluation
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, X, y, test_size=0.2, stratify=True):
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size (float): Fraction of data for testing
            stratify (bool): Whether to stratify the split
        """
        stratify_param = y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_param
        )
        
        print(f"Data split completed:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Train labels distribution: {np.bincount(self.y_train)}")
        print(f"Test labels distribution: {np.bincount(self.y_test)}")
    
    def train_logistic_regression(self, C=1.0, max_iter=1000):
        """
        Train a Logistic Regression model
        
        Args:
            C (float): Regularization strength
            max_iter (int): Maximum iterations
            
        Returns:
            LogisticRegression: Trained model
        """
        print("Training Logistic Regression model...")
        
        lr_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=max_iter,
            C=C
        )
        
        lr_model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = lr_model
        
        print("✓ Logistic Regression model trained successfully!")
        return lr_model
    
    def train_naive_bayes(self, alpha=1.0):
        """
        Train a Naive Bayes model
        
        Args:
            alpha (float): Laplace smoothing parameter
            
        Returns:
            MultinomialNB: Trained model
        """
        print("Training Naive Bayes model...")
        
        nb_model = MultinomialNB(alpha=alpha)
        nb_model.fit(self.X_train, self.y_train)
        self.models['naive_bayes'] = nb_model
        
        print("✓ Naive Bayes model trained successfully!")
        return nb_model
    
    def evaluate_model(self, model_name):
        """
        Evaluate a trained model
        
        Args:
            model_name (str): Name of the model to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train the model first.")
        
        model = self.models[model_name]
        
        # Make predictions
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        test_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(self.y_train, train_pred),
            'test_accuracy': accuracy_score(self.y_test, test_pred),
            'precision': precision_score(self.y_test, test_pred),
            'recall': recall_score(self.y_test, test_pred),
            'f1_score': f1_score(self.y_test, test_pred),
            'auc_roc': roc_auc_score(self.y_test, test_proba),
            'confusion_matrix': confusion_matrix(self.y_test, test_pred),
            'classification_report': classification_report(self.y_test, test_pred),
            'test_predictions': test_pred,
            'test_probabilities': test_proba
        }
        
        # Store results
        self.results[model_name] = metrics
        
        print(f"{model_name.replace('_', ' ').title()} Performance:")
        print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model_name, cv=5, scoring='accuracy'):
        """
        Perform cross-validation on a model
        
        Args:
            model_name (str): Name of the model
            cv (int): Number of folds
            scoring (str): Scoring metric
            
        Returns:
            numpy.ndarray: CV scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train the model first.")
        
        model = self.models[model_name]
        
        # Combine train and test for full cross-validation
        X_full = np.vstack([self.X_train.toarray(), self.X_test.toarray()])
        y_full = np.hstack([self.y_train, self.y_test])
        
        cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring=scoring)
        
        print(f"{model_name.replace('_', ' ').title()} CV Scores: {cv_scores}")
        print(f"Mean CV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store CV results
        if model_name in self.results:
            self.results[model_name]['cv_scores'] = cv_scores
            self.results[model_name]['cv_mean'] = cv_scores.mean()
            self.results[model_name]['cv_std'] = cv_scores.std()
        
        return cv_scores
    
    def compare_models(self):
        """
        Compare all trained models
        
        Returns:
            pandas.DataFrame: Comparison table
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet.")
        
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train Accuracy': metrics['train_accuracy'],
                'Test Accuracy': metrics['test_accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC-ROC': metrics['auc_roc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("Model Comparison:")
        print("=" * 80)
        print(comparison_df.round(4))
        
        # Determine best model
        best_idx = comparison_df['F1-Score'].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        print(f"\nBest performing model: {best_model}")
        
        return comparison_df
    
    def get_feature_importance(self, model_name, feature_names, top_n=10):
        """
        Get feature importance for Logistic Regression
        
        Args:
            model_name (str): Name of the model
            feature_names (array): Names of features
            top_n (int): Number of top features to return
            
        Returns:
            tuple: (top_positive_features, top_negative_features)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'coef_'):
            raise ValueError(f"Model '{model_name}' does not have feature coefficients.")
        
        coef = model.coef_[0]
        
        # Get top positive and negative features
        top_positive_idx = np.argsort(coef)[-top_n:]
        top_negative_idx = np.argsort(coef)[:top_n]
        
        top_positive = [(feature_names[idx], coef[idx]) for idx in reversed(top_positive_idx)]
        top_negative = [(feature_names[idx], coef[idx]) for idx in top_negative_idx]
        
        return top_positive, top_negative
    
    def save_models(self, output_dir='results'):
        """
        Save trained models and results
        
        Args:
            output_dir (str): Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            filename = f"{model_name}_model.pkl"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ {model_name.replace('_', ' ').title()} model saved to {filepath}")
        
        # Save results
        results_filepath = os.path.join(output_dir, 'model_results.pkl')
        with open(results_filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"✓ Results saved to {results_filepath}")
    
    @staticmethod
    def load_models(input_dir='results'):
        """
        Load trained models from disk
        
        Args:
            input_dir (str): Directory to load models from
            
        Returns:
            dict: Loaded models
        """
        models = {}
        
        model_files = {
            'logistic_regression': 'logistic_regression_model.pkl',
            'naive_bayes': 'naive_bayes_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(input_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"✓ {model_name.replace('_', ' ').title()} model loaded")
        
        return models

def train_sentiment_models(X, y, test_size=0.2, output_dir='results'):
    """
    Convenience function to train and evaluate sentiment analysis models
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size (float): Test set size
        output_dir (str): Directory to save results
        
    Returns:
        tuple: (trainer, comparison_df)
    """
    # Initialize trainer
    trainer = SentimentModelTrainer()
    
    # Prepare data
    trainer.prepare_data(X, y, test_size=test_size)
    
    # Train models
    trainer.train_logistic_regression()
    trainer.train_naive_bayes()
    
    # Evaluate models
    trainer.evaluate_model('logistic_regression')
    trainer.evaluate_model('naive_bayes')
    
    # Cross-validation
    trainer.cross_validate_model('logistic_regression')
    trainer.cross_validate_model('naive_bayes')
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Save models
    trainer.save_models(output_dir)
    
    return trainer, comparison_df
