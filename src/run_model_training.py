"""
Run model training for sentiment analysis
This script trains Logistic Regression and Naive Bayes models
"""

import sys
import os
import pandas as pd
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__)))

from model_training import train_sentiment_models

def main():
    """Main function to run model training"""
    
    print("=== MODEL TRAINING ===")
    print("Training sentiment analysis models...")
    
    try:
        # Load preprocessed data
        df = pd.read_csv('data/preprocessed_reviews.csv')
        print(f"✓ Data loaded: {df.shape}")
        
        # Load TF-IDF vectorizer
        with open('results/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✓ Vectorizer loaded: {len(vectorizer.get_feature_names_out())} features")
        
        # Transform text to features
        X = vectorizer.transform(df['cleaned_review'])
        y = df['sentiment'].values
        
        print(f"✓ Features created: {X.shape}")
        print(f"✓ Class distribution: {dict(zip(*zip(*enumerate(y)), [[len([i for i in y if i == c]) for c in [0, 1]]]))}")
        
        # Train models
        trainer, comparison_df = train_sentiment_models(
            X, y, test_size=0.2, output_dir='results'
        )
        
        print("\n=== TRAINING SUMMARY ===")
        print(comparison_df.round(4))
        
        # Show feature importance for Logistic Regression
        feature_names = vectorizer.get_feature_names_out()
        try:
            top_positive, top_negative = trainer.get_feature_importance(
                'logistic_regression', feature_names, top_n=5
            )
            
            print("\n=== TOP POSITIVE FEATURES ===")
            for feature, coef in top_positive:
                print(f"{feature}: {coef:.4f}")
            
            print("\n=== TOP NEGATIVE FEATURES ===")
            for feature, coef in top_negative:
                print(f"{feature}: {coef:.4f}")
                
        except Exception as e:
            print(f"Note: Could not extract feature importance: {e}")
        
        print("\n✓ Model training completed successfully!")
        print("✓ Models saved to results/ directory")
        print("Ready for visualization step.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
