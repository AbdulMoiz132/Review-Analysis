"""
Run feature extraction on preprocessed reviews
This script applies TF-IDF and Count vectorization to the cleaned text data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from feature_extraction import extract_features_from_csv

def main():
    """Main function to run feature extraction"""
    
    print("=== FEATURE EXTRACTION ===")
    print("Converting preprocessed text to numerical features...")
    
    # Parameters
    csv_path = "data/preprocessed_reviews.csv"
    text_column = "cleaned_review"
    label_column = "sentiment"
    max_features = 1000
    test_size = 0.2
    output_dir = "results"
    
    try:
        # Run feature extraction
        extractor, (X_train_tfidf, X_test_tfidf, X_train_count, X_test_count, y_train, y_test) = extract_features_from_csv(
            csv_path=csv_path,
            text_column=text_column,
            label_column=label_column,
            max_features=max_features,
            test_size=test_size,
            output_dir=output_dir
        )
        
        print("\n=== FEATURE EXTRACTION SUMMARY ===")
        print(f"✓ TF-IDF features extracted: {X_train_tfidf.shape[1]}")
        print(f"✓ Count features extracted: {X_train_count.shape[1]}")
        print(f"✓ Training samples: {X_train_tfidf.shape[0]}")
        print(f"✓ Test samples: {X_test_tfidf.shape[0]}")
        print(f"✓ Features and vectorizers saved to '{output_dir}/'")
        
        # Show top features
        print("\n=== TOP TF-IDF FEATURES ===")
        top_tfidf = extractor.get_feature_importance('tfidf', top_n=10)
        for idx, row in top_tfidf.iterrows():
            print(f"{row['feature']}: {row['mean_score']:.4f}")
        
        print("\n=== TOP COUNT FEATURES ===")
        top_count = extractor.get_feature_importance('count', top_n=10)
        for idx, row in top_count.iterrows():
            print(f"{row['feature']}: {row['mean_score']:.4f}")
        
        print("\n✓ Feature extraction completed successfully!")
        print("Ready for model training step.")
        
    except Exception as e:
        print(f"❌ Error during feature extraction: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
