#!/usr/bin/env python3
"""
Main script for the Review Analysis project
This script runs the complete sentiment analysis pipeline from start to finish.

Usage:
    python main.py [--skip-data] [--skip-preprocessing] [--skip-training] [--visualize-only]

Arguments:
    --skip-data: Skip data acquisition step (assumes data/imdb_reviews.csv exists)
    --skip-preprocessing: Skip preprocessing step (assumes data/preprocessed_reviews.csv exists)
    --skip-training: Skip model training step (assumes trained models exist)
    --visualize-only: Only generate visualizations and summary
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
from data_acquisition import create_sample_imdb_dataset, save_dataset
from preprocessing import TextPreprocessor, preprocess_text
from feature_extraction import FeatureExtractor, extract_features_from_csv
from model_training import SentimentModelTrainer, train_sentiment_models

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'results', 'notebooks', 'src']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✓ Directory structure verified")

def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Review Analysis Pipeline                  ║
    ║              Sentiment Analysis on Movie Reviews             ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_step(step_num, step_name, description):
    """Print formatted step information"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {step_name.upper()}")
    print(f"{'='*60}")
    print(f"Description: {description}")
    print("-" * 60)

def run_data_acquisition():
    """Step 1: Data Acquisition and Exploration"""
    print_step(1, "Data Acquisition", "Creating/loading IMDB dataset")
    
    try:
        # Check if data already exists
        data_path = "data/imdb_reviews.csv"
        if os.path.exists(data_path):
            print(f"✓ Dataset already exists at {data_path}")
            import pandas as pd
            df = pd.read_csv(data_path)
        else:
            print("Creating sample IMDB dataset...")
            df = create_sample_imdb_dataset()
            save_dataset(df, data_path)
            print(f"✓ Dataset created and saved to {data_path}")
        
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Sentiment distribution:")
        print(df['sentiment'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"✗ Error in data acquisition: {str(e)}")
        raise

def run_preprocessing(df):
    """Step 2: Data Preprocessing"""
    print_step(2, "Data Preprocessing", "Cleaning and preparing text data")
    
    try:
        # Check if preprocessed data already exists
        preprocessed_path = "data/preprocessed_reviews.csv"
        if os.path.exists(preprocessed_path):
            print(f"✓ Preprocessed data already exists at {preprocessed_path}")
            import pandas as pd
            df_processed = pd.read_csv(preprocessed_path)
        else:
            print("Preprocessing text data...")
            
            # Initialize preprocessor
            preprocessor = TextPreprocessor()
            
            # Create a copy of the dataframe
            df_processed = df.copy()
            
            # Preprocess the reviews
            print("  - Cleaning text...")
            df_processed['cleaned_review'] = df_processed['review'].apply(preprocessor.preprocess_text)
            
            # Save preprocessed data
            df_processed.to_csv(preprocessed_path, index=False)
            print(f"✓ Data preprocessed and saved to {preprocessed_path}")
        
        print(f"✓ Processed dataset shape: {df_processed.shape}")
        
        return df_processed
        
    except Exception as e:
        print(f"✗ Error in preprocessing: {str(e)}")
        raise

def run_feature_extraction(df_processed):
    """Step 3: Feature Extraction"""
    print_step(3, "Feature Extraction", "Converting text to numerical features using TF-IDF")
    
    try:
        print("Extracting features...")
        
        # Use the existing extract_features_from_csv function
        extractor, splits = extract_features_from_csv(
            csv_path="data/preprocessed_reviews.csv",
            text_column='cleaned_review',
            label_column='sentiment'
        )
        
        # Unpack the splits (6 values: X_train_tfidf, X_test_tfidf, X_train_count, X_test_count, y_train, y_test)
        X_train_tfidf, X_test_tfidf, X_train_count, X_test_count, y_train, y_test = splits
        
        print(f"✓ Training set shape (TF-IDF): {X_train_tfidf.shape}")
        print(f"✓ Test set shape (TF-IDF): {X_test_tfidf.shape}")
        print(f"✓ Feature extraction completed")
        
        # Return TF-IDF data for model training (you could also use count data)
        return X_train_tfidf, X_test_tfidf, y_train, y_test, extractor
        
    except Exception as e:
        print(f"✗ Error in feature extraction: {str(e)}")
        raise

def run_model_training(X_train, X_test, y_train, y_test):
    """Step 4: Model Training and Evaluation"""
    print_step(4, "Model Training", "Training Logistic Regression and Naive Bayes models")
    
    try:
        print("Training models...")
        
        # Initialize the model trainer
        trainer = SentimentModelTrainer()
        
        # Set the data directly
        trainer.X_train = X_train
        trainer.X_test = X_test  
        trainer.y_train = y_train
        trainer.y_test = y_test
        
        # Train Logistic Regression
        print("  - Training Logistic Regression...")
        trainer.train_logistic_regression()
        
        # Train Naive Bayes
        print("  - Training Naive Bayes...")
        trainer.train_naive_bayes()
        
        # Save models
        trainer.save_models()
        
        # Evaluate models
        print("\n✓ Model Training Results:")
        print("-" * 40)
        
        # Evaluate Logistic Regression
        lr_results = trainer.evaluate_model('logistic_regression')
        print("Logistic Regression:")
        print(f"  Accuracy: {lr_results['test_accuracy']:.4f}")
        print(f"  Precision: {lr_results['precision']:.4f}")
        print(f"  Recall: {lr_results['recall']:.4f}")
        print(f"  F1-Score: {lr_results['f1_score']:.4f}")
        print()
        
        # Evaluate Naive Bayes
        nb_results = trainer.evaluate_model('naive_bayes')
        print("Naive Bayes:")
        print(f"  Accuracy: {nb_results['test_accuracy']:.4f}")
        print(f"  Precision: {nb_results['precision']:.4f}")
        print(f"  Recall: {nb_results['recall']:.4f}")
        print(f"  F1-Score: {nb_results['f1_score']:.4f}")
        print()
        
        model_results = {
            'Logistic Regression': lr_results,
            'Naive Bayes': nb_results
        }
        
        return model_results
        
    except Exception as e:
        print(f"✗ Error in model training: {str(e)}")
        raise

def run_visualization():
    """Step 5: Visualization and Summary"""
    print_step(5, "Visualization", "Generating word clouds and final summary")
    
    try:
        # Import visualization functions
        import pandas as pd
        import pickle
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        # Load preprocessed data
        df = pd.read_csv("data/preprocessed_reviews.csv")
        
        # Generate word clouds
        print("Generating word clouds...")
        
        # Positive reviews word cloud
        positive_text = ' '.join(df[df['sentiment'] == 1]['cleaned_review'])
        positive_wordcloud = WordCloud(width=800, height=400, 
                                     background_color='white',
                                     max_words=100).generate(positive_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(positive_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Positive Reviews', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/positive_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Negative reviews word cloud
        negative_text = ' '.join(df[df['sentiment'] == 0]['cleaned_review'])
        negative_wordcloud = WordCloud(width=800, height=400,
                                     background_color='white',
                                     max_words=100).generate(negative_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(negative_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Negative Reviews', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/negative_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Word clouds generated and saved to results/")
        
        # Create final summary
        summary = {
            'project_name': 'Review Analysis - Sentiment Classification',
            'dataset_size': len(df),
            'models_trained': ['Logistic Regression', 'Naive Bayes'],
            'features_used': 'TF-IDF Vectorization',
            'visualizations': ['Positive Word Cloud', 'Negative Word Cloud'],
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('results/final_summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        print("✓ Final summary saved to results/final_summary.pkl")
        
    except Exception as e:
        print(f"✗ Error in visualization: {str(e)}")
        raise

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description='Run the Review Analysis pipeline')
    parser.add_argument('--skip-data', action='store_true', 
                       help='Skip data acquisition step')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training step')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only generate visualizations and summary')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup directories
    setup_directories()
    
    start_time = time.time()
    
    try:
        if args.visualize_only:
            # Only run visualization
            run_visualization()
        else:
            # Run complete pipeline
            
            # Step 1: Data Acquisition
            if not args.skip_data:
                df = run_data_acquisition()
            else:
                import pandas as pd
                df = pd.read_csv("data/imdb_reviews.csv")
                print("✓ Skipped data acquisition, loaded existing data")
            
            # Step 2: Data Preprocessing
            if not args.skip_preprocessing:
                df_processed = run_preprocessing(df)
            else:
                import pandas as pd
                df_processed = pd.read_csv("data/preprocessed_reviews.csv")
                print("✓ Skipped preprocessing, loaded existing processed data")
            
            # Step 3: Feature Extraction
            X_train, X_test, y_train, y_test, vectorizers = run_feature_extraction(df_processed)
            
            # Step 4: Model Training
            if not args.skip_training:
                results = run_model_training(X_train, X_test, y_train, y_test)
            else:
                print("✓ Skipped model training, using existing models")
            
            # Step 5: Visualization
            run_visualization()
        
        # Calculate and display total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total execution time: {execution_time:.2f} seconds")
        print("\nResults saved in:")
        print("  📁 data/ - Raw and processed datasets")
        print("  📁 results/ - Models, visualizations, and summary")
        print("\nTo view results:")
        print("  🖼️  Word clouds: results/positive_wordcloud.png, results/negative_wordcloud.png")
        print("  🤖 Models: results/*.pkl files")
        print("  📊 Summary: results/final_summary.pkl")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        print("Please check the error message and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
