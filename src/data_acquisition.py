"""
Data acquisition script for IMDB movie reviews
This script downloads and prepares the IMDB dataset for sentiment analysis
"""

import pandas as pd
import numpy as np
import os

def create_sample_imdb_dataset():
    """
    Creates a sample IMDB-like dataset for sentiment analysis
    In a real project, you would download the actual IMDB dataset from Kaggle
    """
    
    # Sample positive reviews
    positive_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging throughout.",
        "Amazing cinematography and outstanding performances. A must-watch film!",
        "Brilliant storyline with excellent character development. Highly recommended!",
        "One of the best movies I've ever seen. Perfect blend of action and emotion.",
        "Incredible film with stunning visuals and a compelling narrative.",
        "Fantastic movie with great direction and amazing acting performances.",
        "Outstanding film that kept me engaged from start to finish.",
        "Excellent movie with wonderful character arcs and beautiful cinematography.",
        "Spectacular film with incredible attention to detail and superb acting.",
        "Amazing movie that exceeded all my expectations. Truly remarkable!",
        "Perfect movie with great pacing and excellent character development.",
        "Wonderful film with outstanding performances and beautiful storytelling.",
        "Incredible movie with amazing visuals and a captivating plot.",
        "Excellent film that combines great acting with a compelling story.",
        "Outstanding movie with superb direction and amazing cinematography.",
        "Fantastic film with incredible performances and a gripping narrative.",
        "Amazing movie that delivers on all fronts. Highly entertaining!",
        "Perfect film with excellent character development and stunning visuals.",
        "Wonderful movie with great acting and an engaging storyline.",
        "Incredible film that showcases brilliant filmmaking and storytelling."
    ]
    
    # Sample negative reviews  
    negative_reviews = [
        "Terrible movie with poor acting and a boring plot. Complete waste of time.",
        "Worst film I've ever seen. Awful direction and terrible performances.",
        "Boring and predictable movie with weak characters and poor dialogue.",
        "Disappointing film with lackluster acting and a confusing storyline.",
        "Poor movie with bad editing and unconvincing performances.",
        "Awful film with terrible pacing and weak character development.",
        "Bad movie with poor direction and uninspiring performances.",
        "Terrible film with boring dialogue and uninteresting characters.",
        "Disappointing movie with weak plot and mediocre acting.",
        "Poor film with bad cinematography and unconvincing story.",
        "Awful movie with terrible writing and lackluster performances.",
        "Bad film with poor character development and boring plot.",
        "Terrible movie with weak direction and unengaging storyline.",
        "Disappointing film with poor acting and confusing narrative.",
        "Awful movie with bad pacing and unconvincing characters.",
        "Poor film with terrible dialogue and weak performances.",
        "Bad movie with boring plot and mediocre direction.",
        "Terrible film with poor editing and uninteresting story.",
        "Disappointing movie with weak acting and confusing plot.",
        "Awful film with bad character development and boring dialogue."
    ]
    
    # Create DataFrame
    reviews = positive_reviews + negative_reviews
    sentiments = [1] * len(positive_reviews) + [0] * len(negative_reviews)  # 1=positive, 0=negative
    
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def save_dataset(df, filepath):
    """Save dataset to CSV file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample IMDB dataset...")
    df = create_sample_imdb_dataset()
    
    # Save to data directory
    filepath = "data/imdb_reviews.csv"
    save_dataset(df, filepath)
    
    print("\nDataset creation completed!")
    print("Note: In a real project, download the actual IMDB dataset from Kaggle")
