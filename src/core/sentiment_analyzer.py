import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

class SentimentAnalyzer:
    """
    Analyze sentiment in product reviews.
    """

    def __init__(self):
        """
        Initialize sentiment model.
        """
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )

        print("sentiment loaded")

    def analyze_text(self, text:str) -> Dict:
        """
        Analyze sentiment of a single text.
        """
        if not text or pd.isna(text):
            return {'label': 'NEUTRAL', 'score':0.5}
        
        #Truncate to model max length (512 tokens)
        text = str(text)[:500]

        try:
            result = self.sentiment_pipeline(text)[0]
            return result

        except Exception as e:
            print(f"Error analyzing text: {e}")
            return{"label":"NEUTRAL", "score": 0.5}
        
    def analyze_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment analysis to reviews dataframe.
        """
        print(f"Analyzing {len(reviews_df)} reviews...")

        sentiments = []

        for idx, row in reviews_df.iterrows():
            sentiment = self.analyze_text(row['review_text'])
            sentiments.append(sentiment)

            if (len(sentiments)) % 100 == 0:
                print(f"Processed {len(sentiments)}/{len(reviews_df)}")
        
        reviews_df = reviews_df.copy()
        reviews_df['sentiment_label'] = [s['label'] for s in sentiments]
        reviews_df['sentiment_score'] = [s['score'] for s in sentiments]

        print("sentiment analysis complete")
        return reviews_df
    
    def get_sentiment_summary(self, reviews_df: pd.DataFrame, 
                              product_id: str = None, skin_type: str = None) -> Dict:
        """
        Get overall sentiment statistics.
        """

        if 'sentiment_label' not in reviews_df.columns:
            return None
        
        df = reviews_df.copy()

        if product_id:
            df = df[df['product_id'] == product_id]
        
        if skin_type:
            df = df[df['skin_type'] == skin_type]

        if len(df) == 0:
            return None
        
        positive = (df['sentiment_label'] == 'POSITIVE').sum()
        negative = (df['sentiment_label'] == 'NEGATIVE').sum()

        return {
            'product_id': product_id,
            'skin_type': skin_type,
            'review_count': len(df),
            'positive_rate': positive/ len(df),
            'negative_rate': negative/ len(df),
            'avg_sentiment_score': df['sentiment_score'].mean()
        }
    

if __name__ == "__main__":

    reviews = pd.read_csv('data/processed/combined_reviews.csv').head(1000)

    analyzer = SentimentAnalyzer()

    test_texts = [
        "This product is amazing! My skin has never looked better.",
        "Terrible product. Broke me out immediately.",
        "It's okay, nothing special but does the job."
    ]

    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (confidence: {result['score']:.2f})")


    analyzed = analyzer.analyze_reviews(reviews)
    print("Product + skin type:")
    print(analyzer.get_sentiment_summary(analyzed, product_id="P420652", skin_type="dry"))

    print("\nProduct only:")
    print(analyzer.get_sentiment_summary(analyzed, product_id='P420652'))

    print("\n Skin Type only:")
    print(analyzer.get_sentiment_summary(analyzed, skin_type="dry"))                                         
