import os
import sys
import pickle
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.sentiment_analyzer import SentimentAnalyzer

class IngredientInsightExtractor:
    """
    Extract insights about specific ingredients from reviews.
    """
    
    KEY_INGREDIENTS = {
            'retinol': ['retinol', 'retinyl', 'tretnoin', 'retin-a'],
            'vitamin c': ['vitamin c', 'ascorbic acid', 'l-ascorbic', 'vit c'],
            'niacinamide': ['niacinamide', 'nicotinamide'],
            'hyaluronic_acid': ['hyaluronic acid', 'hyaluronate', 'sodium hyaluronate'],
            'salicylic_acid': ['salicylic', 'salicylic acid', 'bha'],
            'glycolic_acid': ['glycolic acid', 'aha', 'glycolic'],
            'lactic_acid': ['lactic acid', 'aha', 'lactic'],
            'peptides': ['peptide', 'peptides', 'matrixyl'],
            'ceramides': ['ceramide', 'ceramides'],
            'azelaic acid': ['azelaic', 'azelaic acid'],
            'aha': ['aha', 'alpha hydoxy'],
            'benzoyl peroxide': ['benzoyl peroxide', 'bp'],
            'tea tree': ['tea tree', 'tea tree oil'],
            'centella': ['centella', 'cica', 'madecassoside'], 
            'panthenol': ['panthenol', 'pro-vitamin b5'],
        } 
    
    
    def __init__(self, reviews_df: pd.DataFrame, sentiment_analyzer: SentimentAnalyzer):
        """
        Initialize with reviews data.
        """
        self.reviews = reviews_df
        self.sentiment_analyzer = sentiment_analyzer
        self._analysis_cache = {}

        print(f" initialize review bucketing per skin type and ingredient per skin type")
        self.buckets = self._build_buckets()
        print(f"buckets built for {len(self.KEY_INGREDIENTS)} ingredients "
              f"across {self.reviews['skin_type'].nunique()} skin types")


    def _build_buckets(self) -> Dict:
        """
        Scan reviews ONCE and bucket by skin_type -> ingredient.
        """
        skin_types = list(self.reviews['skin_type'].dropna().unique()) + ['all']

        buckets = {
            skin_type: {ingredient: [] for ingredient in self.KEY_INGREDIENTS}
            for skin_type in skin_types
        }

        total = len(self.reviews)
        for idx, (_, row) in enumerate(self.reviews.iterrows()):
            if idx % 100000 == 0:
                print(f" Scanning reviews {idx:,}/{total:,}...")

            text = str(row.get('review_text', '')).lower()
            skin = str(row.get('skin_type', '')).lower().strip()

            for ingredient, keywords in self.KEY_INGREDIENTS.items():
                if any(kw in text for kw in keywords):
                    buckets['all'][ingredient].append(row)

                    if skin and skin in buckets:
                        buckets[skin][ingredient].append(row)

        return buckets

    def _ensure_buckets(self):
        """Load precomputed buckets from cache."""
        if self.buckets is not None:
            return
        
        cache_paths = [
            'data/processed/buckets_cache.pkl',  # full dataset cache
            'data/demo/buckets_cache.pkl',        # demo cache
        ]
        
        for cache_path in cache_paths:
            if os.path.exists(cache_path):
                import pickle
                print(f"Loading buckets from {cache_path}...")
                with open(cache_path, 'rb') as f:
                    self.buckets = pickle.load(f)
                print(f"Buckets loaded ({len(self.buckets)} skin types)")
                return
        
        # No cache found — build from scratch
        print("No cache found — building buckets (this takes a few minutes)...")
        self.buckets = self._build_buckets()
        
        # Save for next time
        import pickle
        cache_path = 'data/processed/buckets_cache.pkl'
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.buckets, f)
        print(f"Buckets cached to {cache_path}")


    def _extract_ingredient_sentences(self, text:str, keywords: List[str]) -> List[str]:
        """
        Extract sentences that mention the ingredient.
        """
        if pd.isna(text):
            return []
            
        sentences = re.split(r'[.!?]+', str(text).lower())
        return [
            s.strip() for s in sentences
            if any(kw in s for kw in keywords) and len(s.strip()) > 10
            ]
        
    def analyze_ingredient(self, ingredient:str, skin_type: Optional[str] = None, min_mentions: int = 5) -> Optional[Dict]:
        """
        Analyze ingredient from pre-computed buckets.
        """

        cache_key = (ingredient, skin_type)
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        self._ensure_buckets() # cached demo dataset, remove to run on full data
        if ingredient not in self.KEY_INGREDIENTS:
            return None
            
        bucket_key = skin_type.lower().strip() if skin_type else 'all'

        if bucket_key not in self.buckets:
            return None

        relevant_rows = self.buckets[bucket_key][ingredient]

        if len(relevant_rows) < min_mentions:
                return None
            
        reviews_df = pd.DataFrame(relevant_rows)
        keywords = self.KEY_INGREDIENTS[ingredient]

        # Calculate stats
        avg_rating = reviews_df['rating'].mean()
        recommend_rate = reviews_df['is_recommended'].mean() if 'is_recommended' in reviews_df.columns else 0

        # Sentiment on ingredient-specific sentences
        all_sentences = []
        sentence_sources = []

        for _, review in reviews_df.iterrows():
            sentences = self._extract_ingredient_sentences(
                review.get('review_text', ''), keywords
                )

            for sentence in sentences:
                all_sentences.append(sentence)
                sentence_sources.append(review)

            # --- one batched call instead of one-by-one ---
            if all_sentences:
                results = self.sentiment_analyzer.sentiment_pipeline(all_sentences, batch_size=32)
            else:
                results = []

            sentiments = results
            sample_insights = []
            for sentence, sentiment, review in zip(all_sentences, results, sentence_sources):
                if len(sample_insights) < 5:
                    sample_insights.append({
                        'text': sentence[:200],
                        'sentiment': sentiment['label'],
                        'confidence': float(sentiment['score']),
                        'rating': review['rating'],
                        'skin_type': review.get('skin_type', 'unknown')
                        })

        sentiment_positive_rate = (
            sum(1 for s in sentiments if s['label'] == 'POSITIVE')/ len(sentiments)
            if sentiments else 0.5
            )

        result = { 'ingredient': ingredient,
            'skin_type': skin_type,
            'mention_count': len(relevant_rows),
            'avg_rating': float(avg_rating),
            'recommend_rate': float(recommend_rate),
            'sentiment_positive_rate': float(sentiment_positive_rate),
            'sample_insights': sample_insights
        }
        
        self._analysis_cache[cache_key] = result
        return result

    def compare_ingredient_across_skin_types(self, ingredient: str) -> Dict[str, Dict]:
        """
        Compare ingredient performance across all skin types.
        """

        self._ensure_buckets() # cached demo dataset, remove to run on full data
        comparison = {}

        for skin_type in self.reviews['skin_type'].dropna().unique():
            result = self.analyze_ingredient(ingredient, skin_type=skin_type, min_mentions=3)
            if result:
                comparison[str(skin_type)] = result

        return comparison
            
    def get_ingredient_warnings(self, ingredient: str, skin_type: Optional[str] = None) -> Optional[str]:
        """
        Get warnings about an ingredient's performance across different skin types.
        """
        self._ensure_buckets() # cached demo dataset, remove to run on full data
        analysis = self.analyze_ingredient(ingredient, skin_type)
        if not analysis:
            return None
            
        warnings = []
        if analysis['avg_rating'] < 3.5:
            warnings.append(f"low avg rating ({analysis['avg_rating']:.1f}/5)")
    
        if analysis['recommend_rate'] < 0.4:
            warnings.append(f"only {analysis['recommend_rate']:.0%}")
        if analysis['sentiment_positive_rate'] < 0.5:
            warnings.append(f"mixed sentiment ({analysis['sentiment_positive_rate']:.0%}positive)")
            
        if warnings:
            skin_context = f" for {skin_type} skin" if skin_type else ""
            return f"{ingredient.capitalize()}{skin_context}: {','.join(warnings)}"
            
        return None
        
            

if __name__ == "__main__":
    print("Loding data...")
    reviews = pd.read_csv("data/processed/combined_reviews.csv")

    analyzer = SentimentAnalyzer()
    extractor = IngredientInsightExtractor(reviews, analyzer)

    #Test 1: Single ingredient all skin types
    print("Test 1: Niacinamide (all skin types)")

    result = extractor.analyze_ingredient('niacinamide')
    
    if result:
        print(f"Mentions: {result['mention_count']:,}")
        print(f"Avg Rating: {result['avg_rating']:.2f}/5")
        print(f"Recommend Rate: {result['recommend_rate']:.1%}")
        print(f"Positive Sentiment: {result['sentiment_positive_rate']:.1%}")
        
    print("\n TEST 2: Same ingredient by skin type")

    comparison = extractor.compare_ingredient_across_skin_types('niacinamide')
    for skin_type, data in comparison.items():
        print(f"\n{skin_type.capitalize()}:")
        print(f" Mentions: {data['mention_count']:}")
        print(f" Rating: {data['avg_rating']:.2f}/5")
        print(f" Recommend: {data['recommend_rate']:.1%}")

    print("\n TEST 3: Warnings (dry skin)")
    for ingredient in ['niacinamide', 'retinol', 'salicylic acid', 'hyaluronic acid']:
        warning = extractor.get_ingredient_warnings(ingredient, skin_type ='dry')
        print(f"\n{ingredient.capitalize()}:")
        if warning:
            print(f"{warning}")



