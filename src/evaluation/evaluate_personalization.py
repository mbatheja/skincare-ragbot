import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from core.chatbot import ProductChatbot
from ingestion.reviews_db import ReviewsDatabase

from typing import Dict, List
import json
from scipy import stats

class PersonalizationEvaluator:
    """
    Evaluate the impact of skin type personalization.
    """

    def __init__(self, chatbot:ProductChatbot, reviews_db: ReviewsDatabase):
        self.chatbot = chatbot
        self.reviews_db = reviews_db

    def create_test_set(self, n_samples: int=2000) -> pd.DataFrame:
        """
        Create test set from reviews.
        """

        print(f"Creating test set with {n_samples}samples...")
        reviews = self.reviews_db.reviews.dropna(subset=['skin_type'])

        product_ids = [p['id'] for p in self.chatbot.products]
        reviews = reviews[reviews['product_id'].isin(product_ids)]

        test_set = reviews.sample(min(n_samples, len(reviews)), random_state=42)

        print(f"Created test set with {len(test_set)} reviews")
        print(f"  Skin types: {test_set['skin_type'].value_counts().to_dict()}")
        
        return test_set
    
    def evaluate_baseline(self, test_set: pd.DataFrame) -> Dict:
        """
        Evaluate recommendations without personalization.
        """
        results = []

        for idx, row in test_set.iterrows():
            skin_type = row['skin_type']
            actual_liked = row['is_recommended']
            actual_product_id = row['product_id']

            product = next((p for p in self.chatbot.products if p['id'] == actual_product_id), None)
            if product is None:
                continue

            category = product.get('subcategory') or product.get('category', 'product')

            query = f"{category} for {skin_type} skin"

            try:
                retrieved = self.chatbot._retrieve_relevant_products(query, n_results=10)
                retrieved_ids = [p['id'] for p in retrieved]
                
                found = actual_product_id in retrieved_ids
                rank = retrieved_ids.index(actual_product_id) + 1 if found else 0

                results.append({
                    'found': found,
                    'rank': rank if found else 11,
                    'actual_liked': actual_liked
                })

            except Exception as e:
                print(f"Error: {e}")
                continue

        precision_at_5 = sum(1 for r in results if r["rank"] <= 5 and r["actual_liked"])/len(results)
        precision_at_10 = sum(1 for r in results if r['rank']<=10 and r['actual_liked'])/len(results)

        print(f"Baseline Precision@5: {precision_at_5:.2%}")
        print(f"Baseline Precision@10: {precision_at_10:.2%}")

        return{
            'precision@5': precision_at_5,
            'precision@10': precision_at_10,
            'results': results
        }
    
    def evaluate_personalized(self, test_set: pd.DataFrame) -> Dict:
        """
        Evaluate recommendations with skin type personalization.
        """
        results = []

        for idx, row in test_set.iterrows():
            skin_type = row['skin_type']
            actual_liked = row['is_recommended']
            actual_product_id = row['product_id']

            product = next((p for p in self.chatbot.products if p['id'] == actual_product_id ), None)
            if product is None:
                continue

            category = product.get('subcategory') or product.get('catgeory', 'product' )
            
            query = f"{category} for {skin_type} skin"

            try:
                #retrieve more candidates
                retrieved = self.chatbot._retrieve_relevant_products(query, n_results=15)
                
                scored = self.chatbot._score_products_by_skin_type(retrieved, skin_type)

                top_product = [sp['product'] for sp in scored[:10]]
                retrieved_ids = [p['id'] for p in top_product]

                found = actual_product_id in retrieved_ids

                rank = retrieved_ids.index(actual_product_id) + 1 if found else 11

                results.append({
                    'found': found,
                    'rank': rank,
                    'actual_liked': actual_liked
                })
            
            except Exception as e:
                print(f"Error: {e}")
                continue

        precision_at_5 = sum(1 for r in results if r['rank'] <= 5 and r['actual_liked']) / len(results)
        precision_at_10 = sum(1 for r in results if r['rank'] <= 10 and r['actual_liked']) / len(results)

        print(f"  Personalized Precision@5:  {precision_at_5:.2%}")
        print(f"  Personalized Precision@10: {precision_at_10:.2%}")

        return {
            'precision@5': precision_at_5,
            'precision@10': precision_at_10,
            'results': results
        }


    
    def run_full_evaluation(self, n_samples: int = 2000):
        """
        Run complete evaluation and show results.
        """
        print("Personalization Evaluation")
        
        #create test ste
        test_set = self.create_test_set(n_samples)

        #evaluate both approaches
        baseline = self.evaluate_baseline(test_set)
        personalized = self.evaluate_personalized(test_set)

        #calculate improvement
        if baseline['precision@5'] >0:
            improvement_p5 = (personalized['precision@5'] - baseline['precision@5'])/ baseline['precision@5']*100
        else:
            improvement_p5 = float('inf') if personalized['precision@5'] > 0 else 0.0

        if baseline['precision@10'] >0:
            improvement_p10 = (personalized['precision@10'] - baseline['precision@10'])/ baseline['precision@10']*100
        else:
            improvement_p10 = float('inf') if personalized['precision@10'] > 0 else 0.0


        print("FINAL RESULTS")
        print(f"Test Set Size: {len(test_set)} reviews")
        print(f"Precision@5")
        print(f"Baseline: {baseline['precision@5']:.2%}")
        print(f"Personalized: {personalized['precision@5']:.2%}")
        print(f"Improvement: {improvement_p5:+.1f}%")

        #statistical significance : t-test
        baseline_scores = [1 if r['rank'] <= 5 and r['actual_liked'] else 0 for r in baseline['results']]
        personalized_scores = [1 if r['rank'] <=5 and r['actual_liked'] else 0 for r in personalized['results']]

        t_stat, p_value = stats.ttest_rel(personalized_scores, baseline_scores)

        print('Statistical Significance')
        print(f" t-statistic: {t_stat:.3f}")
        print(f" p-value: {p_value:.4f}")
        print(f"Significant: {'Significant' if p_value<0.05 else 'Not Significant'} (at alpha = 0.05)")

        return {
            'baseline': baseline,
            'personalized': personalized,
            'improvement': {
                'precision@5': improvement_p5,
                'precision@10': improvement_p10
            },
            'p_value': p_value
        }
    
if __name__ == "__main__":
    print("Initializing systems")
    chatbot = ProductChatbot()
    reviews_db = ReviewsDatabase('data/processed/combined_reviews.csv')

    evaluator = PersonalizationEvaluator(chatbot, reviews_db)
    results = evaluator.run_full_evaluation(n_samples=2000)

    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'baseline_precision@5': results['baseline']['precision@5'],
            'personalized_precision@5': results['personalized']['precision@5'],
            'improvement_percentage': results['improvement']['precision@5'],
            'p_value': results['p_value']
        }, f, indent=2)

    print(" Results saved to evaluation_results.json")