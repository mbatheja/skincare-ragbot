import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class ReviewsDatabase:
    """
    Handle review data and queries.
    """
    def __init__(self, reviews_file : str):
        """
        Load reviews data.
        """
        print("Loading reviews database")
        self.reviews = pd.read_csv(reviews_file, low_memory=False, dtype={"author_id": str})
        self.reviews = self._clean_data()

        print(f"Loaded {len(self.reviews):,} reviews")
        print(f"Covering {self.reviews['product_id'].nunique():,} products")
        print(f"Covering {self.reviews['author_id'].nunique():,} users")

    def _clean_data(self) -> pd.DataFrame:
        """
        Clean and prepare review data
        """
        df = self.reviews.copy()
        
        # Normalize skin type
        df["skin_type"] = df["skin_type"].str.lower().str.strip()

        # Handling missing values
        df["skin_type_known"] = df["skin_type"].notna()
            
            ## Missing values imputed with notna
        df["skin_type"] = df["skin_type"].fillna("unknown")

        # Cleaning and Fixing up the data
        df["is_recommended"] = df["is_recommended"].astype(bool)
        
        df = df[df["rating"].between(1,5)]

        return df
    
    def get_reviews_for_product(self, product_id: str, skin_type:Optional[str]=None) -> pd.DataFrame:
        """
        Get reviews for a specific product, optionally filtered by skin type
        """
        reviews = self.reviews[self.reviews["product_id"] == product_id]

        if skin_type:
            skin_type = skin_type.lower().strip()
            reviews = reviews[reviews["skin_type"] == skin_type]

        return reviews
        
    def get_product_stats(self, product_id:str, skin_type: Optional[str] = None) -> Dict:
        """
        Get aggregated stats for a product.
        """
        reviews = self.get_reviews_for_product(product_id, skin_type)

        if len(reviews) == 0:
            return None
        
        return{
            'product_id': product_id,
            'skin_type': skin_type,
            'review_count': len(reviews),
            'avg_rating': reviews['rating'].mean(),
            'recommend_rate': reviews['is_recommended'].mean(),
            'rating_distribution': reviews['rating'].value_counts().to_dict()
        }
    
    def get_skin_type_comparison(self, product_id: str) -> Dict[str, Dict]:
        """
        Compare product ratings across different skin types.
        """

        comparison = {}

        for skin_type in self.reviews['skin_type'].unique():
            stats = self.get_product_stats(product_id, skin_type)
            if stats and stats['review_count'] >= 3:
                comparison[skin_type] = stats

        return comparison
    
    def get_top_products_for_skin_type(self, skin_type:str, min_reviews: int=5, top_n: int = 10) -> List[Dict]:
        """
        Get top-rated products for a specific skin type.
        """

        skin_type = skin_type.lower().strip()
 
        skin_reviews = self.reviews[self.reviews["skin_type"] == skin_type]
 
        product_stats = skin_reviews.groupby('product_id').agg({"rating":["mean", "count"], "is_recommended":"mean"}).reset_index()
 
        product_stats.columns = ["product_id", "avg_rating", "review_count", "recommend_rate"]
 
        product_stats = product_stats[product_stats['review_count'] >= min_reviews]
  
        product_stats["score"] = (product_stats["recommend_rate"]*0.6 + (product_stats["avg_rating"]/5)*0.4)

        product_stats = product_stats.sort_values("score", ascending=False).head(top_n)

        return product_stats.to_dict("records")
    
if __name__ == "__main__":
    reviews_db = ReviewsDatabase("data/processed/combined_reviews.csv")

    print("TEST 1: Product Stats by Skin Type")

    sample_product = reviews_db.reviews['product_id'].value_counts().index[0]
    comparison = reviews_db.get_skin_type_comparison(sample_product)
    print(f"\nProduct: {sample_product}")
    for skin_type, stats in comparison.items():
        print(f"\n{skin_type.capitalize()} skin:")
        print(f" Reviews: {stats['review_count']}")
        print(f" Avg Rating: {stats['avg_rating']:.2f}/5")
        print(f" Recommend Rate: {stats['recommend_rate']:.1%}")
        
    print("TEST 2: Top Products for Dry Skin")

    top_dry = reviews_db.get_top_products_for_skin_type("dry", min_reviews=5, top_n=5)
    for i, product in enumerate(top_dry,1):
        print(f"\n{i}, Product ID: {product['product_id']}")
        print(f"Avg Rating: {product['avg_rating']:.2f}/5")
        print(f" Reviews: {product['review_count']}")
        print(f" Recommend rate: {product['recommend_rate']:.1%}")
