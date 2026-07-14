import pandas as pd
import numpy as np
import glob
import os

def load_all_reviews(data_folder: str = "../data/raw/") -> pd.DataFrame:
    """Load and combine all review CSV files."""
    
    patterns = ["reviews_*.csv"]
    
    review_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(data_folder, pattern))
        review_files.extend(files)
    
    # Remove duplicates
    review_files = list(set(review_files))
    
    # Sort by filename for consistent loading
    review_files.sort()
    
    if not review_files:
        raise FileNotFoundError(f"No review files found in {data_folder}")

    print(f"Found {len(review_files)} files:")
    for f in review_files:
        file_size = os.path.getsize(f) / (1024*1024)
        print(f"  - {os.path.basename(f)} ({file_size:.2f} MB)")

    all_reviews = []
    
    for idx, file_path in enumerate(review_files, 1):
        print(f"\nLoading file {idx}/{len(review_files)}: {os.path.basename(file_path)}")
        
        try:
            # LOAD THE CSV FILE
            df = pd.read_csv(file_path,
                             low_memory=False,
                             dtype={"author_id": str})
            
            print(f" Loaded {len(df):,} rows")
            
            # ADD TO LIST
            all_reviews.append(df)
            
        except Exception as e:
            print(f" Error loading {file_path}: {e}")
            continue
    
    if not all_reviews:
        raise ValueError("No review files loaded successfully")
    
    print("\nCombining all files ...")
    combined = pd.concat(all_reviews, ignore_index=True)
    
    # REMOVE DUPLICATES
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    
    if before > after:
        print(f" Removed {before - after:,} duplicate rows")
    
    if "Unnamed: 0" in combined.columns:
        combined = combined.drop(columns=["Unnamed: 0"])
        print("Dropped 'Unnamed: 0' column")
    
    
    print(f" Final count: {len(combined):,} reviews")
    

    return combined


df_reviews = load_all_reviews('data/raw/')
output_path = 'data/processed/combined_reviews.csv'
df_reviews.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

#EDA

## Basic checks
print(f"Total reviews: {len(df_reviews)}")
print(f"Columns: {df_reviews.columns.tolist()}")

## Data exploration
print("\n--- MISSING DATA ---")
for col in df_reviews.columns:
    missing = df_reviews[col].isna().sum()
    if missing > 0:
        pct = (missing / len(df_reviews)) * 100
        print(f"  {col:30s}: {missing:,} ({pct:.1f}%)")        

print("\n--- SKIN TYPE DISTRIBUTION ---")
skin_type_counts = df_reviews['skin_type'].value_counts(dropna=False)
print("skin_type_counts")
for skin, count in skin_type_counts.items():
    pct = (count / len(df_reviews))*100
    label = str(skin) if pd.notna(skin) else "MISSING"
    print(f"  {label:20s}: {count:,} ({pct:.1f}%)")

print("\n--- RATING DISTRIBUTION ---")
rating_counts = df_reviews['rating'].value_counts().sort_index()
for rating, count in rating_counts.items():
    pct = (count/ len(df_reviews)) * 100
    bar = "█" * int(pct / 2)
    print(f"  {rating} stars: {bar} {count:,} ({pct:.1f}%)")

print("\n--- RECOMMENDATION RATE ---")
recommend_rate = df_reviews['is_recommended'].mean()
print(f"Recommend Rate: {recommend_rate:.1%}")

print("\n--- TOP 10 PRODUCTS BY REVIEW COUNT ---")
top_product = (df_reviews.groupby(['product_id','product_name']).size().sort_values(ascending=False).head(10).reset_index(name="review_count"))
print(top_product)

print("\n--- STATS BY SKIN TYPE ---")
reviews_by_skin = df_reviews.groupby('skin_type', dropna=False).agg(
    avg_rating = ('rating', 'mean'),
    recommend_rate = ('is_recommended', 'mean'),
    review_count = ('product_id', 'count')).round(2)
reviews_by_skin.columns = ['Avg Rating', 'Recommend Rate', 'Review Count']
print(reviews_by_skin.to_string())

for _, row in top_product.iterrows():
    print(f"  {row['product_id']}: {row['review_count']:,} reviews - {str(row['product_name'])[:50]}")

print("\n--- SAMPLE REVIEW ---")
sample = df_reviews[df_reviews['review_text'].notna()].iloc[0]
print(f"Product: {sample['product_name']}")
print(f"Brand: {sample['brand_name']}")
print(f"Skin Type: {sample['skin_type']}")
print(f"Rating: {sample['rating']}/5")
print(f"Recommended: {"Yes" if sample['is_recommended'] else "No"}")
print(f"Review: {str(sample['review_text'])[:200]}...")