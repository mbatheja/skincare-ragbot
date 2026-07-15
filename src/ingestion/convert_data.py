import pandas as pd
import json
import ast 
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load dataset
df = pd.read_csv('../data/raw/product_info.csv')

print(f"Total products in dataset: {len(df)}")

# Data cleaning
df_1 = df.dropna(subset=['product_id', 'product_name', 'brand_name', 'price_usd'])
df_1 = df_1.drop_duplicates(subset=['product_id'])
df_1 = df_1[df_1['out_of_stock'] == 0]

print(f"After filtering: {len(df_1)} products")

print(f"Processing {len(df_1)} products...\n")

# Prep for JSON
products = []

for index, row in df_1.iterrows():
    
    # Parse highlights
    highlights = []
    if pd.notna(row['highlights']):
        try:
            highlights = ast.literal_eval(row['highlights'])
        except:
            highlights = []
    
    # Parse ingredients - KEEP ALL (not just 3!)
    ingredients_list = []
    if pd.notna(row['ingredients']):
        try:
            ingredients_list = ast.literal_eval(row['ingredients'])
        except:
            ingredients_list = []

    product = {
        # Product info
        "id": row['product_id'],
        "name": row['product_name'],
        "category": row['primary_category'] if pd.notna(row['primary_category']) else 'Beauty',
        "subcategory": row['secondary_category'] if pd.notna(row['secondary_category']) else '',
        "product_type": row['tertiary_category'] if pd.notna(row['tertiary_category']) else '',
        "brand": row['brand_name'],
        
        # Pricing
        "price": float(row['price_usd']),
        "on_sale": pd.notna(row['sale_price_usd']),
        "sale_price": float(row['sale_price_usd']) if pd.notna(row['sale_price_usd']) else None,
        
        # Engagement data
        "rating": float(row['rating']) if pd.notna(row['rating']) else 0,
        "reviews_count": int(row['reviews']) if pd.notna(row['reviews']) else 0,
        "loves_count": int(row['loves_count']) if pd.notna(row['loves_count']) else 0,  # ✅ FIXED: was 'loves'
        
        # Product attributes
        "highlights": highlights,
        "ingredients": ingredients_list,  # ✅ ALL ingredients
        "ingredient_count": len(ingredients_list),
        "size": row['size'] if pd.notna(row['size']) else 'Standard',
        
        # Flags
        "limited_edition": bool(row['limited_edition']),
        "new": bool(row['new']),
        "online_only": bool(row['online_only']),
        "sephora_exclusive": bool(row['sephora_exclusive']),
        
        # Variants
        "has_variants": int(row['child_count']) > 0 if pd.notna(row['child_count']) else False,
        "variant_price_range": {
            "min": float(row['child_min_price']) if pd.notna(row['child_min_price']) else None,
            "max": float(row['child_max_price']) if pd.notna(row['child_max_price']) else None
        } if pd.notna(row['child_count']) and row['child_count'] > 0 else None
    }
    
    products.append(product)

# Save to JSON
with open('../data/processed/products.json', 'w') as f:
    json.dump(products, f, indent=2)

print(f" Converted {len(products)} products to data/processed/products.json\n")

# Sample product
print("Sample product:")
sample = products[0]
print(f"  Name: {sample['name']}")
print(f"  Brand: {sample['brand']}")
print(f"  Price: ${sample['price']:.2f}")
print(f"  Rating: {sample['rating']}/5")
print(f"  Ingredients: {sample['ingredient_count']} total")

# Statistics
print("Dataset Statistics:")
print(f"  Average price: ${sum(p['price'] for p in products)/len(products):.2f}")
print(f"  Products on sale: {sum(1 for p in products if p['on_sale'])}")
print(f"  New products: {sum(1 for p in products if p['new'])}")
print(f"  Sephora exclusives: {sum(1 for p in products if p['sephora_exclusive'])}")

print("\nIngredient Statistics:")
products_with_ingredients = [p for p in products if p['ingredients']]
if products_with_ingredients:
    avg_ingredients = sum(p['ingredient_count'] for p in products_with_ingredients) / len(products_with_ingredients)
    print(f"  Products with ingredients: {len(products_with_ingredients)}/{len(products)}")
    print(f"  Average ingredients per product: {avg_ingredients:.1f}")
    print(f"  Max ingredients in a product: {max(p['ingredient_count'] for p in products_with_ingredients)}")