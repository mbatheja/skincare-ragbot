import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from ingestion.reviews_db import ReviewsDatabase
from typing import List, Dict, Any, Optional

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRODUCTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'demo', 'products.json')
REVIEWS_FILE  = os.path.join(PROJECT_ROOT, 'data', 'demo', 'combined_reviews.csv')

class ProductChatbot:
    def __init__(self, products_file: str = None, reviews_file: str = None):
        """Initialize the chatbot."""

        products_file = products_file or PRODUCTS_FILE
        reviews_file  = reviews_file  or REVIEWS_FILE
        
        print(f"Products: {products_file}")
        print(f"Reviews:  {reviews_file}")

        print("Initializing chatbot...")
        # Check API Key
        try:
            import streamlit as st
            self.api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        except:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=self.api_key)
        
        try:
            self.reviews_db = ReviewsDatabase(reviews_file)
            self.has_reviews = True

        except Exception as e:
            print(f"Reviews not available: {e}")
            self.reviews_db = None
            self.has_reviews = False

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        self.chroma_client = chromadb.EphemeralClient()
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="beauty_products",
            embedding_function=self.embedding_function
        )
        print("Vector database initialized")
        
        # Load products
        self.products = self._load_products(products_file)
        print(f"Loaded {len(self.products)} products")

        ## Force reindex if collection size doesn't match
        if self.collection.count() != len(self.products):
            print(f"Collection mismatch: {self.collection.count()} indexed vs {len(self.products)} products")
            print("Clearing and reindexing...")
            try:
                self.chroma_client.delete_collection("beauty_products")
            except Exception:
                pass
            self.collection = self.chroma_client.get_or_create_collection(
                name="beauty_products",
                embedding_function=self.embedding_function
            )
        self._index_products()        
    
        print("Chatbot active")

    def _load_products(self, filepath: str) -> List[Dict]:
        """Load products from JSON file."""
        with open(filepath, 'r') as f:
            products = json.load(f)
        print(f"{len(products)} products loaded")
        return products
        
    def _create_product_text(self, product: Dict) -> str:
        """Create searchable text with complete ingredient list. """ 
        parts = [
            f"Product: {product['name']}",
            f"Brand: {product['brand']}",
            f"Category: {product['category']} - {product['subcategory']}",
            f"Price: ${product['price']:.2f}",
        ]

        # Sale info
        if product.get('on_sale'):
            parts.append(f"On Sale: ${product['sale_price']:.2f}")

        # Highlights
        if product.get('highlights'):
            highlights_text = ','.join(product['highlights'])
            parts.append(f"Features: {highlights_text}")

        # Ingredient search
        if product.get('ingredients'):
            all_ingredients = '|'.join(product['ingredients'])
            parts.append(f"Complete Ingredients List: {all_ingredients}")

        # Social proof
        if product['rating'] > 0:
            parts.append(f"Rating: {product['rating']}/5, ({product['reviews_count']} reviews, {product['loves_count']} loves)")

        # Flags
        flags = []
        if product.get('new'):
            flags.append('New')
        if product.get('limited_edition'):
            flags.append('Limited Edition')
        if product.get('sephora_exclusive'):
            flags.append('Sephora Exclusive')
        if flags:
            parts.append(f"Special: {', '.join(flags)}")
        
        if product.get('size'):
            parts.append(f"Size: {product['size']}")
        
        return "\n".join(parts)

    def _index_products(self):
        """ Index products in vector database."""
        if self.collection.count() > 0:
            print(f" Collection already has {self.collection.count()} items, skipping indexing.")
            return

        print("Indexing products...")

        documents = []
        ids = []
        metadatas = []
        seen_ids = set()

        for product in self.products:
            product_id = product['id']
            if product_id in seen_ids:
                continue

            seen_ids.add(product_id)
            # create searchable text
            
            doc_text = self._create_product_text(product)
            documents.append(doc_text)

            # ID
            ids.append(product['id'])

            # Metadata
            metadatas.append({
                "name": product['name'],
                "price": product['price'],
                "category": product['category']
            })
        
        print(f" {len(self.products) - len(ids)} duplicate IDs removed.")

        
        batch_size = 4000
        total = len(documents)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            print(f" Adding batch {i}-{end} of {total}...")
            self.collection.add(
                documents = documents[i:end],
                ids=ids[i:end],
                metadatas=metadatas[i:end]
            )

        print(f"{total} products indexed. ")

    def _retrieve_relevant_products(self, query: str, n_results: int = 3):
        """ Retrieve relevant products using vector search."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        retrieved_ids = results['ids'][0]
        
        # Get full product details
        relevant_products = [
            p for p in self.products if p['id'] in retrieved_ids
        ]
        print(f' {len(relevant_products)} products found for query: {query}')
        return relevant_products

    def _score_products_by_skin_type(self, products: List[Dict], skin_type: Optional[str]) -> List[Dict]:
        """
        Score and rank products based on reviews from users with same skin type.
        """
        if not self.has_reviews or not skin_type:
            return products
        
        scored_products = []

        for product in products:
            # get base score
            base_score = product.get("rating",0)/5.0
            # get skin type specific score from reviews
            stats = self.reviews_db.get_product_stats(product["id"], skin_type)

            if stats and stats["review_count"] >= 3:
                personalized_score = (
                    stats["avg_rating"]/5.0*0.5 +
                    stats["recommend_rate"]*0.5
                )
                confidence = "high"
                review_count = stats["review_count"]
            
            elif stats and stats["review_count"] > 0:
                personalized_score = (
                    stats["avg_rating"]/ 5.0 * 0.7 +
                    base_score*0.3
                )
                confidence = "medium"
                review_count = stats["review_count"]
            else:
                personalized_score = base_score
                confidence = "low"
                review_count = 0

            scored_products.append({
                "product":product,
                "score": personalized_score,
                "confidence": confidence,
                "skin_type_reviews": review_count,
                "skin_type_stats": stats
            })

        # Sort by score
        scored_products.sort(key=lambda x:(x["confidence"] != "low", x["score"]), reverse=True)
            
        return scored_products

    
    def chat(self, user_message: str, conversation_history: List[Dict] = None, skin_type: Optional[str] = None) -> str:
        """Process user message with skin type personalization and return response."""
        if conversation_history is None:
            conversation_history = []
        
        # Retrieve relevant products
        products = self._retrieve_relevant_products(user_message, n_results=15)

        # Score by skin type
        if skin_type:
            scored_products = self._score_products_by_skin_type(products, skin_type)
            products = [sp["product"] for sp in scored_products[:5]]
            personalization_note = f"\n[Personalized for {skin_type} skin based on reviews from similiar users]\n"
        else:
            products = products[:5]
            personalization_note = ""

        # Build context from retrieved products
        context = "Here are the relevant products:" + personalization_note + "\n"
        for i, p in enumerate(products, 1):
            context += f"{i}. {p['name']} by {p['brand']}\n"
            context += f" Category: {p['category']} - {p['subcategory']}\n"
            context += f"   Price: ${p['price']:.2f}\n"

            if p.get('on_sale'):
                context += f"(SALE: ${p['sale_price']:.2f})"
            context += "\n"

            if p['rating'] >0:
                context += f" Overall: {p['rating']}/5, ({p['reviews_count']} reviews)\n"

            if skin_type and self.has_reviews:
                stats = self.reviews_db.get_product_stats(p['id'], skin_type)
                if stats and stats['review_count'] >=3:
                    context += f"For {skin_type} skin: {stats['avg_rating']:.1f}/5, ({stats['review_count']} reviews, {stats['recommend_rate']:.0%} recommend)\n"

            if p.get("highlights"):
                context += f"Features: {','.join(p['highlights'])}\n"
            context += "\n"

        
        # Create system prompt
        system_prompt = """You are a helpful beauty product advisor.
        When skin type information is provided, pay special attention to the skin-type-specific ratings.
        These show how people with the same skin type rated each product.

        CRITICAL RULES:
        1. Prioritize products with high ratings from users with matching skin type
        2. Mention skin-type-specific ratings when relevant
        3. Only recommend products that match the requested product category
        4. Be specific about why each product is suitable
        5. Keep recommendations concise and helpful 
        """

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": f"{context}\n\nCustomer question: {user_message}"})

        # Call OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    bot = ProductChatbot()
    
    print("Test 1")
    query = "I have dry skin and want a non-comodogenic moisturizer under $50. What do you recommend?"
    retrieved = bot._retrieve_relevant_products(query, n_results=3)
    
    print(f"\nRetrieved products for query: '{query}'")
    for p in retrieved:
        print(f"- {p['name']} (${p['price']:.2f})")
    
    # Test chat
    response = bot.chat(query)
    print("\nChatbot response:")
    print(response)
    
    # Test product text creation
    print("Sample Product Text: \n")

    sample_text = bot._create_product_text(bot.products[0])
    print(sample_text)

