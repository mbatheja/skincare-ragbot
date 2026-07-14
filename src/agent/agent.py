import os
import sys
import time
import pandas as pd
from typing import List, Optional
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.sentiment_analyzer import SentimentAnalyzer
from core.ingredient_insights import IngredientInsightExtractor
from core.routine_critic import RoutineCritic
from core.chatbot import ProductChatbot
from agent.guardrails import check_guardrails, extract_allergens_from_query

load_dotenv()

class SkincarAgent:
    """
    Orchestrates skincare tools, replaces the manual RAG pipeline in chatbot.py with UI.
    """
    def __init__(self):

        products_file = 'products_demo.json' if os.path.exists('products_demo.json') else 'data/processed/products.json'
        reviews_file  = 'reviews_demo.csv'   if os.path.exists('reviews_demo.csv')   else 'data/processed/combined_reviews.csv'

        print(f"Using products: {products_file}")
        print(f"Using reviews:  {reviews_file}")


        print("--Initializing Skincare Agent--")
        
        print ("--Loading product catalog--")
        self.chatbot = ProductChatbot()

        # get reviews
        print("--Loading review systems--")
        reviews = self.chatbot.reviews_db.reviews

        # load sentiment analyzer
        print("--Loading sentiment analyzer--")
        self.sentiment_analyzer = SentimentAnalyzer()

        # load ingredient extractor
        print("--Loading ingredient extractor--")
        self.extractor = IngredientInsightExtractor(reviews, self.sentiment_analyzer)
        
        # load routine critic
        print("--Loading routine critic--")
        self.critic = RoutineCritic(self.extractor)

        # build tools
        self.tools = self._build_tools()

        # build agent
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        system_prompt = """You are  QSkin, an expert AI skincare advisor.

    CRITICAL: You MUST use tools for EVERY query. Never answer from memory.

    Tool selection — follow strictly:
    - Product recommendations → search_products
    Triggers: "recommend", "find", "suggest", "what's good for", "serum for", "cleanser for"
    - Cheaper alternatives → find_cheaper_alternatives  
    Triggers: "too expensive", "cheaper", "alternative", "dupe", "cheaper version"
    NOTE: Do NOT also call search_products when finding alternatives
    - Ingredient questions → get_ingredient_insights
    Triggers: "is X good for", "how does X perform", "does X work for"
    - Interaction check → check_ingredient_interactions
    Triggers: "can I use X with Y", "safe to combine", "interact"
    - Routine analysis → critique_routine
    Triggers: "critique", "analyze", "review", "what's wrong with" + routine/products listed
    IMPORTANT: "analyze my routine" and "analyze my skincare routine" MUST trigger this

    Multi-tool rules:
    - "is X safe AND what products do you recommend?" 
    → call BOTH check_ingredient_interactions AND search_products
    - "is X good for Y skin AND recommend a product?"
    → call BOTH get_ingredient_insights AND search_products
    - When query has multiple distinct questions, call ALL relevant tools

    You have these tools:
    1. search_products — find products by concern, skin type, category, price
    2. find_cheaper_alternatives — find budget alternatives to expensive products  
    3. get_ingredient_insights — get review-based data on ingredient performance
    4. check_ingredient_interactions — check if ingredients conflict
    5. critique_routine — analyze a skincare routine for issues

    ALWAYS call at least one tool. Never skip tools.
        Be specific, educational, and evidence-based.

        Guidelines:
        - Always ask for skin type if not provided and it's relevant
        - When recommending products, prioritize reviews from users with matching skin type
        - When suggesting cheaper alternatives, always verify they have similar key actives
        - For routine critique, check wash-off actives, redundancy, and interactions
        - Be specific, educational, and evidence-based
        - Keep responses concise but informative

        SAFETY RULES:
        - Never recommend prescription products
        - Always suggest patch testing for new actives
        - Never diagnose skin conditions
        - Always mention SPF when recommending vitamin C or retinol"""

        from langgraph.prebuilt import create_react_agent

        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            prompt=system_prompt
        )
        print("The agent is online now \n")

    def _build_tools(self) -> list:
        """
        Convert existing functions into tools.
        """
        chatbot = self.chatbot
        extractor = self.extractor
        critic = self.critic
    
        @tool
        def search_products(
            query:str,
            skin_type: str=None,
            category: str=None,
            max_price: float=None) -> str:
            """Seach for skincare products. Use for any product recommendation request.
            
               Args:
                query: What the user is looking for e.g. 'moisturizer for dry skin
                skin_type: User skin type - dry, oily, combination, normal, sensitive
                category: Product type - moisturizer, serum, cleanser, 
                max_price: Maxiumum price in USD
            """

            # Retrieve candidates
            products = chatbot._retrieve_relevant_products(query, n_results=50)

            # Category filter
            
            if category:
                filtered = [
                    p for p in products
                    if category.lower() in p['category'].lower()
                    or category.lower() in p.get('subcategory', '')
                ]
                products = filtered if filtered else products

            # Pricce filter

            if max_price:
                products = [p for p in products if p['price'] <= max_price]

            if skin_type and chatbot.has_reviews:
                scored = chatbot._score_products_by_skin_type(products, skin_type)
                products = [sp['product'] for sp in scored[:5]]
            else:
                products = products[:5]

            if not products:
                return "No products found matching those criteria."
            
            result = f"Found {len(products)} products: \n\n"
            for i, p in enumerate(products, 1):
                result += f"{i}. {p['name']} by {p['brand']} \n"
                result += f" Category: {p['category']}\n"
                result += f"Price: ${p['price']:.2f}\n"
                result += f"Rating: {p['rating']}/5 ({p['reviews_count']}reviews)\n"

                if skin_type and chatbot.has_reviews:
                    stats = chatbot.reviews_db.get_product_stats(p['id'], skin_type)
                    if stats and stats['review_count'] >= 3:
                        result += f"{skin_type.capitalize()} skin rating: {stats['avg_rating']:.1f}/5"
                        result += f"({stats['review_count']} reviews, {stats['recommend_rate']:.0%} recommend)\n"

                    
                if p.get('highlights'):
                    result +=f" Features: {', '.join(p['highlights'][:3])}\n"
                    result += "\n"

            return result


        # Tool 2: Find Cheaper Alternatives
        @tool
        def find_cheaper_alternatives(
            product_name: str,
            max_price: float = None) -> str:

            """
            Find cheaper alternatives with similar ingredients to user's named product.
            Invoke when user hints at disatisfaction with existing product's price or explicitly asks for alternatives.

            Args:
                product_name: Name of the expensive product e.g. 'Paula's Choice BHA'
                max_price: Maximum price for alternatives in USD
            """

            reference = next(
                (p for p in chatbot.products
                 if product_name.lower() in p['name'].lower()),
                 None
            )

            if not reference:
                return f"'{product_name}' not found in our catalog. Try a different name. "
            
            price_ceiling = max_price or reference['price']*0.7

            dupes = critic.find_product_dupes(
                products=chatbot.products,
                product_id=reference['id'],
                max_price = price_ceiling,
                top_n=4
            )

            if not dupes:
                return f"No cheaper alternatives found for {reference['name']} (${reference['price']:.2f}) under ${price_ceiling:.2f}"
            
            result = f"Cheaper alternatives to {reference['name']} (${reference['price']:.2f}):\n\n"
            for i, dupe in enumerate(dupes, 1):
                p = dupe['product']
                result += f"{i}. {p['name']} by {p['brand']}\n"
                result += f" Price: ${p['price']:.2f}"
                if dupe['price_saving'] > 0:
                    result += f" (saves ${dupe['price_saving']:.2f})"
                    result += "\n"
                    result += f" Ingredient similarity: {dupe['similarity']:.0%}\n"
                    result += f" Matching actives: {', '.join(dupe['matching_actives'])}\n"
                    if dupe['missing_actives']:
                        result += f" Missing actives: {', '.join(dupe['missing_actives'])}\n"
                    result += "\n"
            
            return result
        
        # Tool 3: Ingredient Insights
        @tool
        def get_ingredient_insights(
            ingredient: str,
            skin_type: str = None
        ) -> str:
            """
            Get review_based performance data for a specific ingredient.
            Use when user is interested in product effectiveness.

            Args:
                ingredient: Ingredient name e.g. 'niacinamide'
                skin_type: User skin type for personalized insights
            """

            result = extractor.analyze_ingredient(ingredient)

            if not result:
                result = extractor.analyze_ingredient(ingredient)
                if not result:
                    return f"Insufficient review data for '{ingredient}' in our dataset."
                
            output = f"Ingredient: {ingredient.capitalize()}\n"
            if skin_type:
                output += f"For: {skin_type} skin\n"
            output += f"\nBased on {result['mention_count']:,} reviews:\n"
            output += f" Average rating: {result['avg_rating']:.1f}/5\n"
            output += f" Recommend rate: {result['recommend_rate']:.0%}\n"
            output += f" Positive sentiment:{result['sentiment_positive_rate']:.0%}\n"

            if result.get('sample_insights'):
                output += "/nSample reviews mentioning this ingredient:\n"
                for insight in result['sample_insights'][:3]:
                    output += f"[{insight['skin_type']}] \"{insight['text'][:150]}\"\n"
                    
            warning = extractor.get_ingredient_warnings(ingredient, skin_type)
            if warning:        
                output += "f\n{warning}\n"
            
            return output
        
        # Tool 4: Check Ingredient Interactions
        @tool
        def check_ingredient_interactions(ingredients: List[str]) -> str:
            """
            Check for harmful interactions between ingredients or products.
            To be used when user would like to combine different products in routine.

            Args:
                ingredients: List of ingredient or product names to check 
            """

            # Build dummy product dicts for the critic
            dummy_products = [
                {'name': ing, 'category': 'serum', 'ingredients': [ing]} for ing in ingredients
            ]

            conflicts = critic.check_ingredient_interactions(dummy_products)

            if not conflicts:
                return f"No known harmful interactions found between: {', '.join(ingredients)}"

            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            
            output = f"Found {len(conflicts)} interaction(s):\n\n"

            for conflict in conflicts:
                emoji = severity_emoji.get(conflict['severity'], '')
                output += f"{emoji} {conflict['ingredient_1'].capitalize()} + {conflict['ingredient_2'].capitalize()}\n"
                output += f"Severity: {conflict['severity'].upper()}\n"
                output += f"Issue: {conflict['issue']}\n"
                output += f"Recommendation: {conflict['recommendation']}\n\n"

            return output
        
        # Tool 5: Critique Routine
        @tool
        def critique_routine(
            product_names: List[str],
            skin_type: str = None
        ) -> str:
            """
            Analyze a skincare routine for mismatch between ingredient effectiveness and product format delivery.
            Also analyze cost effectiveness of routine and any harmful interaction between ingredients of various products in user's routine, recommend alternatives where possibble.

            Use when user explicitly asks for routine review/ critique.

            Args:
                product_names: List of product names in the routine
                skin_type: User skin type for personalized review insights
            """

            # Matching product names to catalog
            matched_products = []
            unmatched = []

            for name in product_names:
                match = next(
                    (p for p in chatbot.products
                     if name.lower() in p['name'].lower()),
                     None
                )
                if match:
                    matched_products.append(match)
                else:
                    unmatched.append(name)
                    matched_products.append({
                        'name': name,
                        'category': 'Unknown',
                        'price': 0.0,
                        'ingredients': []
                    })
                
            if unmatched:
                note = f"Note:{', '.join(unmatched)} not found in catalog - limited analysis available.\n\n "
            else:
                note = ""
            
            critique = critic.generate_critique(matched_products, skin_type)
            return note + critique
        
        return [
            search_products,
            find_cheaper_alternatives,
            get_ingredient_insights,
            check_ingredient_interactions,
            critique_routine
        ]
    
    def chat(self, user_message: str, chat_history: list = None) -> str:
        """
        Send message to agent with conversation history.
        """

        start = time.time()
        
        if chat_history is None:
            chat_history  = []
        
        # GUARDRAILS
        guardrail = check_guardrails(user_message)
        if guardrail.should_block:
            return guardrail.message

        allergens = extract_allergens_from_query(user_message)
        if allergens:
            user_message += f" [User is allergic to: {', '.join(allergens)}]"       
        
        lc_history = []
        for msg in chat_history:
            if msg['role'] == 'user':
                lc_history.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                lc_history.append(AIMessage(content=msg['content']))

        
        try:
            result = self.agent_executor.invoke({
                "messages": lc_history + [HumanMessage(content=user_message)]
            })

            elapsed = time.time() - start

            output = result["messages"][-1].content

            if guardrail.should_warn:
                output = guardrail.warning + output

            return output

        except Exception as e:
            print(f"Agent error: {e}")
            return (
                "I ran into an issue processing that request. "
                "Could you try rephrasing or breaking it into smaller questions?"
            )
    
if __name__ == "__main__":
    agent = SkincarAgent()

    test_queries = [
        ("I have dry skin, recommend a moisturizer under $40",[]),
        ("Paula's Choice BHA is too expensive, any alternatives?",[]),
        ("Is niacinamide good for oily skin", []),
        ("Can I use retinol with AHA?", []),
    ]

    for query, history in test_queries:
        print(f"User: {query}")
        response = agent.chat(query, history)
        print(f"\nAgent: {response}")

