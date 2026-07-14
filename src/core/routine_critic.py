import pandas as pd
from typing import List, Dict, Optional
from core.ingredient_insights import IngredientInsightExtractor
from core.sentiment_analyzer import SentimentAnalyzer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class RoutineCritic:
    """
    Critique skincare routines using review insights.
    """

    def __init__(self, ingredient_extractor: IngredientInsightExtractor):
        """
        initialize skincare routine critic.
        """

        self.ingredient_extractor = ingredient_extractor
        
        api_key = os.environ.get("OPENAI_API_KEY")

        self.client = OpenAI(api_key = api_key)

        self.EFFICACY_RULES = {
            "salicyclic acid": {
                "effective_in": ["serum", "treatment", "toner", "mask", "exfoliant"],
                "less_effective_in": ["cleanser", "wash"],
                "reason": "Salicyclic acid is a BHA and is most effective when given time to penetrate pores. Cleanser does not allow enough time as it is washed off immediately."
            },
            "retinol": {
                "effective_in": ["serum", "cream", "treatment", "moisturizer"],
                "less_effective_in": ["cleanser", "mask"],
                "reason": "Retinol needs time to convert to retinoic acid and penetrate skin. Retinol is hyper-sensitive in light, apply only at night time.",
            },
            "vitamin c": {
                "effective_in": ["serum", "powder", "cream"],
                "less_effective_in": ["cleanser", "toner"],
                "reason": "L-ascorbic acid needs specific pH and concentration (10-20%) to be effective.",
            },
            "glycolic_acid": {
                "effective_in": ["toner", "serum", "peel"],
                "less_effective_in": ["cleanser"],
                "reason": "AHAs need time to exfoliate. Cleansers wash off too quickly."
            },
            "azelaic acid": {
                "effective_in": ["cream", "gel"],
                "less_effective_in": ["cleanser"],
                "reason": "Azelaic acid is most effective when applied as a cream or gel directly to the skin.",
            },
            "peptides": {
                "effective_in": ["serum", "cream", "moisturizer"],
                "less_effective_in": ["cleanser"],
                "reason": "Peptides require prolonged contact to support signaling pathways involved in skin repair."
            },
            "benzoyl_peroxide": {
                "effective_in": ["gel", "cream", "treatment", "leave-on"],
                "less_effective_in": ["cleanser"],
                "reason": "Leave-on formulations maintain prolonged antibacterial activity against C. acnes. Cleansers can still be useful for sensitive skin but generally provide reduced efficacy due to short contact time.",
            },
            "hyaluronic_acid": {
                "effective_in": ["serum", "moisturizer", "cream", "gel"],
                "less_effective_in": ["cleanser"],
                "reason": "As a humectant, hyaluronic acid needs to remain on the skin to bind water."
            },
            "ceramides": {
                "effective_in": ["moisturizer", "cream", "lotion", "balm"],
                "less_effective_in": ["serum", "cleanser"],
                "reason": "Ceramides repair the skin barrier and perform best in occlusive, lipid-rich formulations."
            }         
        }

        self.INGREDIENT_INTERACTIONS = {
            ('aha', 'retinol'): {
                'severity': 'high',
                'issue': 'Combined use cases over-exfoliation and severe irritation',
                'recommendation': 'Use retinol at night, AHA on alternate nights'
            },
            ('bha', 'retinol'): {
                'severity': 'high',
                'issue': 'Over-exfoliation risk, skin barrier damage',
                'recommendation': 'Use retinol at night, BHA in morning routine or alternate nights'
            },
            ('aha', 'vitamin c'): {
                'severity': 'medium',
                'issue': 'Both acidic, can destabilize vitamin C and cause irritation',
                'recommendation': 'Use vitamin C in AM, AHA in PM'
            },
            ('benzoyl peroxide', 'retinol'): {
    'severity': 'high',
    'issue': 'Benzoyl peroxide oxidizes retinol, rendering it ineffective',
    'recommendation': 'Use benzoyl peroxide AM, retinol PM on alternate nights'
            },
            ('niacinamide', 'vitamin c'): {
                'severity': 'low',
                'issue': 'May form niacin causing temporary flushing (debated in literature)',
                'recommendation': 'Alternate nights - retinol Monday/Wednesday/Friday, glycolic Tuesday/Thursday' 
            },
            ('glycolic acid', 'retinol'): {
                'severity': 'high',
                'issue': 'Double exfoliation causes irritation and comprises skin barrier',
                'recommendation': 'Alternate nights - retinol Monday/Wednesday/Friday, glycolic Tuesday/Thursday'
            }
        }

    def extract_key_actives(self, ingredients:List[str]) -> List[str]:
        """
        Extract key active ingredients.
        """
        key_actives = []

        ingredient_text = ''.join(ingredients).lower()

        for active in self.ingredient_extractor.KEY_INGREDIENTS.keys():
            keywords = self.ingredient_extractor.KEY_INGREDIENTS[active]
            if any(kw in ingredient_text for kw in keywords):
                key_actives.append(active)

        return key_actives
        
    def check_wash_off_actives(self, products: List[Dict]) -> List[Dict]:
        """
        Find active ingredients in wash-off products.
        """
        issues = []

        for product in products:
            category = product.get('category', '').lower()
            actives = self.extract_key_actives(product.get('ingredients', []))
        
        
            ineffective_actives = []
            for active in actives:
                if active in self.EFFICACY_RULES:
                    rule = self.EFFICACY_RULES[active]
                    if any(term in category for term in rule['less_effective_in']):
                        ineffective_actives.append(active)

            if ineffective_actives:
                issues.append({
                    'product': product['name'],
                    'category': product['category'],
                    'wasted_actives': ineffective_actives,
                    'reason': self.EFFICACY_RULES[ineffective_actives[0]]['reason']
                })
    
        return issues
    
    def check_ingedient_redundancy(self, products:List[Dict]) -> List[Dict]:
        """
        Find duplicate active ingredients across products.
        """
        ingredient_map = {}

        for product in products:
            actives = self.extract_key_actives(product.get('ingredients',[]))

            for active in actives:
                if active not in ingredient_map:
                    ingredient_map[active] = []
                ingredient_map[active].append(product['name'])

        redundancies = []
        for ingredient, product_names in ingredient_map.items():
            if len(product_names) > 1:
                redundancies.append({
                    'ingredient': ingredient,
                    'products': product_names,
                    'count': len(product_names)
                })
        
        return redundancies
    
    def check_ingredient_interactions(self, products: List[Dict]) -> List[Dict]:
        """
        Check for ingredient interactions with adverse affects on skin
        """

        conflicts = []
        ingredient_to_product = {}

        for product in products:
            actives = self.extract_key_actives(product.get('ingredients', []))
            for active in actives:
                if active not in ingredient_to_product:
                    ingredient_to_product[active] = []
                ingredient_to_product[active].append(product['name'])

        all_actives = list(ingredient_to_product.keys())

        for i in range(len(all_actives)):
            for j in range(i+1, len(all_actives)):
                ing1 = all_actives[i]
                ing2 = all_actives[j]

                products_with_ing1 = set(ingredient_to_product[ing1])
                products_with_ing2 = set(ingredient_to_product[ing2])

                if products_with_ing1 == products_with_ing2:
                    continue

                pair = tuple(sorted([ing1, ing2]))

                if pair in self.INGREDIENT_INTERACTIONS:
                    interaction = self.INGREDIENT_INTERACTIONS[pair]
                    conflicts.append({
                        'ingredient_1': ing1,
                        'ingredient_2': ing2,
                        'product_1': list(products_with_ing1),
                        'product_2': list(products_with_ing2),
                        'severity': interaction['severity'],
                        'issue': interaction['issue'] ,
                        'recommendation': interaction['recommendation']
                    })

        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        conflicts.sort(key=lambda x: severity_order[x['severity']])

        return conflicts

    def analyze_routine_with_reviews(self, products:List[Dict], skin_type:Optional[str]= None) -> Dict:
        """
        Analyze routine and add review-based insights.
        """

        analysis = {
          'wash_off_issues': self.check_wash_off_actives(products),
          'redundancies': self.check_ingedient_redundancy(products),
          'interactions': self.check_ingredient_interactions(products),
          'ingredient_insights': [],
          'warnings': [],  
        }

        all_actives = set()
        for product in products:
            actives = self.extract_key_actives(product.get('ingredients',[]))
            all_actives.update(actives)

        for active in list(all_actives)[:3]:
            insight = self.ingredient_extractor.analyze_ingredient(active, skin_type)
            if insight:
                analysis['ingredient_insights'].append(insight)
            
            warning = self.ingredient_extractor.get_ingredient_warnings(active, skin_type)
            if warning:
                analysis['warnings'].append(warning)
        
        return analysis
    
    def find_product_dupes(self, products: List[Dict], product_id: str = None, ingredients: List[str] = None,
                           max_price: float = None, top_n: int = 5) -> List[Dict]:
        
        """
        Find cheaper alternatives with similar ingredients
        """
        if product_id:
            reference = next((p for p in products if p['id'] == product_id ), None)
            if not reference:
                return []
            reference_actives = self.extract_key_actives(reference['ingredients'])
            reference_price = reference['price']
            reference_name = reference['name']

        elif ingredients:
            reference_actives = self.extract_key_actives(ingredients)
            reference_price = max_price or float('inf')
            reference_name = "your product"

        else:
            return []
        
        if not reference_actives:
            return []
        
        candidates = []
        
        
        for product in products:
            if product_id and product['id'] == product_id:
                continue

            if max_price and product['price'] > max_price:
                continue

            product_actives = self.extract_key_actives(product['ingredients'])

            if not product_actives:
                continue

            reference_set = set(reference_actives)
            candidate_set = set(product_actives)

            overlap = reference_set & candidate_set
            union = reference_set  | candidate_set
            similarity = len(overlap)/ len(union) if union else 0

            if len(overlap) == 0:
                continue

            price_saving = reference_price - product['price']

            candidates.append({
              'product': product,
              'similarity': similarity,
              'matching_actives': list(overlap),
              'missing_actives': list(reference_set - candidate_set),
              'extra_actives': list(candidate_set - reference_set),
              'price_saving': price_saving,
              'reference_name': reference_name  
            })

        candidates.sort(key=lambda x: (x['similarity'], x['price_saving']), reverse=True)
        return candidates[:top_n]



    def generate_critique(self, products:List[Dict], skin_type: Optional[str] = None) -> str:
        """
        Generate natural language critique using LLM.
        """

        print("Step 1: Analyzing routine...")
        analysis = self.analyze_routine_with_reviews(products, skin_type)
        print("Step 2: Analysis complete")


        print("Step 3: Building routine summary...")
        routine_summary = "User's Routine:\n"

        for i, p in enumerate(products, 1):
            actives = self.extract_key_actives(p.get('ingredients', []))
            routine_summary += f"{i}. {p['name']}({p['category']}) - ${p['price']:.2f}\n"
            if actives:
                routine_summary += f"Actives: {','.join(actives)}\n"

        print("Step 4: Building issues text...")
        # cost guardrail
        total_cost = sum(p['price'] for p in products)
        cost_context = f"\n Current routine total cost: ${total_cost:.2f}\n"
        cost_context += "IMPORTANT: Any suggested replacement products must keep the Total routine cost at or below this amount. " 
        cost_context += "If recommending an additional product that increases total cost, you MUST also suggest removing an existing product to offset it. "
        cost_context += "Never suggest a revised routine that costs more without explicitly flagging this to the user and justifying the trade-off.\n" 
        
        
        issues_text ="\n Identified Issues:"

        if analysis['wash_off_issues']:
            issues_text += "\n Inefficient Active Delivery \n"
            for issue in analysis['wash_off_issues']:
                issues_text += f"- {issue['product']}: Contains {','.join(issue['wasted_actives'])} but gets washed off\n"
                issues_text += f"Reason: {issue['reason']}\n"

        if analysis['redundancies']:
            issues_text += "\n Ingredient Redundancy: \n"
            for redundancy in analysis['redundancies']:
                issues_text += f"-{redundancy['ingredient'].capitalize()} appears in {redundancy['count']} products: {','.join(redundancy['products'])}\n"

        insights_text = "\n Review-Based Insights:\n"

        if analysis['interactions']:
            issues_text += "\n Ingredient Interactions:\n"
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for conflict in analysis['interactions']:
                emoji = severity_emoji.get(conflict['severity'], '⚠️')
                issues_text += f"{emoji} {conflict['ingredient_1'].capitalize()} + {conflict['ingredient_2'].capitalize()}\n"
                issues_text += f"   Issue: {conflict['issue']}\n"
                issues_text += f"   Fix: {conflict['recommendation']}\n"
        
        
        if analysis['warnings']:
            insights_text += "\n Warnings based on user reviews: \n"
            for warning in analysis['warnings']:
                insights_text += f"-{warning}\n"

        if analysis['ingredient_insights']:
            insights_text += "\n Ingredient Performance"
            if skin_type:
                insights_text += f"for {skin_type} skin"
            insights_text += ":/n"

            for insight in analysis['ingredient_insights']:
                insights_text += f"\n{insight['ingredient'].capitalize()}:\n"
                insights_text += f" - Based on {insight['mention_count']} reviews\n"
                insights_text += f" - Rating: {insight['avg_rating']:.1f}/5\n"
                insights_text += f" -{insight['recommend_rate']:.0%} recommend\n"
                insights_text += f" -{insight['sentiment_positive_rate']:.0%} positive sentiment\n"

        system_prompt = """

You are an expert skincare consultant with deep knowledge of ingredient efficacy and product formulation.
Provide a detailed critique of the user's skincare routine with specific and actionable recommendations.

Focus on:
1. Assessing efficacy - explain why certain ingredients work better in a specific delivery format. E.g.: cleanser vs serum if a product requires being -in.
2. Redundancy - point out ingredients that do the same thing in the routine or if ingredients are repeated across products(wastfeul duplicates)
3. Review insights - use the data from real users with same skin type
4. Routine optimization - suggest better product combinations. If possible recommend cheaper product alternatives. Mention atleast 3 alternatives for every product replacement or addition. If the optimized routine costs more, explain trade-off is worth it. If required, recommend changes in product usage timing.

Be conversational, educational and helpful and provide a concise response.
"""
        messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"{routine_summary}{issues_text}{insights_text}\n\n Provide a comprehensive critique with specific recommendations."}
             ]

        print("Step 5: Calling LLM...")
        response = self.client.chat.completions.create(
        model ="gpt-4o-mini",
        messages=messages,
        max_tokens=1000,
        temperature=0.7
            )
        print("Step 6: LLM response received")
                
        return response.choices[0].message.content
                
if __name__ == "__main__":
    print("\n ROUTINE CRITIC TEST \n")
    print("\n Loading data....")
    reviews = pd.read_csv('data/processed/combined_reviews.csv')

    sentiment_analyzer = SentimentAnalyzer()
    ingredient_extractor = IngredientInsightExtractor(reviews, sentiment_analyzer)
    critic = RoutineCritic(ingredient_extractor)

    test_routine = [
        {
            'name': 'CeraVe Salicylic Acid Cleanser',
            'category': 'Cleanser',
            'price': 18.00,
            'ingredients': ['Water', 'Salicylic Acid', 'Glycerin', 'Ceramides']
        },
        {
            'name': 'The Ordinary Niacinamide 10%',
            'category': 'Serum',
            'price': 7.00,
            'ingredients': ['Aqua', 'Niacinamide', 'Zinc PCA']
        },
        {
            'name': 'CeraVe PM Moisturizer',
            'category': 'Moisturizer',
            'price': 15.00,
            'ingredients': ['Water', 'Niacinamide', 'Ceramides', 'Hyaluronic Acid']
        }
    ]

    print("TEST ROUTINE")
    for i, p in enumerate(test_routine, 1):
        print(f"{i}.{p['name']} (${p['price']:.2f})")
            
    print("ANALYZING...")
        
    analysis = critic.analyze_routine_with_reviews(test_routine, skin_type='dry')

    print("\n Wash-off issues:", len(analysis['wash_off_issues']))
    print("Redundancies:", len(analysis['redundancies']))
    print("Warnings:", len(analysis['warnings']))


    print("\nTEST: Find Dupes from Ingredient List")
    user_ingredients = ['retinol', 'niacinamide', 'hyaluronic acid', 'ceramides']
    print(f"Finding dupes for: {user_ingredients}")

    dupes = critic.find_product_dupes(
        products=test_routine,
        ingredients=user_ingredients,
        max_price=50.0,
        top_n=3
    )

    if dupes:
        for i, dupe in enumerate(dupes, 1):
            p = dupe['product']
            print(f"\n{i}. {p['name']} (${p['price']:.2f})")
            print(f"   Similarity:       {dupe['similarity']:.0%}")
            print(f"   Matching actives: {', '.join(dupe['matching_actives'])}")
    else:
        print("No dupes found in test routine (expected - small test catalog)")    


    print("LLM CRITIQUE")
    
    critique = critic.generate_critique(test_routine, skin_type="dry")
    print(f"\n {critique}")