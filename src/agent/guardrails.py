import re
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from typing import Optional
from dataclasses import dataclass

@dataclass
class GuardrailResult:
    """
    Result of guardrail check.
    """
    should_block: bool
    should_warn: bool
    message: Optional[str] = None
    warning: Optional[str] = None

MEDICAL_CONDITIONS = [
        "eczema", "psoriasis", "rosacea", "dermatitis", "melasma",
        "open wound", "infection", "severe acne", "cystic acne",
        "allergic reaction", "burning", "rash", "inflamed",
        "bleeding", "lesion", "ulcer"
    ]

VULNERABLE_CONTEXTS = [
     (
        r'\b(pregnan\w*|expecting|with child|trimester|prenatal)\b',
            "During pregnancy, certain skincare ingredients should be avoided "
            "including retinoids and high-dose salicylic acid. Please consult "
            "your OB-GYN before using active skincare ingredients. I can still "
            "help with general questions."
        ),
        (   
        r'\b(breastfeed\w*|breast\s?feed\w*|nursing|lactat\w*)\b',
            "While breastfeeding, retinoids should be avoided. Please consult "
            "your doctor before using active skincare ingredients."
        ),
        (
        r'\b(chemo\w*|cancer treatment|radiation therapy|oncolog\w*)\b',
        "Chemotherapy can significantly affect skin sensitivity. Please "
        "consult your oncology team before introducing new skincare products."
        ),
        (
        r'\b(infant\w*|newborn\w*|toddler\w*|babay skin|child skin)\b',
        "Children's skin is more sensitive than adult skin. Please consult "
        "a pediatric dermatologist before using active skincare ingredients "
        "on children under 12."        
        )
]

OFF_TOPIC_KEYWORDS = [
        "recipe", "cooking", "politics", "stock market",
        "investment", "crypto", "relationship", "diet", "stocks",
        "weight loss", "exercise", "fitness", "mental health",
        "depression", "anxiety", "medication", "prescription",
        "vaccine", "drug dosage", "medical diagnosis"
    ]

COMMON_ALLERGENS = [
        "fragrance", "parfum", "essential oil", "lanolin",
        "propylene glycol", "formaldehyde", "parabens",
        "sulfate", "alcohol", "coconut", "nut", "latex",
        "nickel", "beeswax"
    ]

def check_medical_condition(query:str) -> Optional[str]:
        """
        Detect queries about dermatalogical conditions.
        """
        query_lower = query.lower()

        triggered = [condition for condition in MEDICAL_CONDITIONS if condition in query_lower]

        if triggered:
            return (
                f"I noticed you mentioned {', '.join(triggered)}. "
                f"For medical skin conditions, please consult a "
                f"dermatologist before trying new products — I'm not "
                f"able to give advice for active skin conditions. "
                f"I'm happy to help with general skincare questions instead."            
            )
        
        return None
    
def check_vulnerable_context(query: str) -> Optional[str]:
        """
        Detect vulnerable user contexts - pregnancy, breastfeeding etc.
        """

        query_lower = query.lower()

        for pattern, warning_message in VULNERABLE_CONTEXTS:
             if re.search(pattern, query_lower):
                  return warning_message
        return None
            
        return None
    
def check_off_topic(query: str) -> Optional[str]:
        """
        Detect queries outside skincare scope.
        """

        query_lower = query.lower()

        triggered = [kw for kw in OFF_TOPIC_KEYWORDS if kw in query_lower]

        if triggered:
            return (
                f"I'm specialized in skincare product recommendations "
                f"and ingredient analysis - I'm not able to help with "
                f"{', '.join(triggered)}."
                f"Is there a skincare question I can help with?"
            )
        return None
    
def extract_allergens_from_query (query: str) -> list:
        """
        Extract allergens user has mentioned in query.
        Used downstream in product recommendations.
        """

        query_lower = query.lower()

        return [
        allergen for allergen in COMMON_ALLERGENS
        if f"allergic to {allergen}" in query_lower
        or f"allergy to {allergen}" in query_lower
        or f"react to {allergen}" in query_lower
        or f"sensitive to {allergen}" in query_lower            
        ]
    
def check_allergen_in_product(product: dict, allergens:list) -> Optional:
        """
        Check if a recommended product contains user's stated allergens.
        """
        if not allergens:
            return None
        
        ingredients_text = ' '.join(
            product.get('ingredients', [])
        ).lower()

        found = [
            allergen for allergen in allergens
            if allergen in ingredients_text
        ]

        if found:
            return (
            f"Allergen Warning: {product['name']} contains "
            f"{', '.join(found)} which you mentioned sensitivity to. "
            f"Please check the full ingredient list before purchasing."                
            )
        return None

def check_guardrails(query: str) -> GuardrailResult:
        """
        Run all pre-flight guardrail checks on user query.

        Returns GuardrailResult with:
            should_block: True if we should not run the agent
            should_warn: True if we should prepend a warning
            message: What to return to user if blocked
            warning: What to prepend if warning

        """

        medical = check_medical_condition(query)
        if medical:
            return GuardrailResult(
                should_block=True,
                should_warn=False,
                message=medical
            )
        
        off_topic = check_off_topic(query)
        if off_topic:
            return GuardrailResult(
                should_block=True,
                should_warn=False,
                message=off_topic                
            )
        
        vulnerable = check_vulnerable_context(query)
        if vulnerable:
            return GuardrailResult(
                should_block=True,
                should_warn=False,
                warning = f" {vulnerable}\n\n With that in mind:"
            )
        
        return GuardrailResult(
            should_block=False,
            should_warn=False
        )
    
if __name__ == "__main__":

    test_queries = [
        # Should BLOCK
        ("I have severe eczema flaring up, what should I use?",   "medical"),
        ("Can you recommend a diet plan for weight loss?",         "off_topic"),
        ("What are good stocks to invest in?",                     "off_topic"),

        # Should WARN — pregnancy variations
        ("I'm pregnant, what moisturizer is safe?",                "vulnerable_pregnant"),
        ("I'm in my third trimester, what can I use?",             "vulnerable_trimester"),
        ("I'm expecting, is retinol safe?",                        "vulnerable_expecting"),
        ("I need prenatal skincare advice",                        "vulnerable_prenatal"),

        # Should WARN — breastfeeding variations
        ("I'm breastfeeding, can I use retinol?",                  "vulnerable_breastfeeding"),
        ("I'm nursing my baby, what's safe?",                      "vulnerable_nursing"),
        ("I'm lactating, can I use niacinamide?",                  "vulnerable_lactating"),

        # Should WARN — other
        ("I'm on chemotherapy, what should I avoid?",              "vulnerable_chemo"),
        ("What skincare is safe for my infant?",                   "vulnerable_infant"),

        # Should PASS
        ("Recommend a moisturizer for dry skin under $40",         "pass"),
        ("Is niacinamide good for oily skin?",                     "pass"),
        ("Can I use retinol with vitamin C?",                      "pass"),
    ]

    print("GUARDRAIL TESTS")

    for query, expected in test_queries:
        result = check_guardrails(query)

        if result.should_block:
            status = "BLOCKED"
        elif result.should_warn:
            status = "WARNED"
        else:
            status = "PASSED"

        print(f"\n{status} [{expected}]")
        print(f"Query: {query[:60]}...")

        if result.message:
            print(f"Response: {result.message[:100]}...")
        if result.warning:
            print(f"Warning: {result.warning[:100]}...")        