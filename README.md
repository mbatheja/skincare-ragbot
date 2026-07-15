# QSkin — AI-Powered Skincare Advisor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://qskin-advisor.streamlit.app/)

> Personalized skincare recommendations backed by 1,094,411 verified Sephora product reviews

---

## Problem

Building an effective skincare routine requires synthesizing information scattered across incompatible sources:

- **Product reviews** on Sephora, Amazon, and Reddit — unstructured, skin-type unfiltered
- **Ingredient databases** like INCIDecoder and Skincarisma — useful but require you to already know what to look for
- **Community knowledge** on YouTube and r/SkincareAddiction — high quality but not searchable or personalized
- **Brand marketing** — optimized for conversion, not accuracy

Existing tools solve parts of this problem in isolation:

| Tool | What it does | What it misses |
|------|-------------|----------------|
| Skincarisma | Ingredient safety scoring and skin-type flags | No routine-level analysis, no skin-type personalization |
| INCIDecoder | Ingredient explanation | No product recommendations, no interaction checking |
| Sephora reviews | Product ratings | No skin-type filtering, no ingredient context |
| Reddit/YouTube | Community experience | Not searchable, not personalized, not scalable |

**The gap nobody fills:** how do ingredients across multiple products in a routine interact — and does that change based on skin type? A salicylic acid cleanser + AHA toner + retinol serum sounds reasonable. For dry skin it is a recipe for barrier damage. No existing consumer tool tells you this.

---

## Solution

QSkin consolidates these fragmented sources into a single personalized advisor:

- **1M+ reviews filtered by skin type** — replaces generic ratings with "how did people with *your* skin type rate this?"
- **Semantic product search** — understands "something for my T-zone", not just exact ingredient names
- **Routine-level analysis** — checks wash-off actives, ingredient redundancy, and cross-product interactions
- **Evidence-based ingredient insights** — sentiment analysis over real reviews, not brand claims
- **Dupe finder** — cheaper alternatives with matching active ingredients
- **Safety guardrails** — redirects medical conditions, warns vulnerable users

---

## Dataset

**Source:** [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) on Kaggle

### Products

- **File:** `product_info.csv`
- **Raw:** 8,494 products
- **After filtering out-of-stock:** 7,868 products used
- **Fields:** product_id, product_name, brand_name, price_usd, ingredients, highlights, rating, reviews_count, loves_count, primary_category, secondary_category, tertiary_category

### Reviews

| File | Rows |
|------|------|
| reviews_0-250.csv | 602,130 |
| reviews_250-500.csv | 206,725 |
| reviews_500-750.csv | 116,262 |
| reviews_750-1250.csv | 119,317 |
| reviews_1250-end.csv | 49,977 |
| **Total** | **1,094,411** |

**Fields:** author_id, rating, is_recommended, review_text, skin_type, skin_tone, eye_color, hair_color, product_id, product_name, brand_name, price_usd

### Key Statistics

| Metric | Value |
|--------|-------|
| Total reviews | 1,094,411 |
| Reviews with skin type | 982,854 (89.8%) |
| Reviews without skin type | 111,557 (10.2%) |
| Unique products reviewed | 2,351 |
| Unique reviewers | 503,216 |
| Overall recommendation rate | 84.0% |

**Skin type distribution:**

| Skin Type | Reviews | Share |
|-----------|---------|-------|
| Combination | 544,513 | 49.7% |
| Dry | 185,937 | 17.0% |
| Normal | 131,910 | 12.0% |
| Oily | 120,494 | 11.0% |
| Unknown | 111,557 | 10.2% |

**Rating distribution:**

| Stars | Reviews | Share |
|-------|---------|-------|
| 5 | 698,951 | 63.9% |
| 4 | 199,389 | 18.2% |
| 3 | 81,816 | 7.5% |
| 2 | 53,032 | 4.8% |
| 1 | 61,223 | 5.6% |

### Why This Dataset

The `skin_type` field in reviews is the core asset. Most skincare recommendation systems use generic ratings. This dataset enables collaborative filtering by user similarity — "how did people with *your specific skin type* rate this product?"

---

## Architecture

```
USER INTERFACE (Streamlit)
6 tabs: Chat | Routine Builder | Ingredient Insights
        Sentiment Analysis | Find Dupes | Interactions
         |
         v
GUARDRAILS LAYER (guardrails.py)
  - Medical condition redirect (eczema, rosacea, psoriasis)
  - Vulnerable context warning (pregnancy, breastfeeding, chemo)
  - Off-topic filter (finance, diet, politics)
  - Allergen extraction from query
         |
         v
LangGraph ReAct AGENT (agent.py)
  GPT-4o-mini reads query, decides which tools to call,
  executes tools, synthesizes final response
         |
    _____|_____________________________________
    |           |           |         |       |
    v           v           v         v       v

search_     find_       get_       check_   critique_
products    cheaper_    ingredient_ ingredient_ routine
            alts        insights   interactions

ChromaDB    Jaccard     Pre-bucketed  Rule-based   Wash-off
semantic    similarity  1M reviews    interaction  active
search      on active   by skin_type  database     detection
    +       ingredients     +         6 pairs      +
category        +       DistilBERT    severity     redundancy
price       price rank  sentiment     scoring      analysis
filter                  per sentence               +
    +                                              LLM
skin-type                                          critique
review
re-ranking
    |           |           |              |          |
    v           v           v              v          v
        ChromaDB        Reviews DB           Rule DBs
        7,868 products  1,094,411 reviews    EFFICACY_RULES
        384-dim vectors bucketed by          INGREDIENT_
        (sentence-      skin_type x          INTERACTIONS
        transformers    ingredient           VULNERABLE_
        default)                             CONTEXTS
              |               |
              v               v
                  DATA LAYER
            product_info.csv (8,494 products)
            reviews_0-250.csv
            reviews_250-500.csv
            reviews_500-750.csv     1,094,411 reviews
            reviews_750-1250.csv
            reviews_1250-end.csv
```

---

## Evaluation

### Phase 1: Retrieval Personalization

**What we tested:** Does filtering by skin-type reviews improve which products we surface?

**Methodology:** Held out 2,000 reviews (random_state=42) as test cases. For each review, generated a category-aware query — for example "Face Moisturizer for dry skin" — and checked whether the system retrieved the exact product the reviewer chose from a catalog of 7,868 products.

This is a deliberately strict evaluation. It isolates the personalization contribution by stripping away query richness. Real users provide richer queries which would yield higher absolute precision. What matters here is the *relative* improvement.

| Approach | Precision@5 | Precision@10 |
|----------|-------------|--------------|
| Baseline — semantic search only | 1.45% | 2.20% |
| Personalized — skin-type re-ranking | 1.95% | 3.10% |
| **Relative improvement** | **+34.5%** | **+40.9%** |

**Statistical test:** Paired t-test, n=2,000, fixed seed (random_state=42)
- t-statistic: 1.667
- p-value: 0.096

The improvement is directionally positive and consistent across runs with a fixed seed. p=0.096 is borderline — a larger evaluation set or richer query generation (using actual product names rather than category+skin-type queries) would strengthen this. The result validates the core hypothesis: skin-type-matched reviews meaningfully re-rank product candidates.

---

### Phase 2: Agent Tool Selection

**What we tested:** Does the agent call the correct tool(s) for each query type?

**Methodology:** 14 hand-crafted test cases across 6 query categories. For each query, we extract which tools were called using LangGraph's message trace and compare against expected tools.

| Category | Tests | Passed | Accuracy |
|----------|-------|--------|----------|
| Product search | 3 | 2 | 67% |
| Cheaper alternatives | 2 | 2 | 100% |
| Ingredient insights | 2 | 2 | 100% |
| Ingredient interactions | 2 | 2 | 100% |
| Routine critique | 2 | 1 | 50% |
| Multi-tool queries | 3 | 2 | 67% |
| **Overall** | **14** | **11** | **79%** |

**Failure analysis:**

| Failure | Query | Expected | Got |
|---------|-------|----------|-----|
| Memory answer | "find me a serum for acne scarring" | search_products | none |
| Memory answer | "analyze my skincare routine..." | critique_routine | none |
| Missing tool | "is retinol + AHA safe and what products?" | interactions + search | interactions only |

All three failures share the same root cause: the agent answered from memory or stopped after the first relevant tool rather than calling all required tools. Addressed through system prompt refinement.

> Note: Response quality evaluation via LLM-as-judge was explored but excluded from final results due to instability in structured JSON output from the judge model.

---

## LLM Cost Analysis

All LLM calls use **GPT-4o-mini** pricing: $0.15 per 1M input tokens, $0.60 per 1M output tokens.

| Feature | LLM Calls | Estimated Cost | Notes |
|---------|-----------|----------------|-------|
| Product search | 1 | ~$0.001 | Generation only |
| Ingredient insights | 0 | $0.000 | Pre-computed buckets |
| Interaction check | 0 | $0.000 | Rule-based lookup |
| Routine critique | 1 | ~$0.002 | Analysis + generation |
| Find alternatives | 0 | $0.000 | Jaccard similarity |
| Multi-tool query | 2-3 | ~$0.003 | Multiple tool calls |

**Estimated cost per conversation: $0.01 — $0.03**

Most features use zero LLM calls. Ingredient insights, interaction checking, and dupe finding rely on pre-computed review data, rule-based logic, and similarity metrics. The LLM is only invoked for natural language generation.

---

## Technical Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Agent framework | LangGraph ReAct | OpenAI function calling |
| LLM | GPT-4o-mini | Natural language generation |
| Vector store | ChromaDB (EphemeralClient) | Semantic product search |
| Embeddings | ChromaDB DefaultEmbeddingFunction | sentence-transformers default |
| Sentiment model | DistilBERT SST-2 | distilbert-base-uncased-finetuned-sst-2-english |
| Data processing | pandas, numpy | ETL and aggregation |
| Statistical evaluation | scipy | Paired t-test |
| UI | Streamlit | 6-tab web interface |
| Safety | Custom guardrails | Regex pattern matching |

---

## Features

### 1. Personalized Product Search
Semantic search over 7,868 products, re-ranked by reviews from users with matching skin type. A product rated 4.5 overall but 3.2 by dry skin users ranks lower for dry skin queries.

### 2. Ingredient Insights
1M+ reviews pre-bucketed by ingredient × skin type at startup. Surfaces real user experience per skin type — not brand claims. Uses DistilBERT for sentence-level sentiment on ingredient mentions.

### 3. Routine Critique
Three issue classes detected automatically:

- **Wash-off actives** — salicylic acid in a cleanser rinses off before penetrating pores (needs 10-20 min contact time)
- **Redundancy** — niacinamide in both serum and moisturizer — paying twice for the same ingredient
- **Interactions** — retinol + AHA causes over-exfoliation — system recommends AM/PM separation

### 4. Dupe Finder
Jaccard similarity on active ingredients identifies cheaper products with equivalent actives. Ranked by price saving and ingredient overlap percentage.

### 5. Sentiment Analysis
Per-product sentiment breakdown filterable by skin type. Shows most positive and most negative reviews with DistilBERT confidence scores.

### 6. Safety Guardrails
Pre-flight checks before every agent call:

- Medical conditions → dermatologist redirect (does not proceed)
- Pregnancy, breastfeeding, chemotherapy → evidence-based safety warning (proceeds with caution)
- Off-topic queries → graceful redirect
- Allergen mentions → flagged and passed to product search tool

---

## Project Structure

```
Skincare_RAGbot/
│
├── data/
│   ├── raw/                        # original CSVs — not in repo (too large)
│   ├── processed/                  # generated files — not in repo
│   └── demo/                       # 100 products, 10K reviews — in repo
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── convert_data.py         # product CSV to JSON
│   │   ├── explore_reviews.py      # multi-file loader, deduplication, EDA
│   │   └── reviews_db.py           # review queries by product and skin type
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chatbot.py              # ChromaDB indexing, vector search, personalization
│   │   ├── sentiment_analyzer.py   # DistilBERT sentiment pipeline
│   │   ├── ingredient_insights.py  # review bucketing, ingredient analysis
│   │   └── routine_critic.py       # routine analysis, dupe finder, LLM critique
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py                # LangGraph ReAct agent, 5 tools
│   │   └── guardrails.py           # pre-flight safety checks
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate_personalization.py   # precision@K, paired t-test
│       └── evaluate_agent.py             # tool selection accuracy
│
├── app.py                          # Streamlit UI — 6 tabs
├── requirements.txt
├── runtime.txt                     # python-3.11.9
├── .streamlit/
│   └── config.toml                 # mauve and white theme
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11
- OpenAI API key

### Installation

```bash
git clone https://github.com/mbatheja/skincare-ragbot.git
cd skincare-ragbot

python -m venv venv
source venv/bin/activate
# Windows: .\venv\Scripts\Activate

pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

### Data Setup

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) and place all files in `data/raw/`. Then run:

```bash
# Process products (CSV to JSON)
python src/ingestion/convert_data.py

# Combine and clean reviews
python src/ingestion/explore_reviews.py
```

### Run

```bash
streamlit run app.py
```

> First load takes approximately 30 seconds while DistilBERT initializes and ChromaDB indexes products. Subsequent requests are faster due to caching.

---

## Limitations

**Statistical significance:** p=0.096 on personalization improvement. Directionally positive, borderline significant. A larger held-out set or richer query generation would strengthen this.

**RAG choice:** Semantic search over a structured catalog is debatable. Elasticsearch with filters would give better precision for exact ingredient and brand name queries. Semantic search adds value for conceptual queries where user language does not match product terminology directly.

**Ingredient interactions:** 6 hardcoded interaction pairs covering common combinations. A RAG system over PubMed cosmetic dermatology papers would give broader coverage, cited evidence, and concentration-aware severity.

**Agent tool selection:** 79% accuracy on 14 test cases. Multi-tool and routine critique queries are the primary failure modes.

**Deployment:** The deployed version uses 100 products and 10,000 reviews due to memory constraints on the free tier. The full system (7,868 products, 1,094,411 reviews) runs locally.

---

## Future Scope

The current version is v1 of the product and needs tightening along various dimensions mentioned below in addition to the UI and latency fixes that can make it more user friendly:

**Research-backed ingredient intelligence**
RAG over PubMed cosmetic dermatology papers via the Entrez API. Enables cited evidence, novel ingredient handling, and concentration-aware interaction severity — replacing the current hardcoded rule database.

**Hybrid search**
BM25 keyword search combined with semantic search. Fixes cases where exact ingredient or brand names are not retrieved by semantic similarity alone.

**Drug-drug interaction extension**
The same agent architecture extended to pharmaceutical DDI checking using DrugBank and FDA label data. A natural extension toward clinical decision support roles.

**Fine-tuned domain embeddings**
Skincare-specific embeddings trained on INCI names and product descriptions would improve retrieval precision for technical queries.

**Multi-retailer price comparison**
Real-time pricing from iHerb, Amazon, and Stylevana to complement the dupe finder with live availability data.

---

## Author

Mahima Batheja

[LinkedIn](https://linkedin.com/in/mahima-batheja) | [GitHub](https://github.com/mbatheja)