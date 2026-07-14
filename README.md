# QSkin — AI-Powered Skincare Advisor

An agentic AI system that provides personalized skincare recommendations, routine analysis, and ingredient insights — backed by 1M+ verified Sephora product reviews.

---

## Problem

Building an effective skincare routine requires synthesizing information scattered across incompatible sources:

- **Product reviews** on Sephora, Amazon, and Reddit — unstructured, skin-type unfiltered
- **Ingredient databases** like INCIDecoder and Skincarisma — useful but require you to already know what to look for
- **Community knowledge** on YouTube and r/SkincareAddiction — high quality but not searchable or personalized
- **Brand marketing** — optimized for conversion, not accuracy

Existing tools solve parts of this problem in isolation. None address the full picture:

| Tool | What it does | What it misses |
|------|-------------|----------------|
| Skincarisma | Ingredient safety scoring | No routine-level analysis, no skin-type personalization |
| INCIDecoder | Ingredient explanation | No product recommendations, no interaction checking |
| Sephora reviews | Product ratings | No skin-type filtering, no ingredient context |
| Reddit/YouTube | Community experience | Not searchable, not personalized, not scalable |

**The gap nobody fills:** how do ingredients across *multiple products in a routine* interact with each other — and does that interaction change based on your skin type?

A salicylic acid cleanser + AHA toner + retinol serum sounds like a solid routine. For dry skin it's a recipe for barrier damage. No existing consumer tool tells you this.

---

## Solution

QSkin consolidates these fragmented sources into a single personalized advisor:

- **1M+ Sephora reviews filtered by skin type** → replaces generic star ratings with "how did people with *your* skin type rate this?"
- **Semantic product search** → understands "something for my T-zone" not just exact ingredient names
- **Routine-level analysis** → checks wash-off actives, ingredient redundancy, and cross-product interactions in a single pass
- **Evidence-based ingredient insights** → sentiment analysis over real reviews surfaces what users actually experienced, not what the brand claims
- **Dupe finder** → finds cheaper alternatives with matching active ingredients when a recommended product is out of budget

---

## Architecture
User Query
↓
Guardrails (medical redirect, off-topic, vulnerable contexts)
↓
LangChain ReAct Agent (GPT-4o-mini)
↓ decides which tools to call
┌─────────────────────────────────────────────────────┐
│  Tool 1: search_products                            │
│    → ChromaDB semantic search                       │
│    → Category + price filtering                     │
│    → Skin-type review scoring                       │
│                                                     │
│  Tool 2: find_cheaper_alternatives                  │
│    → Jaccard similarity on active ingredients       │
│    → Price-ranked alternatives                      │
│                                                     │
│  Tool 3: get_ingredient_insights                    │
│    → Pre-bucketed reviews by skin type              │
│    → DistilBERT sentiment analysis                  │
│                                                     │
│  Tool 4: check_ingredient_interactions              │
│    → Rule-based interaction database                │
│    → Severity classification                        │
│                                                     │
│  Tool 5: critique_routine                           │
│    → Wash-off active detection                      │
│    → Redundancy analysis                            │
│    → LLM-generated optimization advice             │
└─────────────────────────────────────────────────────┘
↓
GPT-4o-mini synthesizes tool results
↓
Personalized response with citations

---

## Key Features

### Personalized Product Search
Semantic vector search over 7,800+ products, re-ranked by reviews from users with matching skin type. Products rated 4.8/5 by dry skin users surface above products rated 4.8/5 overall.

### Ingredient Insights
Pre-computed sentiment analysis over 1M+ reviews, bucketed by ingredient × skin type. Answers "does niacinamide actually work for oily skin?" with real data — not marketing claims.

### Routine Critique
Identifies three classes of routine problems:
- **Wash-off actives**: salicylic acid in a cleanser gets rinsed off before penetrating pores
- **Redundancy**: niacinamide in both serum and moisturizer — paying twice for the same ingredient
- **Interactions**: retinol + AHA causes over-exfoliation — suggests AM/PM separation

### Dupe Finder
Finds cheaper alternatives using Jaccard similarity on active ingredients. "Paula's Choice BHA is $34 — here are 3 alternatives under $20 with 80%+ ingredient overlap."

### Safety Guardrails
Pre-flight checks before every agent call:
- Medical conditions (eczema, psoriasis) → dermatologist redirect
- Vulnerable contexts (pregnancy, breastfeeding, chemotherapy) → safety warning with regex pattern matching
- Off-topic queries → graceful redirect

---

## Evaluation

### Phase 1: Retrieval Personalization
Evaluated on 2,000 held-out reviews using category-aware queries:

| Approach | Precision@5 | Precision@10 |
|----------|-------------|--------------|
| Baseline (semantic only) | 0.90% | 1.60% |
| Personalized (skin-type scoring) | 1.65% | 2.80% |
| **Improvement** | **+83%** | **+75%** |

Statistical significance: p=0.007 (paired t-test, n=2,000)

### Phase 2: Agent Tool Selection
Evaluated on 14 test cases across 6 query categories:

| Category | Accuracy |
|----------|----------|
| Product search | 100% |
| Ingredient insights | 100% |
| Interactions | 100% |
| Routine critique | 100% |
| Alternatives | 100% |
| Multi-tool queries | 67% |
| **Overall** | **~85%** |

Multi-tool accuracy is lowest — complex queries requiring 2+ tools are the primary failure mode.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | LangChain ReAct + OpenAI function calling |
| LLM | GPT-4o-mini |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Sentiment analysis | DistilBERT (distilbert-base-uncased-finetuned-sst-2-english) |
| Data processing | pandas, numpy |
| UI | Streamlit |
| Statistical evaluation | scipy (paired t-test) |

---

## Dataset

- **Products**: 7,868 Sephora products (name, brand, price, ingredients, category, ratings)
- **Reviews**: 1,094,411 reviews across 5 files (rating, skin type, review text, recommendation)
- **Source**: [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) — Kaggle

---

## Project Structure
Skincare_RAGbot/
├── data/
│   ├── raw/                    # original CSVs (not in repo — too large)
│   └── processed/              # generated files (not in repo)
│
├── src/
│   ├── ingestion/
│   │   ├── convert_data.py     # CSV → JSON product conversion
│   │   ├── explore_reviews.py  # multi-file review loader + EDA
│   │   └── reviews_db.py       # review queries by product/skin type
│   │
│   ├── core/
│   │   ├── chatbot.py          # vector search + personalization
│   │   ├── sentiment_analyzer.py # DistilBERT sentiment
│   │   ├── ingredient_insights.py # ingredient performance by skin type
│   │   └── routine_critic.py   # routine analysis + dupe finder
│   │
│   ├── agent/
│   │   ├── agent.py            # LangChain ReAct agent + 5 tools
│   │   └── guardrails.py       # safety checks
│   │
│   └── evaluation/
│       ├── evaluate_personalization.py
│       ├── evaluate_agent.py
│       └── evaluate_sentiment.py
│
├── app.py                      # Streamlit UI (6 tabs)
├── requirements.txt
├── runtime.txt                 # python-3.11.9
├── Procfile                    # Railway deployment
└── README.md
---

## Setup

### Prerequisites
- Python 3.11
- OpenAI API key

### Installation

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/skincare-ragbot.git
cd skincare-ragbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Data Setup

Download the dataset from Kaggle and place files in `data/raw/`:
- `product_info.csv`
- `reviews_0-250.csv`
- `reviews_250-500.csv`
- `reviews_500-750.csv`
- `reviews_750-1250.csv`
- `reviews_1250-end.csv`

Then run the data pipeline:

```bash
# Process products
python src/ingestion/convert_data.py

# Combine reviews
python src/ingestion/explore_reviews.py
```

### Run

```bash
# Test agent
python src/agent/agent.py

# Launch UI
streamlit run app.py
```

---

## Limitations

**RAG choice**: Semantic search over a product catalog is debatable — structured Elasticsearch with filters would give better precision for exact ingredient queries. Semantic search adds value for conceptual queries ("something for my T-zone") but is overkill for brand/ingredient name lookup.

**Ingredient interactions**: Currently rule-based (9 rules, 6 interactions). Only covers common combinations. A RAG system over cosmetic dermatology papers would give better coverage and cited evidence.

**Agent reliability**: Multi-tool queries (requiring 2+ tool calls) achieve 67% accuracy vs 100% for single-tool queries. Complex queries like "find a cheaper alternative and check if it works for dry skin" occasionally result in only one tool being called.

**Deployment data**: Deployed version uses a 10K review sample and 100 product subset due to Railway memory constraints. Full dataset runs locally.

---

## Future Scope

**Research-backed ingredient intelligence**
Replace hardcoded rules with RAG over PubMed cosmetic dermatology papers. Enables cited evidence, novel ingredient handling, and concentration-aware interaction severity.

**Hybrid search**
Combine BM25 keyword search (for ingredient/brand name precision) with semantic search (for conceptual queries). Current pure semantic search occasionally returns wrong product categories.

**Fine-tuned embeddings**
Domain-specific embeddings trained on skincare product descriptions would improve retrieval precision for technical queries (INCI names, active concentrations).

**Multi-retailer price comparison**
Integrate iHerb, Amazon, and Stylevana APIs to show real-time pricing and availability alongside product recommendations.

---

## Author

Mahima Batheja
[LinkedIn](https://linkedin.com/in/mahima-batheja) | [GitHub](https://github.com/mbatheja)