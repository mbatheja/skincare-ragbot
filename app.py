import sys
import os
import streamlit as st
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from agent.agent import SkincarAgent

def format_product(p):
    """Safely format product for display in selectbox."""
    if not p:
        return "Select a product"
    try:
        name  = p.get('name', 'Unknown')
        brand = p.get('brand', 'Unknown')
        price = p.get('price', 0)
        return f"{brand} — {name} (${float(price):.2f})"
    except Exception:
        return f"{p.get('brand', '')} — {p.get('name', 'Unknown')}"

st.set_page_config(
    page_title="SkinQ",
    layout="wide"
)

# Load agent
@st.cache_resource
def load_agent():
    return SkincarAgent()

agent = load_agent()

# Sidebar
with st.sidebar:
    st.title("QSkin")
    st.markdown("*Advise on all your skincare questions*")
    st.markdown("---")

    st.markdown("### Your Profile")
    skin_type = st.selectbox(
        "Skin Type",
        ["dry", "combination", "oily", "normal", "sensitive", "unknown"],
        help = "Used across all tabs for personalized recommendations"
    )
    st.session_state.skin_type = (
        None if skin_type == "Not specified" else skin_type
    )

    skin_concerns = st.multiselect(
        "Skin Concerns",
        ["acne", "dryness", "oiliness", "dark spots", "hyperpigmentation",
         "scarring", "wrinkles", "sensitivity", "dullness", "large pores",
         "dark circles", "redness"],
         help="Used to personalize recommendations and routine critique."
    )

    st.session_state.skin_concerns = skin_concerns
    st.markdown("---")

    st.markdown("### Quick prompts")
    st.caption("• Recommend a moisturizer under $40")
    st.caption("• Glow Recipe Watermelon Pink Juice Oil-Free Moisturizer is too expensive")
    st.caption("• Is niacinamide good for oily skin?")
    st.caption("• Can I use retinol with vitamin C?")
    st.caption("• Review my routine: SK-II Facial Treatment Cleanser, First Aid Beauty Ultra Repair Cream")

#Build user context
def get_user_context() -> str:
    """
    Build context string from sidebar profile fro agent calls.
    """
    parts = []
    if st.session_state.get('skin_type'):
        parts.append(f"{st.session_state.skin_type} skin")
    if st.session_state.get('skin_concerns'):
        parts.append(f"concerns: {','.join(st.session_state.skin_concerns)}")
    return f"[User profile - {';'.join(parts)}]" if parts else ""

# Services
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Agent Chat",
    "Routine Builder",
    "Ingredient Insights",
    "Sentiment Analysis",
    "Find Dupes",
    "Interactions"
])

# Tab 1: Agent Chat
with tab1:
    st.header("Skincare Advisor")
    st.caption("Ask me anything about skincare")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    if prompt := st.chat_input("Ask about products, ingredients, or your routine..."):
        context = get_user_context()
        enriched_prompt = f"{context}{prompt}" if context else prompt

        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                response = agent.chat(
                    user_message = enriched_prompt,
                    chat_history = st.session_state.messages[:-1]
                )
            st.markdown(response)
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response
            })
        
        if st.session_state.messages:
            if st.button('Clear conversation', key='clear_chat'):
                st.session_state.messages = []
                st.rerun()

#Tab 2: Routine Builder
with tab2:
    st.header("Routine Builder & Analyzer")

    st.markdown("### Step 1: Your Skin Profile")

    col1, col2 = st.columns(2)
    with col1:
        routine_skin_type = st.selectbox(
            "Skin Type",
            ["dry", "oily", "combination", "normal", "sensitive"],
            index = ["dry", "oily", "combination", "normal", "sensitive"].index(
                st.session_state.skin_type
            ) if st.session_state.get('skin_type') else 0,
            key = "routine_skin_type"
            )
        
    with col2:
        routine_concerns = st.multiselect(
            "Skin Concerns",
            ["acne", "dryness", "oiliness", "dark spots", "hyperpigmintation",
             "scarring", "wrinkles", "sensitivity", "dullness", "large pores",
             "dark circles", "redness"],
             default = st.session_state.get('skin_concerns', []),
             key="routine_concerns"
        )

        concern_description = st.text_area(
            "Describe your skin concerns (optional)",
            placeholder="e.g. I have hormonal acne on my chin, some acne scarring and my skin gets very dry in winter",
            height=80

        )

        #Step 2: Product selection
        st.markdown("### Step 2: Select Products")
        st.caption("Tip: All products shown. Recommendations will  personalized to your skin type.")
        def get_by_category(*keywords):
            """
            Get products matching any of the keywords in category fields.
            """

            return [
                p for p in agent.chatbot.products
                if any(
                    kw.lower() in p.get('category', '').lower()
                    or kw.lower() in p.get('subcategory', '').lower()
                    or kw.lower() in p.get('product_type', '').lower()
                    for kw in keywords
                )
            ]
        
        def format_product(p):
            """
            Format product for dropdown display.
            """

            if p is None:
                return "Skip"
            rating = f"{p['rating']:.1f}" if p['rating'] > 0 else ""
            return f"{p['name']} - ${p['price']:.2f}{rating}"
        
        col1, col2, col3 = st.columns(3)

        with col1:
            cleanser_opts = get_by_category('cleanser', 'wash', 'cleansing')
            selected_cleanser = st.selectbox(
                "Cleanser",
                options=[None] + cleanser_opts,
                format_func=format_product,
                key="sel_cleanser"
            )

            toner_opts = get_by_category('toner', 'essence')
            selected_toner = st.selectbox(
                "Toner (if any)",
                options=[None] + toner_opts,
                format_func = format_product,
                key="sel_toner"
            )

        with col2:
            serum_opts = get_by_category('serum', 'ampoule', 'essence')
            selected_serum = st.selectbox(
                "Serum (if any)",
                options=[None] + serum_opts,
                format_func=format_product,
                key="sel_serum"
            )

            eye_opts = get_by_category('eye cream', 'eye serum', 'eye')
            selected_eye = st.selectbox(
                "Eye cream (if any)",
                options=[None] + eye_opts,
                format_func = format_product,
                key="sel_eye"
            )

        with col3:
            moisturizer_opts = get_by_category('moisturizer', 'cream', 'lotion')
            selected_moisturizer = st.selectbox(
                "Moisturizer",
                options=[None] + moisturizer_opts,
                format_func=format_product,
                key="sel_moisturizer"
            )

            spf_opts = get_by_category('sunscreen', 'spf', 'sun')
            selected_spf = st.selectbox(
                "Sunscreen",
                options=[None] + spf_opts,
                format_func = format_product,
                key="sel_spf"
            )

    #Routine summary

    selected_products = [
        p for p in [
            selected_cleanser, selected_toner, selected_serum,
            selected_eye, selected_moisturizer, selected_spf
        ] if p is not None
    ]

    if selected_products:
        st.markdown("---")
        st.markdown("### Your Routine")

        total_cost = sum(p["price"] for p in selected_products)
        cols = st.columns(len(selected_products))

        for col,p in zip(cols, selected_products):
            with col:
                st.markdown(f"**{p['category']}**")
                st.write(p['name'])
                st.write(f"${p['price']:.2f}")
                if p['rating'] > 0:
                    st.caption(f"{p['rating']:.1f}/5")

        st.write(f"**Total: ${total_cost:.2f}**")

        #Analyze
        if st.button("Analyze My Routine", type="primary", key="analyze_routine"):
            with st.spinner("Analyzing your routine..."):

                # Rich prompt for agent
                product_names = [p['name'] for p in selected_products]
                concern_text = ""

                if routine_concerns:
                    concern_text += f"My skin concerns are: {','.join(routine_concerns)}."
                
                if concern_description:
                    concern_text += f"Additional context: {concern_description}"
                
                prompt = (
                    f"Please critique this skincare routine for someone with"
                    f"{routine_skin_type} skin.{concern_text}"
                    f"Routine: {','.join(product_names)}."
                    f"Check for wash-off actives, ingredient redundancy, "
                    f"harmful interactions, and suggest optimizations."
                )

                critique = agent.chat(user_message=prompt, chat_history=[])

            st.markdown("### Analysis")
            st.markdown(critique)

            # Quick stats
            st.markdown("---")
            st.markdown('### Quick Stats')

            analysis = agent.critic.analyze_routine_with_reviews(
                selected_products,
                skin_type=routine_skin_type
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Wash-off Issues",
                    len(analysis['wash_off_issues']),
                    delta="Found" if analysis['wash_off_issues'] else "None",
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Redundancies",
                    len(analysis['redundancies']),
                    delta="Found" if analysis['redundancies'] else "None",
                    delta_color="inverse"
                )

            with col3:
                st.metric(
                    "Interactions",
                    len(analysis['interactions']),
                    delta="Found" if analysis ['interactopns'] else "None"
                )

            with col4:
                st.metric("Total Cost", f"${total_cost:.2f}")

    # Tab 3: Ingredient Insights
    with tab3:
        st.header("Ingredient Insights")
        st.caption("Review-based performance data for key skincare actives.")

        col1, col2 = st.columns([2,1])
        with col1:
            selected_ingredient = st.selectbox(
                "Select Ingredient",
                options=sorted(agent.extractor.KEY_INGREDIENTS.keys()),
                key="insight_ingredient"
            )

        with col2:
            insight_skin_type = st.selectbox(
                "Skin Type",
                ["dry", "oily", "combination", "normal", "sensitive"],
                index=["dry", "oily", "combination", "normal", "sensitive"].index(
                    st.session_state.skin_type
                ) if st.session_state.get('skin_type') else 0,
                key = "insight_skin_type"
                )
        
        if st.button("Analyze Ingredient", type="primary", key="analyze_ingredient"):
            with st.spinner(f"Analyzing {selected_ingredient} for {insight_skin_type} skin..."):
                result = agent.extractor.analyze_ingredient(
                    selected_ingredient,
                    skin_type=insight_skin_type
                )

            if result:
                #Metrics
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Reviews Analyzed", f"{result['mention_count']:,}")
                with col2:
                    st.metric("Avg Rating", f"{result['avg_rating']:.1f}/5")
                with col3:
                    st.metric("Recommend Rate", f"{result['recommend_rate']:.0%}")
                with col4:
                    st.metric("Positive Sentiment", f"{result['sentiment_positive_rate']:.0%}")
                
                # Warning
                warning = agent.extractor.get_ingredient_warnings(
                    selected_ingredient,
                    skin_type=insight_skin_type
                )

                if warning:
                    st.warning(warning)
                else:
                    st.success(
                        f"{selected_ingredient.capitalize()} performs well"
                        f"for {insight_skin_type} skin based on user reviews."
                    )

                # Skin type comparison
                st.markdown("---")
                st.markdown("### Performance Across Skin Types")
                st.caption("Comparing this ingredient across all skin types in our review data.")

                comparison = agent.extractor.compare_ingredient_across_skin_types(
                    selected_ingredient
                )

                if comparison:
                    comparison_data = []
                    for st_type, data in comparison.items():
                        comparison_data.append({
                            'Skin Type': st_type.capitalize(),
                            'Reviews': data['mention_count'],
                            'Avg Rating': round(data['avg_rating'], 2),
                            'Recommend Rate': f"{data['recommend_rate']:.0%}",
                            'Positive Sentiment': f"{data['sentiment_positive_rate']:.0%}"
                        })

                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(
                        df_comparison,
                        use_container_width=True,
                        hide_index=True
                    )
                
                if result.get('sample_insights'):
                    st.markdown("---")
                    st.markdown(f"### What {insight_skin_type.capitalize()} Skin Users Say")

                    for insight in result['saample_insights'][:5]:
                        sentiment_icon = "🍏" if insight['sentiment'] == 'POSITIVE' else "🍎"
                        confidence = insight.get('confidence', 0)
                        st.markdown(
                            f"{sentiment_icon} *\"{insight['text'][:250]}\"*"
                        )
                        st.caption(
                            f"Rating: {insight['rating']}/5 |"
                            f"Sentiment confidence: {confidence:.0%}"
                        )
                        st.markdown("---")
                
                else:
                    st.info(
                        f"Not enough review data for **{selected_ingredient}**"
                        f"+ **{insight_skin_type}** skin. Try a different combination."
                    )

        # Tab 4: Sentiment Analysis

        with tab4:
            st.header("Product Sentiment Analysis")
            st.caption("Analyze how users feel about a specfic product from their review text.")

            col1, col2 = st.columns([3,1])
            with col1:
                product_search = st.text_input(
                    "Search for a product",
                    placeholder="e.g. e.g. Sunday Riley Ceramic Slip Cleanser, First Aid Beauty Ultra Repair Cream",
                    key="sentiment_product_search"
                )
                
            with col2:
                sentiment_skin_type = st.selectbox(
                    "Filter by skin type (optional)",
                    ["All skin types", "dry", "oily", "combination", "normal", "sensitive"],
                    key="sentiment_skin_type"
                )

            # Search and display matching products
            if product_search:
                matching_products = [
                    p for p in agent.chatbot.products
                    if product_search.lower() in p['name'].lower()
                    or product_search.lower() in p['brand'].lower()
                    or product_search.lower() in f"{p['brand']} {p['name']}".lower()
                ]

                if not matching_products:
                    st.warning(f"No products found matching '{product_search}'")
                else:
                    selected_for_sentiment = st.selectbox(
                        "Select product",
                        options = matching_products,
                        format_func = format_product,
                        key = "sentiment_selected_product"
                    )

                    if st.button("Analyze Sentiment", type="primary", key="run_sentiment"):
                        with st.spinner(f"Analyzing reviews for {selected_for_sentiment['name']}..."):

                            # Get reviews for this product
                            skin_filter = (
                                None if sentiment_skin_type == "All skin types"
                                else sentiment_skin_type
                            )

                            reviews_for_product = agent.chatbot.reviews_db.get_reviews_for_product(
                                product_id=selected_for_sentiment['id'],
                                skin_type=skin_filter
                            )

                            if len(reviews_for_product) ==0:
                                st.info(
                                    f"No reviews found for this product"
                                    f"{'with' + skin_filter + 'skin' if skin_filter else ''}."
                                )
                            
                            else:
                                sample_reviews = reviews_for_product.head(200)

                                st.caption(
                                    f"Analyzing {len(sample_reviews)} reviews"
                                    f"{'from' + skin_filter + 'skin users' if skin_filter else '' }..." 
                                )

                                analyzed = agent.sentiment_analyzer.analyze_reviews(sample_reviews)

                                # Sentiment summary
                                summary = agent.sentiment_analyzer.get_sentiment_summary(
                                    analyzed,
                                    product_id=selected_for_sentiment['id'],
                                    skin_type=skin_filter
                                )

                                st.markdown("---")
                                st.markdown(f"### Results for **{selected_for_sentiment['name']}**")

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Reviews Analyzed", summary['review_count'])
                                with col2:
                                    st.metric("Positive", f"{summary['positive_rate']:.0%}")
                                with col3:
                                    st.metric("Negative", f"{summary['negative_rate']:.0%}")
                                with col4:
                                    st.metric(
                                        "Avg Confidence",
                                        f"{summary['avg_sentiment_score']:.0%}"
                                    )

                                #Sentiment by rating
                                st.markdown("---")
                                st.markdown("#### Sentiment Breakdown by Star Rating")

                                rating_sentiment = (
                                    analyzed.groupby('rating')['sentiment_label']
                                    .value_counts(normalize=True)
                                    .unstack(fill_value=0)
                                    .reset_index()
                                )

                                if 'POSITIVE' in rating_sentiment.columns:
                                    st.dataframe(
                                        rating_sentiment.round(2),
                                        use_container_width=True,
                                        hide_index=True
                                    )

                                #Sample Reviews
                                st.markdown("---")
                                st.markdown("#### Sample Reviews")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("** Most Positive**")
                                    positive_reviews = (
                                            analyzed[analyzed['sentiment_label'] == 'POSITIVE']
                                            .nlargest(3, 'sentiment_score')
                                        )

                                    for _, row in positive_reviews.iterrows():
                                        st.caption(
                                            f" {int(row['rating'])}/5 |"
                                            f"Confidence: {row['sentiment_score']:.0%}"
                                        )
                                        text = str(row.get('review_text', ''))[:300]
                                        st.markdown(f"*\"{text}...\"*")
                                        st.markdown("---")
                                
                                with col2:
                                    st.markdown("** Most negative**")
                                    negative_reviews  = (
                                        analyzed[analyzed['sentiment_label'] == 'NEGATIVE']
                                    )
                                    for _, row in negative_reviews.iterrows():
                                        st.caption(
                                            f"{int(row['rating'])}/5 |"
                                            f"Confidence: {row['sentiment_score']:.0%}"
                                        )
                                        text = str(row.get('review_text', ''))[:300]
                                        st.markdown(f"*\"{text}...\"*")
                                        st.markdown("---")

        # Tab 5: Find Dupes
        with tab5:
            st.header("Find Cheaper Alternatives")
            st.caption(
                "Find products with similar active ingredients at a lower price."
                "Similarity is based on matching key actives, not full ingredient lists."
            )

            col1, col2, col3 = st.columns([3,1,1])
            with col1:
                dupe_search = st.text_input(
                    "Product name",
                    placeholder="e.g. Charlotte Tilbury Goddess Clay Mask, SK-II Facial Treatment Toner",
                    key = "dupe_search"
                )
            
            with col2:
                dupe_max_price = st.number_input(
                    "Max budget ($)",
                    min_value=1.0,
                    max_value=500.0,
                    value=30.0,
                    key="dupe_max_price"
                )
            
            with col3:
                dupe_top_n = st.number_input(
                    "Results",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="dupe_top_n"
                )

            if dupe_search:
                #Show matching products
                matches = next(
                    (p for p in agent.chatbot.products
                    if dupe_search.lower() in p['name'].lower()
                    or dupe_search.lower() in f"{p['brand']} {p['name']}".lower()
                    or all(word in f"{p['brand']} {p['name']}".lower() 
                            for word in dupe_search.lower().split())),
                    None
                )

                if not matches:
                    st.warning(f"No products found matching '{dupe_search}'")
                else:
                    reference_product = st.selectbox(
                        "Select reference product",
                        options = matches,
                        format_func= format_product,
                        key="dupe_reference"
                    )
                
                    if st.button("Find Alternatives", type="primary", key="find_dupes"):
                        with st.spinner(f"Searching for alternatives..."):
                            dupes = agent.critic.find_product_dupes(
                              products= agent.chatbot.products,
                              product_id=reference_product['id'],
                              max_price=dupe_max_price,
                              top_n=int(dupe_top_n)  
                            )

                        st.markdown("---")
                        st.markdown(
                            f"**Reference:** {reference_product['name']}"
                            f"by {reference_product['brand']} - "
                            f"${reference_product['price']:.2f}"
                        )

                        # Show reference acctives
                        ref_actives = agent.critic.extract_key_actives(
                            reference_product.get('ingredients', [])
                        )
                        if ref_actives:
                            st.caption(f"Key actives: {','.join(ref_actives)}")
                        
                        st.markdown("---")

                        if dupes:
                            for dupe in dupes:
                                p = dupe['product']
                                savings = dupe['price_saving']

                                with st.expander(
                                    f"**{p['name']}** by {p['brand']} -"
                                    f"${p['price']:.2f} | "
                                    f"{dupe['similarity']:.0%} match"
                                    + (f" | saves ${savings:.2f}" if savings > 0 else "")
                                ):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Price", f"${p['price']:.2f}")
                                    with col2:
                                        st.metric("Similarity", f"{dupe['similarity']:.0%}")
                                    with col3:
                                        st.metric("Savings", f"${savings:.2f}" if savings > 0 else "~Same price")

                                    st.write(
                                        f"**Matching actives:**"
                                        f"{','.join(dupe['matching_actives']) or 'None detected'}"
                                    )
                                    if dupe['missing_actives']:
                                        st.write(
                                            f"**Missing actives**"
                                            f"{','.join(dupe['missing_actives'])}"
                                        )
                                    if dupe['extra_actives']:
                                        st.write(
                                            f" **Extra actives**"
                                            f"{','.join(dupe['extra_actives'])}"
                                        )
                                    
                                    if p['rating'] > 0:
                                        st.caption(
                                            f"{p['rating']:.1f}/5"
                                            f"({p['reviews_count']} reviews)"
                                        )
                        
                        else:
                            st.info(
                                f"No alternatives found under ${dupe_max_price:.2f}"
                                f"with matching active ingredients."
                                f"Try increasing your budget."
                            )

    #Tab 6: Ingredient Interactions
    with tab6:
        st.header("Ingredient Interaction Checker")
        st.caption("Check whether ingredients in your routine interact negatively.")

        interaction_mode = st.radio(
            "Input mode",
            ["Enter ingredients", "Enter product names"],
            horizontal=True,
            key="interaction_mode_radio"
        )

        ingredients = []

        if interaction_mode == "Enter ingredients":
            interaction_input = st.text_area(
                "Enter ingredients (one per line or comma-separated)",
                placeholder="retinol\naha\nvitamin c",
                height=120,
                key="ingredient_text_input"
            )
            if interaction_input:
                ingredients = [
                    i.strip()
                    for i in interaction_input.replace('\n', ',').split(',')
                    if i.strip()
                ]

        else:  # Enter product names
            interaction_input = st.text_area(
                "Enter product names (one per line)",
                placeholder="First Aid Beauty Ultra Repair Cream\nOLEHENRIKSEN Glow Cycle Serum",
                height=120,
                key="product_name_input"
            )
            if interaction_input:
                product_names = [
                    n.strip() for n in interaction_input.split('\n') if n.strip()
                ]
                for name in product_names:
                    match = next(
                        (p for p in agent.chatbot.products
                        if name.lower() in p['name'].lower()
                        or name.lower() in f"{p['brand']} {p['name']}".lower()
                        or all(word in f"{p['brand']} {p['name']}".lower() 
                                for word in name.lower().split())),
                        None
                    )

                    if match:
                        actives = agent.critic.extract_key_actives(
                            match.get('ingredients', [])
                        )
                        ingredients.extend(actives)
                        st.caption(
                            f"✓ {match['name']} → "
                            f"{', '.join(actives) if actives else 'no actives detected'}"
                        )
                    else:
                        st.caption(f"✗ '{name}' not found in catalog")
                        ingredients.append(name)

        if st.button(" Check Interactions", type="primary", key="check_interactions_btn"):
            if not ingredients:
                st.warning("Please enter ingredients or product names first.")
            else:
                dummy_products = [
                    {
                        'name':        ing,
                        'category':    'serum',
                        'price':       0,
                        'ingredients': [ing]
                    }
                    for ing in ingredients
                ]
                conflicts = agent.critic.check_ingredient_interactions(dummy_products)

                st.markdown("---")
                st.markdown(f"**Checking:** {', '.join(ingredients)}")
                st.markdown("---")

                if conflicts:
                    severity_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                    for conflict in conflicts:
                        color = severity_color.get(conflict['severity'], '⚠️')
                        with st.expander(
                            f"{color} {conflict['ingredient_1'].capitalize()} + "
                            f"{conflict['ingredient_2'].capitalize()}",
                            expanded=conflict['severity'] == 'high'
                        ):
                            st.write(f"**Issue:** {conflict['issue']}")
                            st.write(f"**Fix:** {conflict['recommendation']}")
                else:
                    st.success(
                        f"✅ No known interactions between: {', '.join(ingredients)}"
                    )
                    
                    with st.expander("View all known interactions"):
                        interactions_data = []
                        for pair, details in agent.critic.INGREDIENT_INTERACTIONS.items():
                            interactions_data.append({
                                'Ingredient 1': pair[0].capitalize(),
                                'Ingredient 2': pair[1].capitalize(),
                                'Severity': details['severity'].upper(),
                                'Issue': details['issue'],
                                'Fix': details['recommendation']
                            })

                        df_interactions = pd.DataFrame(interactions_data)
                        st.dataframe(
                            df_interactions, 
                            use_container_width=True,
                            hide_index=True
                        )

