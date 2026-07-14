import pandas as pd
import numpy as np
from sentiment_analyzer import SentimentAnalyzer
from ingredient_insights import IngredientInsightExtractor
import matplotlib.pyplot as plt

def evaluate_sentiment_accuracy():
    """
    Evaluate if sentiment matches ratings.
    """

    reviews = pd.read_csv('Data/combined_reviews.csv').head(1000)

    analyzer = SentimentAnalyzer()
    analyzed = analyzer.analyze_reviews(reviews)

    # Compare sentiment to ratings
    high_rated = analyzed[analyzed['rating']>=4]
    low_rated = analyzed[analyzed['rating']<=2]

    # Calculate accuracy
    high_rated_positive = (high_rated['sentiment_label']=="POSITIVE").mean()
    low_rated_negative = (low_rated['sentiment_label']=="NEGATIVE").mean()

    print(f"\nSample size:{len(analyzed)}reviews")
    print(f"\nHigh ratings (4-5 stars): {len(high_rated)}reviews")
    print(f"Correctly identified as POSITIVE: {high_rated_positive:.1%}")

    print(f"\nLow ratings (1-2 stars): {len(low_rated)} reviews")
    print(f" Correctly identified as NEGATIVE: {low_rated_negative:.1%}")

    overall_accuracy = (
        (high_rated['sentiment_label']=='POSITIVE').sum() +
        (low_rated['sentiment_label']=='NEGATIVE').sum()
    )/ (len(high_rated) + len(low_rated))
    
    print(f"\nOverall accuracy: {overall_accuracy:.1%}")

    #Visualize
    fig, axes = plt.subplots(1,2,figsize=(12,5))

    #Plot 1: Sentiment by rating
    sentiment_by_rating = analyzed.groupby('rating')['sentiment_label'].value_counts(normalize=True).unstack()
    sentiment_by_rating.plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title('Sentiment Distribution by Rating')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Proportion')
    axes[0].legend(title='Sentiment')

    #Plot 2: Sentiment score distribution
    analyzed.boxplot(column='sentiment_score', by='rating', ax=axes[1])
    axes[1].set_title('Sentiment Score by Rating')
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Sentiment Score')

    plt.tight_layout()
    plt.savefig('sentiment_evaluation.png', dpi=150, bbox_inches='tight')
    print('\n Visualization saved')

    return overall_accuracy

def evaluate_ingredient_insights():
    """
    Show ingredient insight examples.
    """
    print("INGREDIENT INSIGHTS EXAMPLES")

    reviews = pd.read_csv("Data/combined_reviews.csv")
    analyzer = SentimentAnalyzer()
    extractor = IngredientInsightExtractor(reviews, analyzer)

    test_ingredients = ['niacinamide', 'retinol', 'vitamin c', 'hyaluronic acid']

    results = []

    for ingredient in test_ingredients:
        print(f"\n{ingredient.upper()}")

        comparison = extractor.compare_ingredient_across_skin_types(ingredient)

        if not comparison:
            print("Insufficient data")
            continue

        for skin_type,data in comparison.items():
            print(f"\n {skin_type.capitalize()} skin:")
            print(f"Reviews: {data['mention_count']}")
            print(f"Rating: {data['avg_rating']:.2f}/5")
            print(f"Recommend:{data['recommend_rate']:.0%}")
            print(f"Positive sentiment: {data['sentiment_positive_rate']:.0%}")

            results.append({
               'ingredient': ingredient,
               'skin_type': skin_type,
               'rating': data['avg_rating'],
               'recommend_rate': data['recommend_rate']
            })

        if results:
            df_results = pd.DataFrame(results)
            pivot = df_results.pivot(index='ingredient', columns='skin_type', values='rating')

            print("INGREDIENT RATINGS BY SKIN TYPE")
            print(pivot.round(2))

if __name__ == "__main__":
    accuracy = evaluate_sentiment_accuracy()

    evaluate_ingredient_insights()
    print("EVALUATION COMPLETE")
    print(f"\n Sentiment Analysis Accuracy: {accuracy:.1%}")
    print("Check sentiment_evaluation.png for visualizations")




