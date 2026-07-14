import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from agent.agent import SkincarAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# TEST_CASES
## Single tool query
TEST_CASES = [
    {
        "query": "recommend a moisturizer for dry skin under $40",
        "expected_tools": ["search_products"],
        "category": "product_search"
    },
    {
        "query": "find me a serum for acne scarrring",
        "expected_tools": ["search_products"],
        "category": "product_search"
    },
    {
        "query":"what cleanser is good for oily skin",
        "expected_tools": ["search_products"],
        "category": "product_search"
    },
    {
        "query": "Paula's Choice BHA is too expensive, any alternatives?",
        "expected_tools": ["find_cheaper_alternatives"],
        "category": "alternatives"
    },
    {
        "query": "find a cheaper version of Drunk Elephant Protini",
        "expected_tools": ["find_cheaper_alternatives"],
        "category": "alternatives"
    },
    {
        "query": "is niacinamide good for oily skin?",
        "expected_tools": ["get_ingredient_insights"],
        "category": "ingredient"
    },
    {
        "query": "how does retinol perform for sensitive skin",
        "expected_tools": ["get_ingredient_insights"],
        "category": "ingredient"
    },
    {
        "query": "can I use retinol with vitamin C?",
        "expected_tools": ["check_ingredient_interactions"],
        "category": "interactions"
    },
    {
        "query": "is it safe to combine AHA and BHA?",
        "expected_tools": ["check_ingredient_interactions"],
        "category": "interactions"
    },
    {
        "query": "please ctitique my routine: Cetaphil Gentle cleanser, Purito Centella Toner, Purito luminuous ceramide moisturizer, Photostable Gold Matte Sunscreen.",
        "expected_tools": ["critique_routine"],
        "category": "routine"
    },
    {
        "query": "analyze my skincare routine - retinol serum, glycolic acid toner, moisturizer",
        "expected_tools": ["critique_routine"],
        "category": "routine"
    },
## Multi-tool queries
    {
        "query": "is niacinamide good for dry skin and can you recommend a serum with it?",
        "expected_tools": ["get_ingredient_insights", "search_products"],
        "category": "multi-tool"
    },
    {
        "query": "I want to use retinol with AHA - is that safe and what products do you recommend?",
        "expected_tools": ["check_ingredient_interactions", "search_products"],
        "category": "multi_tool"
    },
    {
        "query": "find a cheaper altenrative to The Ordinary Retinol and check if it works for dry skin",
        "expected_tools": ["find_cheaper_alternatives", "get_ingredient_insights"],
        "category": "multi_tool"
    }
]

class AgentEvaluator:
    """Evaluate the LangChain agent on tool selection and response quality."""

    def __init__(self, agent: SkincarAgent):
        """
        Initialize agent
        """
        self.agent = agent

    def get_tool_calls(self, query: str) -> List[str]:
        """
        Run agent and extract which tools were called.
        Return list of tool names in order called. 
        """
        result = self.agent.agent_executor.invoke({
            "messages": [HumanMessage(content=query)]
        })

        tools_called = []

        for message in result.get("messages", []):
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name') or tool_call.get('function', {}).get('name')
                    if tool_name:
                        tools_called.append(tool_name)
        
        final_response = result["messages"][-1].content if result.get("messages") else ""
        
        return tools_called, final_response

    def evaluate_tool_selection(self, test_cases: List[Dict] = None) -> pd.DataFrame:
        """
        Evaluate whether agent calls correct tools for each query.
        """

        if test_cases is None:
            test_cases = TEST_CASES

        print("TOOL SELECTION EVALUATION")
        print(f"Running {len(test_cases)} test cases... \n")

        results = []

        for i, test in enumerate(test_cases, 1):
            query = test["query"]
            expected = set(test["expected_tools"])
            category = test["category"]

            print(f"Test {i}/{len(test_cases)}: [{category}]")
            print(f"Query: {query[:60]}....")

            try:
                tools_called, response = self.get_tool_calls(query)
                tools_called_set = set(tools_called)
                all_expected_called = expected.issubset(tools_called_set)
                unexpected_tools = tools_called_set - expected

                # for multi-tool
                overlap = expected & tools_called_set
                partial_score = len(overlap) / len(expected) if expected else 0
                status = "🟢" if all_expected_called else "🔴"
                print(f"Expected: {sorted(expected)}")
                print(f"Got: {tools_called}")
                print(f"Result: {status}")

                if unexpected_tools:
                    print(f" Unexpected: {sorted(unexpected_tools)}")
                print()

                results.append({
                    "query": query,
                    "category": category,
                    "expected_tools": sorted(expected), 
                    "tools_called": tools_called,
                    "all_correct": all_expected_called,
                    "partial_score": partial_score,
                    "unexpected_tools": sorted(unexpected_tools),
                    "response_length": len(response),
                    "n_tools_called": len(tools_called),
                    "n_tools_expected": len(expected)
                })

            except Exception as e:
                print(f"ERROR: {e}\n")
                results.append({
                    "query": query,
                    "category": category,
                    "expected_tools": sorted(expected),
                    "tools_called": [],
                    "all_correct": False, 
                    "partial_score": 0.0,
                    "unexpected_tools": [],
                    "response_length": 0,
                    "n_tools_called": 0,
                    "n_tools_expected": len(expected),
                    "error": str(e)
                })
        return pd.DataFrame(results)
    
    def evaluate_response_quality(self, test_cases=None) -> pd.DataFrame:
        """
        Use GPT-4o as judge to evaluate response quality.
        """

        if test_cases is None:
            test_cases = TEST_CASES

        print(f"Running {len(test_cases)} test cases... \n")

        # Use LLM as judge
        judge = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY")
            )

        results = []

        for i, test in enumerate(test_cases, 1):
            query = test["query"]
            category = test["category"]

            print(f"Test {i}/{len(test_cases)}: [{category}]")
            print(f"Query: {query[:60]}...")

            try:
                #Get response
                _,response = self.get_tool_calls(query)

                # Judge prompt
                judge_prompt = f"""You are evaluating a Skincare AI advisor's response.
                
                User query: "{query}"
                Agent response: "{response}"

                Rate the response on these 4 criteria, each scored 1-5:

                1. RELEVANCE (1-5): Does the response directly address what was asked? 1 = completely off topic, 5 = perfectly on topic
                2. SPECIFICITY (1-5): Does it mention specific products/ingredients/prices from the catalog? 1 = vague generalities, 5 = specific names, prices, ratings cited
                3. ACCURACY (1-5): Is the advice scientifically sound and safe? 1 = harmful/wrong advice, 5 = accurate, evidence-based
                4. HELPFULNESS (1-5): Would a real user find this useful? 1 = useless, 5 = highly actionable
                
                Return ONLY a JSON object like this, no other text:
                {{"relevance":4, "specificity":3, "accuracy":5, "helpfulness":4, "reasoning": "brief scoring explanation"}}"""

                judgment = judge.invoke(judge_prompt)

                scores = json.loads(judgment.content.strip())

                avg_score = (
                scores["relevance"] +
                scores["specificity"] +
                scores["accuracy"] +
                scores["helplessness"]) / 4

                print(f"  Relevance:    {scores['relevance']}/5")
                print(f"  Specificity:  {scores['specificity']}/5")
                print(f"  Accuracy:     {scores['accuracy']}/5")
                print(f"  Helpfulness:  {scores['helpfulness']}/5")
                print(f"  Average:      {avg_score:.1f}/5")
                print(f"  Reasoning:    {scores['reasoning'][:100]}")
                print()        

                results.append({
                    "query": query,
                    "category": category,
                    "response": response[:200],
                    "relevance": scores["relevance"],
                    "specificity": scores["specificity"],
                    "accuracy": scores["accuracy"],
                    "helpfulness": scores["helpfulness"],
                    "avg_score": avg_score,
                    "reasoning": scores["reasoning"]
                })

            except Exception as e:
                print(f"ERROR: {e}\n")
                results.append({
                    "query": query,
                    "category": category,
                    "response": "",
                    "relevance": 0,
                    "specificity": 0,
                    "accuracy": 0,
                    "helpfulness": 0,
                    "avg_score": 0,
                    "error": str(e)
                })
        
        return pd.DataFrame(results)
    
    def print_summary(self, tool_df: pd.DataFrame, quality_df: pd.DataFrame = None):
        """
        Print final evaluation summary.
        """

        print("EVALUATION SUMMARY")
        print("\n")

        print("\n TOOL SELECTION ACCURACY")

        overall_accuracy = tool_df["all_correct"].mean()
        avg_partial = tool_df["partial_score"].mean()

        print(f"Overall accuracy: {overall_accuracy:.0%}")
        print(f"Avg partial score: {avg_partial:.0%}")
        print(f"Total test cases: {len(tool_df)}")
        print(f"Passed: {tool_df['all_correct'].sum()}")
        print(f"Failed: {(~tool_df['all_correct']).sum()}")

        # By category
        print("\nBy category:")
        category_results = tool_df.groupby("category")["all_correct"].agg(["mean", "count", "sum"]).round(2)
        category_results.columns = ["Accuracy", "Total", "Passed"]
        print(category_results.to_string())

        # Failed cases
        failed = tool_df[~tool_df["all_correct"]]
        if len(failed) > 0:
            print(f"\n Failed cases ({len(failed)}):")
            for _, row in failed.iterrows():
                print(f"Query: {row['query'][:60]}...")
                print(f"Expected: {row['expected_tools']}")                
                print(f"Got: {row['tools_called']}")
                print()
        
        # Quality summary
        if quality_df is not None and len(quality_df) > 0:
            print("\n RESPONSE QUALITY (LLM-as-Judge)")

            print(f"Overall avg score:  {quality_df['avg_score'].mean():.1f}/5")
            print(f"Relevance:          {quality_df['relevance'].mean():.1f}/5")
            print(f"Specificity:        {quality_df['specificity'].mean():.1f}/5")
            print(f"Accuracy:           {quality_df['accuracy'].mean():.1f}/5")
            print(f"Helpfulness:        {quality_df['helpfulness'].mean():.1f}/5")

            print("\nBy category:")
            cat_quality = quality_df.groupby("category")["avg_score"].mean().round(2)
            print(cat_quality.to_string())

    def save_results(self, tool_df: pd.DataFrame,
                    quality_df: pd.DataFrame = None,
                    output_file: str = "agent_evaluation_results.json"):
        """
        Save results to JSON.
        """
        results = {
            "timestamp":   datetime.now().isoformat(),
            "n_test_cases": len(tool_df),

            "tool_selection": {
                "overall_accuracy": float(tool_df["all_correct"].mean()),
                "avg_partial_score": float(tool_df["partial_score"].mean()),
                "by_category": tool_df.groupby("category")["all_correct"].mean().round(2).to_dict(),
                "details": tool_df.to_dict("records")
            }
        }

        if quality_df is not None and len(quality_df) > 0:
            results["response_quality"] = {
                "avg_score":      float(quality_df["avg_score"].mean()),
                "avg_relevance":  float(quality_df["relevance"].mean()),
                "avg_specificity": float(quality_df["specificity"].mean()),
                "avg_accuracy":   float(quality_df["accuracy"].mean()),
                "avg_helpfulness": float(quality_df["helpfulness"].mean()),
                "by_category": quality_df.groupby("category")["avg_score"]
                                         .mean().round(2).to_dict(),
                "details": quality_df.to_dict("records")
            }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n Results saved to {output_file}")

if __name__ == "__main__":
    print("Loading agent...")
    agent = SkincarAgent()
    evaluator = AgentEvaluator(agent)

    print("\nRunning tool selection evaluation...")
    tool_df = evaluator.evaluate_tool_selection()

    run_quality = input("\nRun response quality evaluation? (uses GPT-4o, costs more) [y/n]: ")

    quality_df = None
    if run_quality.lower() == 'y':
        print("\nRunning response quality evaluation...")
        quality_df = evaluator.evaluate_response_quality()

    evaluator.print_summary(tool_df, quality_df)

    evaluator.save_results(tool_df, quality_df)