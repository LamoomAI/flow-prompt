
import os
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
import typing as t
from dotenv import load_dotenv
import logging

from lamoom.ai_models.tools.base_tool import ToolDefinition, ToolParameter
from lamoom.settings import LAMOOM_GOOGLE_SEARCH_RESULTS_COUNT

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ID = os.getenv("SEARCH_ENGINE_ID")

@dataclass
class WebSearchResult:
    url: str
    title: str
    snippet: str
    content: str

class WebCall:
    @staticmethod
    def scrape_webpage(url: str) -> str:
        """
        Scrapes the content of a webpage and returns the text.
        """
        if not url.startswith(("https://", "http://")):
            url = "https://" + url
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            clean_text = text.splitlines()
            clean_text = [element.strip()
                        for element in clean_text if element.strip()]
            clean_text = '\n'.join(clean_text)
            return clean_text
        else:
            return "Failed to retrieve the website content."

    @staticmethod
    def perform_web_search(query: str) -> t.List[WebSearchResult]:
        """
        Performs a web search and returns a list of search results.
        """
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': API_KEY,
            'cx': SEARCH_ID,
            'num': LAMOOM_GOOGLE_SEARCH_RESULTS_COUNT
        }

        response = requests.get(url, params=params)
        results = response.json()

        search_results = []
        
        if 'items' in results:
            for result in results['items']:
                content = WebCall.scrape_webpage(result['link'])
                search_results.append(WebSearchResult(
                    url=result['link'],
                    title=result['title'],
                    snippet=result['snippet'],
                    content=content
                ))
        logger.debug(f"Retrieved search_results {search_results}")
        return search_results

    @staticmethod
    def format_search_results(results: t.List[WebSearchResult]) -> str:
        """
        Formats search results into a readable string.
        """
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"<result_{i}>\n"
            formatted += f"Title: {result.title}\n"
            formatted += f"URL: {result.url}\n"
            formatted += f"Snippet: {result.snippet}\n"
            formatted += f"Content: {result.content[:500]}...\n"
            formatted += f'</result_{i}>'
        return formatted
    
    @staticmethod
    def execute(query: str) -> str:
        return WebCall.format_search_results(
            WebCall.perform_web_search(query)
        )


def perform_web_search(query: str) -> str:
    """
    Performs a web search and returns formatted results.
    """
    results = WebCall.execute(query)
    return results


WEB_SEARCH_TOOL = ToolDefinition(
    name="web_call",
    description="""
Performs a web search using a search engine to find up-to-date information or details not present in the internal knowledge. Today is {current_datetime_strftime} {timezone}.

<tool_call>
{
"tool_name": "web_call",
"parameters": {
"query": "A search query string"
}
}
""",
    parameters=[
        ToolParameter(name="query", type="string", description="The search query to use.", required=True)
    ],
    execution_function=perform_web_search,
    max_count_of_executed_calls=10
)


def best_customer_experience(reasoning: str) -> str:
    return {'best_customer_experience': reasoning}

BEST_CUSTOMER_EXPERIENCE_TOOL = ToolDefinition(
    name="best_customer_experience",
    description="""Provides reasoning from the perspective of a customer to enhance understanding of customer needs and improve service quality.
1. Start from best_customer_experience
```
<tool_call>
{
"tool_name": "best_customer_experience",
"parameters": {
"reasoning": "As a customer I want to ..."
},
}
</tool_call>
## BEST CUSTOMER EXPERIENCE
{best_customer_experience}""",
    parameters=[
        ToolParameter(name="reasoning", type="string", description="The reasoning from the customer's perspective.", required=True)
    ],
    execution_function=best_customer_experience,
    update_json_context=True,
    max_count_of_executed_calls=1
)

def make_plan(steps: dict) -> str:
    return {'plan': steps}

MAKE_PLAN_TOOL = ToolDefinition(
    name="make_plan",
    description="""Creates a structured plan with steps to achieve a specific goal. Each step should include expected results, actions to take, and the current state.

2. Continue from making a plan, and after going through each step and mark it you're on the step:

<tool_call>
{
"tool_name": "make_plan",
"parameters": {
    'step1': {'expected_results': '', 'actions': '', 'where_we_are': ''},
    ...
    }
}
</tool_call>
<thinking_step_n>
## Step N
- repeat expected results
- repeat current_state from step above
- repeat actions:
- query required to be called;

</thinking_step_n>
<actions>
</actions>
## PLAN
{plan}
""",
    parameters=[
        ToolParameter(name="steps", type="object", description="A dictionary where each key is a step name and the value is another dictionary with 'expected_results', 'actions', and 'where_we_are'. Ex. {'step1': {'expected_results': '', 'actions': '', 'where_we_are': ''}}", required=True)
    ],
    execution_function=make_plan,
    update_json_context=True,
    max_count_of_executed_calls=1
)


def update_plan(step_data: dict) -> str:
    return {'plan': {**step_data}}


UPDATE_PLAN_TOOL = ToolDefinition(
    name="update_plan",
    description="""
Updates a specific step in the existing plan with new expected results, actions, and current state.
<tool_call>
{
"tool_name": "update_plan",
"parameters": {
    'step_name': {'expected_results': '', 'actions': '', 'where_we_are': ''}
    }
}
</tool_call>
""",
    parameters=[
        ToolParameter(name="step_name", type="object", description="A dictionary where each key is a step name and the value is another dictionary with 'expected_results', 'actions', and 'where_we_are'. Ex. {'step1': {'expected_results': '', 'actions': '', 'where_we_are': ''}}", required=True),
    ],
    execution_function=update_plan,
    update_json_context=True,
    max_count_of_executed_calls=10
)

