import openai
from agent.client import call_openai_api
from agent.utils import parse_llm_output_json


class BaseSarcasmAgent:
    def __init__(self, api_key, agent_name, prompt_suggestion):
        self.api_key = api_key
        self.agent_name = agent_name
        self.prompt_suggestion = prompt_suggestion
        self.client = openai.OpenAI(
        api_key=api_key,
        base_url="",
        default_query={"api-version": "preview"},
        timeout=60.0
        )

    def _build_context_section(self, context: str) -> str:
        if context and "no web search" not in context.lower() and "no background knowledge" not in context.lower():
            return f"""
            ### External Context
            The following background knowledge was retrieved from a web search. Use this information to make a more accurate and informed judgment from your specific perspective.
            ---
            {context}
            ---
            """
        return ""

    def build_prompt(self, text, context=None):
        raise NotImplementedError("Each agent must implement build_prompt()!")

    def analyze(self, text, context=None):
        prompt = self.build_prompt(text, context)
        response = call_openai_api(self.client, prompt)
        full_response = response.strip()

        result = parse_llm_output_json(full_response)
        return result
