import openai
from agent.client import call_openai_api
from agent.utils import parse_llm_output_json


class BaseSarcasmAgent_mustard:
    def __init__(self, api_key, agent_name, prompt_suggestion=""):
        self.api_key = api_key
        self.agent_name = agent_name
        self.prompt_suggestion = prompt_suggestion
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="",
            default_query={"api-version": "preview"},
            timeout=60.0
        )

    def build_prompt(self, text, web_context=None, utterance_context=None):
        raise NotImplementedError("Each agent must implement build_prompt()!")

    def analyze(self, text, web_context=None, utterance_context=None):

        prompt = self.build_prompt(text, web_context, utterance_context)
        response = call_openai_api(self.client, prompt)
        full_response = response.strip()

        # The parsing utility remains the same
        result = parse_llm_output_json(full_response)
        return result
