from agent.BaseAgent import BaseSarcasmAgent
from agent.utils import parse_llm_output_json_summarize
from agent.client import call_openai_api


class SummarizationAgent(BaseSarcasmAgent):
    def __init__(self, api_key):
        suggestion = """
        Checklist for a good summary:
        1. Does the summary accurately reflect the strongest signals from the expert agents?
        2. Is the summary a single, grammatically correct, and logically coherent sentence?
        3. Does it clearly explain the primary reasons for the sarcasm (or lack thereof), integrating key perspectives like semantic incongruity, emotional polarity, or rhetorical devices?
        """
        super().__init__(api_key, "SummarizationAgent", suggestion)

    def build_prompt(self, agent_outputs: dict, original_text: str):
        analysis_summary = "\n".join(
            [f"- {agent_name}: [Strength: {data.get('strength')}] {data.get('explanation', 'No explanation.')}"
             for agent_name, data in agent_outputs.items()]
        )

        return f"""
        ### Role
        You are a meticulous Lead Analyst. Your task is to synthesize findings from a panel of expert agents into a structured, evidence-based summary. This summary will serve as high-quality training data for a smaller student model.

        ### Context
        A panel of expert agents has analyzed the following text:
        - Original Text: "{original_text}"

        Their individual findings are as follows:
        {analysis_summary}

        ### Instruction
        Your primary goal is to create a neutral, detailed, and structured analytical summary based **only** on the evidence provided by the agents.
        1.  **Synthesize, do not judge:** Do NOT add your own "sarcastic" or "not sarcastic" conclusion. Your role is to present the evidence coherently.
        2.  **Follow the structure:** Adhere strictly to the "Summary Structure" provided below. Use the exact headings.
        3.  **Be concise:** The entire summary should be a brief paragraph, ideally under 120 words.

        ### Summary Structure
        You must format your summary using the following three key points:
        - **Overall Assessment:** A high-level sentence summarizing the general consensus or **the main points of disagreement**. If the findings are mixed or conflicting, state that clearly (e.g., "The agents' analyses are divided, pointing to ambiguity in the text.").
        - **Primary Evidence:** Detail the strongest 1-2 pieces of evidence that support the main findings (this could include evidence for both sides if there is a conflict).
        - **Secondary/Conflicting Signals:** Briefly mention any weaker signals or specific agent findings that create the conflict or offer a balanced view. If there are no conflicts and all signals are strong, state "No significant conflicting signals were found."

        ### Example of a perfect output (for a conflicting case)
        {{
            "summary_sentence": "Overall Assessment: The agents' analyses suggest a moderate likelihood of sarcasm, though the evidence is conflicting. Primary Evidence: The Semantic Incongruity Agent highlighted a sharp mismatch between the text's literal meaning and the context. Secondary/Conflicting Signals: However, the Pragmatic Contrast Agent found the speaker's quirky tone could be literal, and the Emotion Agent detected no emotional inversion, creating significant ambiguity."
        }}

        ### Your Output Format
        Respond ONLY with a single valid JSON object, with exactly the following key:
        {{"summary_sentence": "<Your structured analytical summary>"}}
        Do NOT insert line breaks or markdown formatting inside the summary_sentence string.
        """

    def summarize(self, agent_outputs: dict, original_text: str):
        prompt = self.build_prompt(agent_outputs, original_text)

        response = call_openai_api(self.client, prompt)

        full_response = response.strip()

        result = parse_llm_output_json_summarize(full_response)
        return result