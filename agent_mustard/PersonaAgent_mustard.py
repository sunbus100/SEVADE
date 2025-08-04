from agent_mustard.BaseAgent_mustard import BaseSarcasmAgent_mustard


class PersonaConflictAgent_mustard(BaseSarcasmAgent_mustard):

    def __init__(self, api_key):
        super().__init__(api_key, "PersonaConflictAgent_mustard", '')

    def build_prompt(self, text, web_context=None, utterance_context=None):
        context_str = web_context if (web_context and "no web search" not in web_context.lower()) else "Not available."
        utterance_context_str = utterance_context if utterance_context else "No direct utterance context provided."

        return f"""
        ### Role
        Expert persona conflict analyst—detect sarcasm only if the projected persona and actual statement are in *sharp, undeniable conflict*. Do not overinterpret ambiguous or playful inconsistencies.

        ### Instruction
        Apply this strict checklist:

        1. **Persona Identification:**
            - What persona, stance, or self-image does the speaker project in the statement?
        2. **Statement Consistency:**
            - Does any part of the statement *strongly* contradict the projected persona?
            - Or are inconsistencies subtle, explainable as humor, or within normal conversational range?
        3. **Context Check:**
            - Use external context only to clarify well-known personas. If context is weak, rely on textual evidence.
        4. **Synthesize Judgment:**
            - Only if the persona conflict is *clear, strong, and without reasonable alternative reading*, assign a high score.
            - For playful, weak, or ambiguous inconsistencies, lean toward a LOW score.

        ### Analysis Target
        - **Utterance Context (The Conversation So Far)**: {utterance_context_str}
        - **Original Text**: "{text}"
        - **External Context**: {context_str}

        ### Output Format
        Provide a score for *persona conflict only*. Respond ONLY with a single-line JSON object:
        {{"PERSPECTIVE STRENGTH": <float from 0.0 (no conflict) to 1.0 (clear, strong conflict)>, "EXPLANATION": "<Briefly state the persona and the conflicting statement.>"}}
        """
