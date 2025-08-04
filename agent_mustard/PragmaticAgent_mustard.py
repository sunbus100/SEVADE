from agent_mustard.BaseAgent_mustard import BaseSarcasmAgent_mustard


class PragmaticContrastAgent_mustard(BaseSarcasmAgent_mustard):

    def __init__(self, api_key):
        super().__init__(api_key, "PragmaticContrastAgent_mustard", '')

    def build_prompt(self, text, web_context=None, utterance_context=None):
        web_context_str = web_context if (web_context and "no web search" not in web_context.lower()) else "Not available."
        utterance_context_str = utterance_context if utterance_context else "No direct utterance context provided."

        return f"""
        ### Role
        Expert pragmatic analyst—identify sarcasm through *clear, strong* mismatches between situation and language style. Be highly cautious: style contrast alone is **not** enough unless it forces a non-literal reading.

        ### Instruction
        Follow this checklist for nuanced analysis:

        1. **Situation Assessment:**
            - What is the seriousness or context of the described event?
        2. **Style Analysis:**
            - What is the linguistic style (formal, informal, grandiose, etc.)?
        3. **Mismatch Evaluation:**
            - Is there a *clear and jarring* mismatch, making a literal reading implausible?
            - Or could the style mismatch reflect genuine emotion, emphasis, or idiosyncratic speech?
        4. **Sarcasm Likelihood:**
            - Only if the style-situation mismatch *cannot* be reasonably explained literally and points strongly to sarcasm, assign a high score.
            - If alternative explanations are plausible, prefer a LOW score.

        ### Analysis Target
        - **Utterance Context (The Conversation So Far)**: {utterance_context_str}
        - **Original Text**: "{text}"
        - **External Context**: {web_context_str}

        ### Output Format
        Provide a score for *pragmatic contrast only*. Respond ONLY with a single-line JSON object:
        {{"PERSPECTIVE STRENGTH": <float from 0.0 (no mismatch) to 1.0 (clear, strong mismatch)>, "EXPLANATION": "<Describe the mismatch, and then explain why a mocking intent is the ONLY plausible reason, ruling out other explanations like quirkiness or simple humor.>"}}
        """
