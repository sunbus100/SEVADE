from agent.BaseAgent import BaseSarcasmAgent


class PragmaticContrastAgent(BaseSarcasmAgent):
    def __init__(self, api_key):
        super().__init__(api_key, "PragmaticContrastAgent", '')

    def build_prompt(self, text, context=None):
        context_str = context if (context and "no web search" not in context.lower()) else "Not available."

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
        - **Original Text**: "{text}"
        - **External Context**: {context_str}

        ### Output Format
        Provide a score for *pragmatic contrast only*. Respond ONLY with a single-line JSON object:
        {{"PERSPECTIVE STRENGTH": <float from 0.0 (no mismatch) to 1.0 (clear, strong mismatch)>, "EXPLANATION": "<Briefly describe the mismatch and why it suggests sarcasm or not.>"}}
        """

