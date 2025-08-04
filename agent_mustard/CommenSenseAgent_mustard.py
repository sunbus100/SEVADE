from agent_mustard.BaseAgent_mustard import BaseSarcasmAgent_mustard


class CommonSenseViolationAgent_mustard(BaseSarcasmAgent_mustard):

    def __init__(self, api_key):
        super().__init__(api_key, "CommonSenseViolationAgent_mustard", '')

    def build_prompt(self, text, web_context=None, utterance_context=None):
        web_context_str = web_context if (web_context and "no web search" not in web_context.lower()) else "Not available."
        utterance_context_str = utterance_context if utterance_context else "No direct utterance context provided."

        return f"""
        ### Role
        Expert commonsense analyst for sarcasm detection—be highly cautious: only *extreme* and *obvious* violations of common sense support sarcasm.

        ### Instruction
        Evaluate the statement's degree of commonsense violation using this strict internal checklist:

        1. **Obviousness Check:**
            - Is the statement, taken literally, clearly impossible, absurd, or universally recognized as false by any reasonable adult?
            - Is the violation so blatant that *no reasonable explanation* could make it literal?
            - If the statement could be interpreted as a joke, exaggeration, or is plausible in any context, do **not** treat it as a clear commonsense violation.
        2. **Specialized Knowledge Filter:**
            - Ignore anything requiring expert knowledge to judge. Only consider violations that an average adult would spot instantly.
        3. **Intent Check:**
            - Does the statement seem designed to mock or challenge commonsense on purpose (sarcastic intent)? Or is it just hyperbole, error, or confusion?
        4. **Synthesize Judgment:**
            - If, after all steps, the violation is *not* extreme and unmistakable, lean toward a LOW score.

        ### Analysis Target
        - **Utterance Context (The Conversation So Far)**: {utterance_context_str}
        - **Original Text**: "{text}"

        ### Output Format
        Provide a commonsense violation score ONLY if the violation is *extreme, obvious, and serves a clear mocking purpose*. Respond ONLY with a single-line JSON object:
        {{"PERSPECTIVE STRENGTH": <float from 0.0 (no violation) to 1.0 (strong, clear violation)>, "EXPLANATION": "<State the violated principle and explain why it is definitely intended to mock something in the context, rather than just being an absurd joke.>"}}
        """
