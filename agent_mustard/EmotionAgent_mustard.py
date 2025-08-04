from agent_mustard.BaseAgent_mustard import BaseSarcasmAgent_mustard


class EmotionPolarityInverterAgent_mustard(BaseSarcasmAgent_mustard):

    def __init__(self, api_key):
        super().__init__(api_key, "EmotionPolarityInverterAgent_mustard", '')

    def build_prompt(self, text, web_context=None, utterance_context=None):
        context_str = web_context if (web_context and "no web search" not in web_context.lower()) else "Not available."
        utterance_context_str = utterance_context if utterance_context else "No direct utterance context provided."

        return f"""
        ### Role
        You are a precision analysis tool that functions as an **Emotion Polarity Meter**.

        ### Measurement Protocol
        Your sole task is to measure the degree of contradiction between the surface sentiment of the words used in the "Original Text" and the objective sentiment of the situation being described. Use the "External Context" to understand the reality of the situation. **Only assign a high score (above 0.5) if the polarity inversion is clear, strong, and justified by both text and context. If the evidence is weak or ambiguous, assign a low score (below 0.5).**

        ### Strict Operational Rules
        1.  **Strict Criteria:** Only measure strong, obvious inversions (e.g., positive words in clearly negative situations). If the inversion is subtle, ambiguous, or open to interpretation, be conservative and rate low.
        2.  **Ignore Pure Emotion:** If the text is simply emotional (angry, happy) but not inverted, rate as 0.0.
        3.  **Err on the Side of Caution:** When in doubt, lower your score.

        ### Analysis Target
        - **Utterance Context (The Conversation So Far)**: {utterance_context_str}
        - **Original Text**: "{text}"
        - **External Context**: {context_str}

        ### Output Format
        Provide the measurement for the **Emotion Polarity Inversion feature only**. Respond ONLY with a single-line JSON object:
        {{"PERSPECTIVE STRENGTH": <float from 0.0 (no inversion) to 1.0 (strong inversion)>,"EXPLANATION": "<A 1-sentence explanation citing the positive/negative words and the contradictory negative/positive situation.>"}}
        """
