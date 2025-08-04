from agent.BaseAgent import BaseSarcasmAgent


class RhetoricalDeviceAgent(BaseSarcasmAgent):
    def __init__(self, api_key):
        super().__init__(api_key, "RhetoricalDeviceAgent", '')

    def build_prompt(self, text, context=None):
        # This agent does not typically need external context.
        context_str = context if (context and "no web search" not in context.lower()) else "Not available."

        return f"""
        ### Role
        You are an expert rhetorical analyst identifying sarcasm *strictly* through rhetorical cues. Be highly cautious; direct negativity without rhetorical contradiction is **not** sarcasm.

        ### Instruction
        Analyze the statement for sarcasm based *only* on its rhetorical features. **You must use this checklist internally for a thorough analysis:**

        1.  **Common Devices Check:**
            * **Irony:** Is the literal meaning clearly the *opposite* of the intended meaning conveyed through tone or context implied by rhetoric?
            * **Hyperbole:** Is exaggeration used? If yes, does this exaggeration create **absurdity, mockery, or a clear contradiction** with reality/expectations (indicative of sarcasm), OR is it just for emphasis (not sarcasm)?
            * **Metaphor/Simile:** Is a comparison used? If yes, does the comparison create a **sharply contrasting or mocking tone**, suggesting a non-literal sarcastic intent?
            * **Understatement (Litotes):** Is something expressed weakly to imply the opposite strongly (e.g., "He's not the sharpest tool in the shed")? Does it create ironic contrast?
            * **Contrast/Juxtaposition:** Are opposing ideas/images placed together? Does this create an **ironic or mocking effect**?
        2.  **Subtle Devices Check:**
            * **Sarcastic Question:** Is a question asked where the answer is obviously the opposite, used to mock or criticize?
            * **Sarcastic Hypothetical/Analogy:** Is an absurd or far-fetched scenario/comparison presented to mock the real situation?
        3.  **Overall Rhetorical Impact:**
            * Considering any devices found, do they collectively **force a non-literal interpretation** that is clearly mocking or contradictory?
            * Or, despite potential devices, does the overall rhetorical effect remain compatible with a literal, non-sarcastic reading or simple emphasis/negativity? If the rhetorical reading isn't *clearly* sarcastic, **lean NO.**

        ### Analysis Target
        - **Original Text**: "{text}"

        ### Output Format
        Provide a score for the presence of **sarcasm-related Rhetorical Devices only**. Respond ONLY with a single-line JSON object:
        {{"PERSPECTIVE STRENGTH": <float from 0.0 (no relevant device) to 1.0 (a clear, strong device)>,"EXPLANATION": "<A 1-sentence explanation naming the specific device found (e.g., Hyperbole, Rhetorical Question).>"}}
        """
