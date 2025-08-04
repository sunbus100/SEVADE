from agent.BaseAgent import BaseSarcasmAgent


class SemanticIncongruityAgent(BaseSarcasmAgent):
    def __init__(self, api_key):
        super().__init__(api_key, "SemanticIncongruityAgent", '')

    def build_prompt(self, text, context=None):
        context_str = context if (context and "no web search" not in context.lower()) else "Not available."

        return f"""
        ### Role
        Expert semantic analyst with deep understanding of sentiment and commonsense reasoning, focused on accurate sarcasm detection.
        
        ### Instruction
        You are analyzing the given statement for potential sarcasm based purely on its semantics, emotions, and commonsense reasoning. **You must follow these reasoning steps internally to ensure thorough analysis:**
        
        1.  **Context Awareness:**
            - Review provided Context Summary to establish background.
            - How does the statement's tone align with or contrast against the expected tone derived from the context?
        2.  **Semantic Parsing:**
            - Identify the literal meaning of the statement.
            * Identify the potential implied meaning. Is there a difference? (Note: Direct criticism/negativity alone is **not** necessarily sarcasm).
        3.  **Emotion Analysis:**
            - What emotion is literally expressed (if any)?
            - Is there a contrast between the expressed emotion and the literal meaning of the words?
            - Is there a contrast between the expressed emotion/statement and the emotion expected in the given context?
        4.  **Commonsense Reasoning:**
            - Does the literal statement, taken at face value, align with or contradict common knowledge or logical expectations in the situation described by the context? Explain the alignment or contradiction briefly to yourself.
        5.  **Synthesize for Sarcasm:**
            - Based *only* on the semantic, emotional, and commonsense analysis above, is there a clear **contradiction, inversion of meaning, or sharp emotional mismatch** that strongly suggests the literal meaning is not the intended meaning?
      
        ### Analysis Target
        - **Original Text**: "{text}"
        - **External Context**: {context_str}

        ### Output Format
        Provide your two-part analysis. Respond ONLY with a single-line JSON object:
        {{"PERSPECTIVE STRENGTH": <float from 0.0 (no incongruity) to 1.0 (strong incongruity)>, "EXPLANATION": "<Your expert interpretation, explicitly stating which intent (sarcastic or literal) is more likely and why.>"}}
        """
