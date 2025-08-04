import json
import random
from agent.client import call_openai_api
from agent.utils import parse_llm_output_json_unfied


class ControllerAgent_mustard:

    def __init__(self, api_key, agent_classes, summarization_agent, web_search_agent, n_initial=3, max_rounds=3,
                 vote_threshold=0.5, llm_client=None):
        self.api_key = api_key
        self.agents = {name: cls(api_key) for name, cls in agent_classes.items()}
        self.agent_list = list(self.agents.keys())
        self.n_initial = n_initial
        self.max_rounds = max_rounds
        self.llm_client = llm_client
        self.summarization_agent = summarization_agent
        self.web_search_agent = web_search_agent
        self.vote_threshold = vote_threshold
        self.agent_descriptions = {
            "SemanticIncongruityAgent": "Detects mismatch between literal meaning and context/world knowledge.",
            "PragmaticContrastAgent": "Analyzes violation of expressive conventions for a given situation.",
            "RhetoricalDeviceAgent": "Identifies use of irony, hyperbole, and other rhetorical figures.",
            "EmotionPolarityInverterAgent": "Checks if expressed emotion contradicts the expected sentiment.",
            "CommonSenseViolationAgent": "Assesses if the statement contradicts common sense or logic.",
            "PersonaConflictAgent": "Looks for internal conflicts in the speaker's projected persona."
        }

    def _select_initial_agents_dynamically(self, text: str) -> list:

        print("--- [Pre-Analysis: Dynamically selecting initial agents] ---")
        agent_options_str = "\n".join([f"- {name}: {desc}" for name, desc in self.agent_descriptions.items()])

        prompt = f"""
        ### Role
        You are a highly efficient text analysis dispatcher. Your job is to read an input text and select the most promising perspectives (agents) for sarcasm detection.

        ### Available Analysis Perspectives (Agents):
        {agent_options_str}

        ### Task
        Read the following input text. Based on its content, which **{self.n_initial}** perspectives are the MOST relevant to activate first to determine if the text is sarcastic?

        ### Input Text:
        "{text}"

        ### Output Format
        Respond ONLY with a comma-separated list of the {self.n_initial} most relevant agent names from the list above. Do not add any other text or explanation.
        Example: PragmaticContrastAgent,EmotionPolarityInverterAgent,RhetoricalDeviceAgent
        """
        try:
            response = call_openai_api(self.llm_client, prompt)
            selected_names = [name.strip() for name in response.split(',')]
            valid_selected_agents = [name for name in selected_names if name in self.agent_list]
            if len(valid_selected_agents) >= self.n_initial:
                return valid_selected_agents[:self.n_initial]
            else:
                print(
                    "Warning: Initial agent selection returned insufficient valid agents. Falling back to random selection.")
                return random.sample(self.agent_list, self.n_initial)
        except Exception as e:
            print(f"Error during initial agent selection: {e}. Falling back to random selection.")
            return random.sample(self.agent_list, self.n_initial)

    def _run_debate_round(self, text: str, current_outputs: dict, web_context: str, utterance_context: str) -> dict:

        print(f"--- [Debate Phase] Agents re-evaluating based on peer feedback ---")

        if not current_outputs:
            return current_outputs

        evidence_report = "\n".join(
            [f"- {name} reading: {result.get('strength', 0.0):.2f}. Reason: {result.get('explanation', 'N/A')}"
             for name, result in current_outputs.items() if result]
        )

        agent_to_rethink = min(current_outputs.keys(),
                               key=lambda k: abs(current_outputs.get(k, {}).get('strength', 0.5) - 0.5))

        debate_prompt = f"""
        ### Role
        You are the {agent_to_rethink}. You are participating in a panel discussion to analyze a text for sarcasm.

        ### Utterance Context "{utterance_context}" 
        ### Original Text: "{text}"
        ### External Context: {web_context}
        
        ### Your Initial Analysis:
        - Strength: {current_outputs.get(agent_to_rethink, {}).get('strength')}
        - Explanation: {current_outputs.get(agent_to_rethink, {}).get('explanation')}

        ### Your Colleagues' Analyses (The Debate):
        Here is what your fellow agents concluded. You must consider their perspectives.
        {evidence_report}

        ### Your Task: Re-evaluate and Refine
        Given your colleagues' findings, please re-evaluate the original text.
        1.  **Acknowledge Conflict/Synergy:** Does their evidence support or contradict your initial view? Explicitly state the key point of synergy or disagreement.
        2.  **Refine Your Reasoning:** Based on this new information, refine your original explanation. Explain HOW your perspective contributes to a more unified final conclusion.
        3.  **Provide an Updated Score:** Output a potentially revised "PERSPECTIVE STRENGTH" score and a new, more nuanced "EXPLANATION".

        ### Output Format
        Respond ONLY with a single-line JSON object with your updated analysis:
        {{"PERSPECTIVE STRENGTH": <float>, "EXPLANATION": "<Your new, refined explanation that incorporates the debate.>"}}
        """

        try:
            rethought_response = call_openai_api(self.agents[agent_to_rethink].client, debate_prompt)

            if rethought_response and rethought_response.strip().startswith('{'):
                updated_result = json.loads(rethought_response)
                current_outputs[agent_to_rethink] = {
                    "strength": updated_result.get("PERSPECTIVE STRENGTH"),
                    "explanation": updated_result.get("EXPLANATION")
                }
            else:
                print(
                    f"Warning: Received invalid or empty response from API for {agent_to_rethink} during debate. Skipping update for this agent.")

        except (json.JSONDecodeError, AttributeError) as e:
            print(
                f"Error during debate re-evaluation for {agent_to_rethink}: {e}. Raw response was: '{rethought_response}'")

        return current_outputs

    def _is_reinforcement_needed(self, text: str, post_debate_explanations: dict, utterance_context: str) -> bool:

        print("--- [Gating Decision] Assessing if reinforcement is necessary ---")

        explanations_str = "\n".join(
            [f"- {name}: {exp}" for name, exp in post_debate_explanations.items()]
        )

        prompt = f"""
        You are a pragmatic meta-controller. A team of agents has just debated their analysis of a text.

        ### Utterance context and Text:
        Utterance Context: "{utterance_context}"
        Text: "{text}"

        ### Their Post-Debate Conclusions:
        {explanations_str}

        ### Your Task:
        Based on their conclusions, is the analysis still clearly incomplete, contradictory, or stuck?
        - If YES, the current team is insufficient and needs a new perspective.
        - If NO, the current team's analysis is coherent enough to proceed to a final decision.

        ### Output Format:
        Respond ONLY with a single valid JSON object. The key must be "decision" and the value must be either "Yes" or "No".
        {{"decision": <yes/no>}}
        """
        try:
            response_str = call_openai_api(self.llm_client, prompt)
            decision = parse_llm_output_json_unfied(response_str)
            print(f"Controller decision on needing reinforcement: {decision}")
            return "yes" in decision
        except (json.JSONDecodeError, AttributeError):
            print(f"Warning: Failed to parse reinforcement decision. Defaulting to 'No'. Raw response: {response_str}")
            return False

    def llm_select_most_complementary(self, current_agents: list, candidates: list, text: str, explanations: dict,
                                      utterance_context: str) -> str:

        prompt = f"""
        You are a meta-reasoning assistant. The current sarcasm detection system has activated the following perspectives and they have already debated their initial findings:
        - Active Agents: {', '.join(current_agents)}
        - Their post-debate explanations are:{chr(10).join([f"- {name}: {exp}" for name, exp in explanations.items()])}
        The analysis seems to have reached a conflict or a dead end. To resolve this, consider the available candidate perspectives that are NOT yet active:
        - Candidate Agents: {', '.join(candidates)}

        Which **single candidate** would best resolve the current conflict or fill the biggest gap?
        Utterance context: "{utterance_context}"
        Sentence: "{text}"

        ### Output Format
        Only output the single best agent name from the candidate list.
        """
        try:
            agent_name = call_openai_api(self.llm_client, prompt).strip()
            return agent_name if agent_name != "None" else None
        except Exception:
            return random.choice(candidates) if candidates else None

    def _make_final_decision_by_vote(self, agg_outputs: dict) -> dict:
        '''
        This function is used to directly give the final prediction without using the finetune BERT.
        '''
        if not agg_outputs:
            return {"decision": "NOT SARCASTIC", "reasoning": "No valid agent outputs to analyze."}

        sarcastic_votes = sum(1 for result in agg_outputs.values() if result.get('strength', 0) > self.vote_threshold)
        literal_votes = len(agg_outputs) - sarcastic_votes

        decision = "SARCASTIC" if sarcastic_votes > literal_votes else "NOT SARCASTIC"
        reasoning = f"Rule-based decision: SARCASTIC votes ({sarcastic_votes}) vs. NOT SARCASTIC votes ({literal_votes})."
        return {"decision": decision, "unified_reasoning": reasoning}

    def analyze(self, text: str, utterance_context: str = None) -> dict:
        activated_agents = set()
        outputs = {}
        round_count = 0

        web_context = self.web_search_agent.search_and_summarize(text)

        initial_agents = self._select_initial_agents_dynamically(text)
        print(f"\n--- [Initial Analysis Round] Activating: {', '.join(initial_agents)} ---")
        for name in initial_agents:
            if name in self.agents:
                outputs[name] = self.agents[name].analyze(text, web_context=web_context,
                                                          utterance_context=utterance_context)
                activated_agents.add(name)

        for round_count in range(self.max_rounds):
            print(f"\n M-AI-CO Start Round {round_count + 1}/{self.max_rounds} ".center(50, "="))

            if len(activated_agents) > 1:
                outputs = self._run_debate_round(text, outputs, web_context=web_context,
                                                 utterance_context=utterance_context)

            post_debate_explanations = {name: out.get("explanation", "") for name, out in outputs.items() if out}

            if not self._is_reinforcement_needed(text, post_debate_explanations, utterance_context):
                print("Controller concluded that the current agent team is sufficient. Ending evolution loop.")
                break

            candidate_pool = [name for name in self.agent_list if name not in activated_agents]
            if not candidate_pool:
                print("No more agents available to add. Ending evolution loop.")
                break

            next_agent_to_add = self.llm_select_most_complementary(
                current_agents=list(activated_agents), candidates=candidate_pool, text=text,
                explanations=post_debate_explanations, utterance_context=utterance_context
            )

            if next_agent_to_add and next_agent_to_add in self.agents:
                print(f"--- [Reinforcement Action] Controller is adding new agent: **{next_agent_to_add}** ---")
                outputs[next_agent_to_add] = self.agents[next_agent_to_add].analyze(text, web_context=web_context,
                                                                                    utterance_context=utterance_context)
                activated_agents.add(next_agent_to_add)
            else:
                print(f"Selection process did not yield a valid agent to add. Ending evolution loop.")
                break

        print("\n--- [Final Synthesis] Making decision and summarizing ---")
        final_agg_outputs = {name: out for name, out in outputs.items() if out and out.get('strength') is not None}
        final_decision_data = self._make_final_decision_by_vote(final_agg_outputs)

        summary_sentence = self.summarization_agent.summarize(
            agent_outputs=final_agg_outputs, original_text=text).get("summarization")

        return {
            "final_decision": final_decision_data.get("decision"),
            "unified_reasoning": final_decision_data.get("unified_reasoning"),
            "summary_sentence": summary_sentence,
            "outputs": final_agg_outputs,
            "activated_agents": list(activated_agents),
            "rounds_completed": round_count + 1
        }
