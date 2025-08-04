import pandas as pd
import time
import re
from tqdm import tqdm
import logging
import concurrent.futures
import openai

from agent_mustard.ControllerAgent_mustard import ControllerAgent_mustard
from agent_mustard.CommenSenseAgent_mustard import CommonSenseViolationAgent_mustard
from agent_mustard.PersonaAgent_mustard import PersonaConflictAgent_mustard
from agent_mustard.EmotionAgent_mustard import EmotionPolarityInverterAgent_mustard
from agent_mustard.RhetoricalAgent_mustard import RhetoricalDeviceAgent_mustard
from agent_mustard.PragmaticAgent_mustard import PragmaticContrastAgent_mustard
from agent_mustard.SemanticAgent_mustard import SemanticIncongruityAgent_mustard
from agent.SummarizeAgent import SummarizationAgent
from agent.WebSearchAgent import WebSearchAgent
from agent.utils import eval_performance


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def process_row(i, row, api_keys, agent_classes, controller_params):
    api_key = api_keys[i % len(api_keys)]
    try:
        llm_client = openai.OpenAI(
            api_key=api_key,
            base_url="",
            default_query={"api-version": "preview"},
            timeout=30.0
        )

        summarization_agent = SummarizationAgent(api_key=api_key)
        web_search_agent = WebSearchAgent(llm_client=llm_client)

        controller = ControllerAgent_mustard(
            api_key=api_key,
            agent_classes=agent_classes,
            summarization_agent=summarization_agent,
            web_search_agent=web_search_agent,
            llm_client=llm_client,
            **controller_params
        )

        text = row['Text']
        utterance_context = row.get('Context', None)

        result = controller.analyze(text, utterance_context=utterance_context)

        final_decision = result.get('final_decision', 'UNCERTAIN')
        if re.search(r"not", str(final_decision), re.IGNORECASE):
            label = 0
        elif re.search(r"sarcastic", str(final_decision), re.IGNORECASE):
            label = 1
        else:
            label = -1

        return {
            **row,
            'labels': label,
            'final_decision': final_decision,
            'rounds': result.get('rounds_completed', -1),
            'outputs': str(result.get('outputs', {})),
            'summary_sentence': result.get('summary_sentence', 'NO SUMMARY')
        }
    except Exception as e:
        logger.error("Error on row %s: %s", i, str(e))
        return {
            **row,
            'labels': -1,
            'final_decision': 'ERROR',
            'rounds': -1,
            'outputs': str(e),
            'summary_sentence': 'ERROR'
        }

api = ''

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Running multi-agent strategy for sarcasm detection on Mustard dataset.')
    parser.add_argument('--dataset_path', type=str, default='datasets/sarcasm')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--metric_path', type=str, default='output')
    parser.add_argument('--task_name', type=str, default='mustard')
    parser.add_argument('--api_keys', type=str, default=api)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    task_name = args.task_name
    time_now = time.time()
    dataset_path = f'{args.dataset_path}/test_{task_name}.csv'
    output_path = f'{args.output_path}/output_{task_name}_{time_now}.csv'
    metric_path = f'{args.metric_path}/metric_{task_name}_{time_now}.json'
    api_keys = [k.strip() for k in args.api_keys.split(',') if k.strip()]
    num_workers = args.workers

    logger.info(f"Using {num_workers} threads with {len(api_keys)} API keys for Mustard dataset.")

    df = pd.read_csv(dataset_path, encoding_errors='ignore')
    df.dropna(subset=['Text', 'Context'], inplace=True)

    agent_classes = {
        "SemanticIncongruityAgent": SemanticIncongruityAgent_mustard,
        "PragmaticContrastAgent": PragmaticContrastAgent_mustard,
        "RhetoricalDeviceAgent": RhetoricalDeviceAgent_mustard,
        "EmotionPolarityInverterAgent": EmotionPolarityInverterAgent_mustard,
        "CommonSenseViolationAgent": CommonSenseViolationAgent_mustard,
        "PersonaConflictAgent": PersonaConflictAgent_mustard
    }
    controller_params = {'n_initial': 3, 'max_rounds': 3, 'vote_threshold': 0.5}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_row, i, row, api_keys, agent_classes, controller_params) for i, row in
                   df.iterrows()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Mustard"):
            try:
                results.append(future.result(timeout=120))
            except Exception as e:
                logger.error("A row task failed with timeout or other error: %s", e)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    try:
        df_eval = out_df.copy()
        mask = df_eval["summary_sentence"].astype(str).str.lower().isin(
            ["error", "no summary", "client_init_error", "error_max_retries"])
        df_eval = df_eval[~mask]
        if 'Label' not in df_eval.columns:
            logger.warning("'Label' column not found in output. Skipping evaluation.")
        elif df_eval.empty:
            logger.warning("No valid rows for evaluation.")
        else:
            y_true = df_eval['Label'].values
            y_pred = df_eval['labels'].values
            eval_performance(y_true, y_pred, metric_path=metric_path)
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")

    logger.info(f"All done. Final results saved to: {output_path}")
