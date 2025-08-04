import pandas as pd
import time
import re
from tqdm import tqdm
import logging
import concurrent.futures
from agent.ControllerAgent import ControllerAgent
from agent.CommenSenseAgent import CommonSenseViolationAgent
from agent.PersonaAgent import PersonaConflictAgent
from agent.EmotionAgent import EmotionPolarityInverterAgent
from agent.RhetoricalAgent import RhetoricalDeviceAgent
from agent.SemanticAgent import SemanticIncongruityAgent
from agent.PragmaticAgent import PragmaticContrastAgent
from agent.SummarizeAgent import SummarizationAgent
from agent.utils import eval_performance
from agent.WebSearchAgent import WebSearchAgent
import openai

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
        controller = ControllerAgent(
            api_key=api_key,
            agent_classes=agent_classes,
            summarization_agent=summarization_agent,
            web_search_agent=web_search_agent,
            llm_client=llm_client,
            **controller_params
        )
        text = row['Text']
        result = controller.analyze(text)
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
            'rounds': result.get('rounds', -1),
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
    parser = argparse.ArgumentParser(description='Running multi_agent strategy for sarcasm detection.')
    parser.add_argument('--dataset_path', type=str, default='datasets/sarcasm')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--metric_path', type=str, default='output')
    parser.add_argument('--task_name', type=str, default='iacv1')
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

    logger.info(f"Using {num_workers} threads with {len(api_keys)} API keys.")

    df = pd.read_csv(dataset_path, encoding_errors='ignore')
    df.dropna(inplace=True)


    agent_classes = {
        "SemanticIncongruityAgent": SemanticIncongruityAgent,
        "PragmaticContrastAgent": PragmaticContrastAgent,
        "RhetoricalDeviceAgent": RhetoricalDeviceAgent,
        "EmotionPolarityInverterAgent": EmotionPolarityInverterAgent,
        "CommonSenseViolationAgent": CommonSenseViolationAgent,
        "PersonaConflictAgent": PersonaConflictAgent
    }
    controller_params = {'n_initial': 3,
                         'max_rounds': 3}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, row in tqdm(list(df.iterrows()), total=len(df), desc="Submitting tasks"):
            futures.append(executor.submit(process_row, i, row, api_keys, agent_classes, controller_params))
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as e:
                logger.error("A row task failed: %s", e)

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