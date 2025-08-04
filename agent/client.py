import time
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def call_openai_api(client, input_prompt, retries=3, wait_time=10):
    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": input_prompt,
                }],
                max_tokens=512,
                model='gpt-4o',
                stream=False
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"[call_openai_api] Failed on attempt {attempt + 1}/{retries}: {e}")
            time.sleep(wait_time)

    return "ERROR"




