from bs4 import BeautifulSoup
from agent.client import call_openai_api
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException


class WebSearchAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }

    # No changes to this method
    def _should_i_search(self, text: str) -> bool:

        prompt = f"""
            ### Role
            You are a pragmatic text analyst. Your task is to determine if understanding the following text requires external background knowledge.

            ### Instruction
            Look for specific named entities (e.g., people, organizations, specific events), technical jargon, or references to online trends (like hashtags) that are not self-explanatory.
            - If the text mentions such specific items, it needs a search.
            - If the text is generic, self-contained, or expresses a personal feeling without specific external references, it does not need a search.

            Based on this, does the following text require a web search to be fully understood for sarcasm detection?

            ### Text:
            "{text}"

            ### Your Decision:
            Respond with ONLY the word "Yes" or "No".
            """
        try:
            response = call_openai_api(self.llm_client, prompt, wait_time=5).strip().lower()
            print(f"[WebSearchAgent Decision] Search needed? -> {response}")
            return "yes" in response
        except Exception:
            return True

    def _create_search_query(self, text: str) -> str:
        prompt = f"""
        ### Task
        From "{text}", extract the 1-2 most essential keywords for a web search. If none, respond with "no search".
        """
        response = call_openai_api(self.llm_client, prompt, wait_time=5)
        return response.strip()

    def _summarize_search_results(self, snippets: list) -> str:
        if not snippets:
            return "No relevant search results found."

        search_result_str = "\n".join(f"- {s}" for s in snippets)
        prompt = f"""
        ### Task
        Summarize the key information from the following search results in one sentence:
        {search_result_str}
        """
        response = call_openai_api(self.llm_client, prompt, wait_time=5)
        return response.strip()


    def search_and_summarize(self, text: str) -> str:
        if not self._should_i_search(text):
            return "No web search required."

        search_query = self._create_search_query(text)
        if "no search" in search_query.lower() or not search_query.strip():
            return "No background knowledge retrieved."

        print(f"Generating Search Keywords: {search_query}")


        driver = None
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={search_query.replace(' ', '+')}"

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")

            DRIVER_PATH = './drivers/chromedriver.exe'
            service = Service(executable_path=DRIVER_PATH)

            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(30)

            try:
                driver.get(search_url)
            except TimeoutException:
                print(f"Pages loading timeout: {search_url}")
                return "External search failed due to page load timeout."

            page_html = driver.page_source
            soup = BeautifulSoup(page_html, "html.parser")
            result_containers = soup.find_all("div", class_="result")
            if not result_containers:
                return "No relevant search results found."
            snippets = [c.find("a", class_="result__snippet").get_text(strip=True) for c in result_containers[:3] if
                        c.find("a", class_="result__snippet")]
            if not snippets:
                return "No relevant search results found."
            return self._summarize_search_results(snippets)

        except Exception as e:
            return f"Fail while searching."
        finally:
            if driver:
                driver.quit()
