import time
import random
import logging
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai.types import Part, Content, GenerateContentConfig, ThinkingConfig
from google.genai.errors import ClientError, ServerError
from collections import deque
from chatwhatcartobuy.utils.getpath import get_path
from chatwhatcartobuy.utils.txtparser import read_txt_file
from chatwhatcartobuy.rag.retriever import Retriever

logger = logging.getLogger(__name__)
    
class ChatBot:
    
    SYSTEM_PROMPT = read_txt_file(get_path(start_path=__file__, target='system_prompt.txt', subdir='llm'))
    
    def __init__(self, retriever: Retriever, model_name: str='gemini-2.5-flash', thinking_budget=1024):
        # client requires GEMINI_API_KEY env variable; Load .env file at the start of application.
        self.client = genai.Client()
        self.retriever = retriever
        self.model_name = model_name
        self._session_history = [] # Content storage
        self._thinking_budget = thinking_budget
        self._requests_window = deque()
        logging.debug('Initialized ChatBot with model=%s and thinking_budget%s=', model_name, self._thinking_budget)
    
    def _retrieve_context(self, query: str):
        rag_context = self.retriever.retrieve(query)
        logging.debug('Retrieved context from database. Approx %d tokens.', len(rag_context)//4)
        return rag_context
    
    # def _cache_contents(self, contents: List[Content]):
    #     # Potential cost savings from caching system prompt and subsequent responses
    #     self.cache = self.client.caches.create(
    #         model=self.model_name,
    #         config=CreateCachedContentConfig(
    #             contents=contents,
    #             system_instruction=self.SYSTEM_PROMPT,
    #             ttl='600s'
    #         )
    #     )
    #     return self
    
    def _get_response(self, query):
        logger.debug('Received query with approx. %d tokens.', len(query) // 4)
        # Track conversation history from user and model
        self._session_history.append(Content(role='user', parts=[Part.from_text(text=query)]))
        # Check if conversation history needs to be trimmed
        self._limit_input_tokens()
        response = self.client.models.generate_content(
            model=self.model_name, 
            contents=self._session_history,
            config=GenerateContentConfig(
                # Allow model to think; improves answer quality
                thinking_config=ThinkingConfig(thinking_budget=self._thinking_budget)
                )
            )
        self._session_history.append(Content(role='model', parts=[Part.from_text(text=response.text)]))
        return response.text
    
    def retrieve_context(self, query):
        rag_context = self._retrieve_context(query)
        logger.debug('Retrieved %d documents from vector store, with approximately %d tokens.', len(rag_context.split('\n\n')), len(rag_context) // 4)
        return rag_context
    
    def chat(self, query, retries=5, backoff=1):
        # Handle transient failures 
        for attempt in range(retries):    
            try:
                return self._get_response(query)
            except ClientError as e:
                if e.code == 429:
                    cd = backoff**attempt + random.uniform(0,1)
                    logger.warning('Too many requests detected. Backing off for %.1fs before retrying for attempt %d.', cd, attempt + 1)
                    time.sleep(cd)
            except ServerError as e:
                if e.code in [500, 503]:
                    cd = backoff**attempt + random.uniform(0,1)
                    logger.warning('Server error detected. Backing off for %.1fs before retrying for attempt %d.', cd, attempt + 1)
                    time.sleep(cd)
        logger.critical('Maximum retries exhausted. Retries: %d; Backoff: %d', retries, backoff)
        raise RuntimeError('Maximum retries exhausted.')

    # def begin_session(self, query):
    #     rag_context = self._retrieve_context(query)
    #     self._session_history.append(Content(role='user', parts=[Part.from_text(text=rag_context)]))
    #     self._session_history.append(Content(role='user', parts=[Part.from_text(text=query)]))
    #     response = self.client.models.generate_content(
    #         model=self.model_name,
    #         contents=query,
    #         config=GenerateContentConfig(
    #             thinking_config=ThinkingConfig(thinking_budget=self._thinking_budget)
    #         )
    #     )
    #     self._session_history.append(Content(role='model', parts=[Part.from_text(text=response.text)]))
    #     return response.text
    
    def _limit_input_tokens(self, token_limit=25000):
        # Keep trimming contents so long as input token exceeds token limit
        while True:
            token_ct = self.client.models.count_tokens(
                model=self.model_name, 
                contents=self._session_history
                )
            token_ct = token_ct.total_tokens
            if token_ct <= token_limit:
                break
            self._session_history.pop(0)
        logger.debug('Current token count from session history, based on token counter API: %d', token_ct)
        return self