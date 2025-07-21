import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from chatwhatcartobuy.utils.getpath import get_path
from chatwhatcartobuy.rag.retriever import Retriever

logger = logging.getLogger(__name__)

class ChatBot:
    INSTRUCTIONS = (
"""
You are a friendly, knowledgeable, and humanistic assistant specialized in providing detailed advice to prospective 
buyers of secondhand cars. You leverage insights drawn from genuine discussions and real-life experiences 
shared by users on Reddit, supplemented by authoritative automotive sources.

When responding to user queries, carefully adhere to the following guidelines:
1. Prioritize evidence-based insights drawn from relevant Reddit discussions, clearly referencing user experiences.
2. Provide objective, balanced, and actionable information, highlighting both pros and cons.
3. Clarify common issues, reliability factors, potential repair costs, and maintenance advice.
4. Keep the responses thorough and structured logically with bullet points or numbered lists where beneficial.
5. Present a conclusion, summary, or synthesis at the end of your response to help guide the user.
    
Below are relevant Reddit discussions to inform your answer. Review them carefully to ensure the advice provided is 
specific, relevant, and actionable.
"""
)
    
    def __init__(self, retriever: Retriever, model: str='gemini-2.5-flash', is_thinking: bool=False):
        # client requires GEMINI_API_KEY env variable; Load .env file at the start of application.
        self.client = genai.Client()
        self.retriever = retriever
        self.model = model
        self.set_thinking(is_thinking)
        logging.debug('Initialized ChatBot with model=%s and thinking_budget%s=', model, self.thinking_budget)
    
    def set_thinking(self, is_thinking: bool):
        if not isinstance(is_thinking, bool): 
            raise TypeError('is_thinking must be either True or False, and not truthy/falsy values.')
        if is_thinking:
            self.thinking_budget = 1024
        else:
            self.thinking_budget = 0
        return self
    
    def set_thinking_budget(self, value: int):
        # Directly modify thinking budget
        if value < -1: 
            raise ValueError('Valid values range from -1 to 24576 (Gemini 2.5 Flash) or 32768 (Gemini 2.5 Pro).')
        if not isinstance(value, int): 
            raise TypeError('Value must be an integer from -1 to 24576/32768.')
        self.thinking_budget = value
        
    def add_instructions(self, instructions: str):
        self.INSTRUCTIONS += f"\n\nADDITIONAL INSTRUCTIONS: {instructions}"
    
    def query(self, user_query: str):
        rag_context = self.retriever.retrieve(user_query)
        logging.debug('Retrieved context from database. Approx %d tokens.', len(rag_context)//4)
        prompt = (f"{self.INSTRUCTIONS}\n\n"
                  f"### Relevant Reddit discussions\n{rag_context}\n\n"
                  f"### User Query\n{user_query}\n\n"
                  f"### Chatbot Response\n"
                  )
        logging.debug('Prompt constructed. Approx. %d tokens.', len(prompt)//4)
        response = self.client.models.generate_content(
            model=self.model, 
            contents=prompt,
            config=types.GenerateContentConfig(
                # Allow model to think; improves answer quality
                thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget) 
                )
            )
        return response.text