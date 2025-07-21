#! /usr/bin/env python
import time
import logging
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from dotenv import load_dotenv
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from chatwhatcartobuy.rag.documents import get_documents_and_ids, preprocess_raw_parquet, pandas_to_parquet
from chatwhatcartobuy.rag.retriever import Retriever
from chatwhatcartobuy.llm.chatbot import ChatBot
from chatwhatcartobuy.config.logging_config import setup_logging
from chatwhatcartobuy.utils.getpath import get_path, get_repo_root, get_latest_path

logger = logging.getLogger(__name__)

def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2',
        model_kwargs={'device': 'mps'},
        encode_kwargs={'batch_size': 256, 'normalize_embeddings': True},
        query_encode_kwargs={'normalize_embeddings': True}
        )

def prepare_document_files():
    logger.debug('Preprocessing raw datasets in preparation for document generation.')
    submissions = preprocess_raw_parquet(get_latest_path('raw/submission-dataset/*/'))
    comments = preprocess_raw_parquet(get_latest_path('raw/comment-dataset/*/'))
    logger.debug('Exporting latest preprocessed datasets to %s', get_repo_root()/f'data/processed/{datetime.now():%Y-%m-%d}')
    pandas_to_parquet(submissions, file_name='submissions.parquet', partition_by_date=True)
    pandas_to_parquet(comments, file_name='comments.parquet', partition_by_date=True)

def build_retriever():
    # Build a retriever with new vector store
    logger.debug('Parsing latest Reddit data as documents for RAG.')
    submission_docs, submission_ids = get_documents_and_ids(file_path=get_latest_path('processed/*/submissions.parquet'))
    comment_docs, comment_ids = get_documents_and_ids(file_path=get_latest_path('processed/*/comments.parquet'))
    
    # Batch add to local vector database; batch_size=5000
    return (
        Retriever(submission_k=10, comment_n=20, embeddings=build_embeddings())
        .load_vector_store(collection_name='reddit-data', persist_directory=get_repo_root()/'chroma')
        .add_documents(submission_docs, ids=submission_ids)
        .add_documents(comment_docs, ids=comment_ids)
        )

def build_chatbot(**kwargs):
    return ChatBot(
        **kwargs, 
        retriever=build_retriever(), 
        model='gemini-2.5-flash', 
        is_thinking=True
        )

INTRO = """
Welcome to 'Chat, what car to buy?', an AI Chatbot that helps you decide on the best used car for you.

Chat is powered by Gemini 2.5 Flash, a lightweight LLM that combines general knowledge with real crowd wisdom from Reddit. You'll get recommendations and warnings based on lived experiences from real car owners.

Type your question (e.g. "Best used SUV under $10k?" or "What are common problems for a 2012 Civic?") and let Chat handle the research.

Type 'exit' at any time to leave the app.
"""

ADDTL_INSTRUCTIONS = """
Please format your response so it is clear, visually organized, and easy to read in a plain-text terminal (CLI) environment.
Do NOT use Markdown formatting (no # headers, no **bold**, no *italic*, no tables, no backticks).
Instead:
- Use all-caps or underlines for section headings (e.g., MODEL SUMMARY or ==== MODEL SUMMARY ====)
- Use regular hyphens or asterisks for bullet points (- Pro: ...)
- Separate sections with whitespace or plain dividers (like -----)
- Do not use color codes or other markup.

EXAMPLE FORMATTING:

TOP 2 RECOMMENDATIONS

1. MAZDA 3 (2014-2018)

Pros:
- Reliable according to most owners ("Nothing has broken in 105k miles", Source 1)
- Sporty driving feel
...

-----------------------------------------------------

2. TOYOTA COROLLA (2008-2019)

Pros:
- Legendary reliability
...
"""

def print_intro():
    console = Console()
    console.print(
        Panel(
            INTRO.strip(), 
            title="[bold blue]Welcome![/bold blue]", 
            expand=False, 
            border_style="blue")
        )

def chat_loop(chatbot, exit_kwords = ['exit', 'quit', 'terminate']):
    console = Console()
    print_intro() 
    while True:
        user_query = Prompt.ask("[bold green]You[/bold green]", default="", show_default=False)
        if user_query.strip().lower() in exit_kwords:
            break
        time.sleep(0.1)
        console.print("\n[yellow]Chatbot is thinking...[/yellow]\n")
        answer = chatbot.query(user_query)
        console.print(
            Panel.fit(
                Text(answer, style="white"), 
                title="[bold blue]Chatbot[/bold blue]", 
                border_style="blue")
            )
        console.print(f"[dim]Type {'/'.join(exit_kwords)} to leave the app.[/dim]\n")
    time.sleep(0.1)
    console.print("\n\n[red]Exiting the app...[/red]\n")

def main(build_new_vector_store: bool=None):
    if build_new_vector_store:
        prepare_document_files()
        retriever = build_retriever()
    else:
        retriever = Retriever(submission_k=10, comment_n=20, embeddings=build_embeddings())
        retriever.load_vector_store(collection_name='reddit-data', persist_directory=get_repo_root()/'chroma')
    chatbot = ChatBot(retriever)
    # Instructions for LLM to avoid markdown and rich formatting and focus on CLI-appropriate formatting
    chatbot.add_instructions(ADDTL_INSTRUCTIONS)
    chat_loop(chatbot)

if __name__ == '__main__':
    setup_logging(level=logging.DEBUG, file_prefix='chatbot-cli-app', output_to_file=True)
    try:
        load_dotenv(get_path(start_path=__file__, target='.env', subdir='config'))
        # Building a new vector store every time increases latency due to additional processes
        # Also, a new vector store can only be built when chroma/ directory is deleted
        # Otherwise, we risk appending duplicate documents to the vector db--rendering retriever useless
        while True:
            setup = input('[SETUP] Build a new vector store? (Y/N): ').lower()
            if setup in ['y', 'n']:
                print('\n')
                main(build_new_vector_store=True) if setup == 'y' else main()
                break
    except Exception as e:
        logger.critical('Exception occured: %s', e)
        raise e