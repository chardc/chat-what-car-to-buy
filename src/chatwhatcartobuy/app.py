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
from chatwhatcartobuy.utils.txtparser import read_txt_file

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

INTRO = read_txt_file(get_path(start_path=__file__, target='cli_app_intro.txt', subdir='chatwhatcartobuy'))

def print_intro(console):
    console.print(
        Panel(
            INTRO.strip(), 
            title="[bold blue]Welcome![/bold blue]", 
            expand=False, 
            border_style="blue")
        )

def ask_user_input():
    return Prompt.ask("[bold green]You[/bold green]", default="", show_default=False).lower()

def print_response(console, answer):
    console.print(
            Panel.fit(
                Text(answer, style="white"), 
                title="[bold blue]Chatbot[/bold blue]", 
                border_style="blue"
                ))

def chat_loop(chatbot: ChatBot, exit_kwords = ['exit', 'quit', 'terminate']):
    console = Console()
    print_intro(console)
    # chatbot.begin_session(ask_user_input())
    turn = 0
    while True:
        user_query = ask_user_input()
        if user_query.strip() in exit_kwords:
            break
        if turn == 0:
            context = chatbot.retrieve_context(user_query)
            user_query += f'\n\n{context}'
        console.print("\n[yellow]Chatbot is thinking...[/yellow]\n")
        print_response(console, chatbot.chat(user_query))
        console.print(f"[dim]Type {'/'.join(exit_kwords)} to leave the app.[/dim]\n")
    console.print("\n\n[red]Exiting the app...[/red]\n")

def main(build_new_vector_store: bool=None):
    if build_new_vector_store:
        prepare_document_files()
        retriever = build_retriever()
    else:
        retriever = Retriever(submission_k=15, comment_n=30, embeddings=build_embeddings())
        retriever.load_vector_store(collection_name='reddit-data', persist_directory=get_repo_root()/'chroma')
    chatbot = ChatBot(retriever)
    # Instructions for LLM to avoid markdown and rich formatting and focus on CLI-appropriate formatting
    chat_loop(chatbot)

def prompt_yes_no():
    while True:
        choice = input('[SETUP] Build a new vector store? (Y/N): ').lower().strip()
        if choice in ['y', 'n']:                
            return True if choice == 'y' else False
        print('Invalid input. Try again. (Y/N)')

if __name__ == '__main__':
    try:
        setup_logging(level=logging.DEBUG, file_prefix='chatbot-cli-app', output_to_file=True)
        load_dotenv(get_path(start_path=__file__, target='.env', subdir='config'))
        # Building a new vector store every time increases latency due to additional processes
        # Also, a new vector store can only be built when chroma/ directory is deleted
        # Otherwise, we risk appending duplicate documents to the vector db--rendering retriever useless
        main(build_new_vector_store=prompt_yes_no())
    except Exception as e:
        logger.critical('Exception occured: %s', e)
        raise e
    
# Sample prompt
# Chat, I want to buy a reliable, affordable, and sporty sedan or hatchback under $15K aud. I'm looking at Mazda 3 but also considering other japanese manufactured models. Can you recommend me any models that are at most 15 years old, and can you present me with a comprehensive pros and cons of each model. Try to limit your recommendations to the top 2 models that you think best fits my budget and profile as a student with limited income.