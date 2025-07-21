import logging
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
        model_kwargs={'devide': 'mps'},
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
    logger.debug('Parsing latest Reddit data as documents for RAG.')
    submission_docs, submission_ids = get_documents_and_ids(file_path=get_latest_path('processed/*/submissions.parquet'))
    comment_docs, comment_ids = get_documents_and_ids(file_path=get_latest_path('processed/*/comments.parquet'))
    
    # Batch add to local vector database; batch_size=5000
    return (
        Retriever(submission_k=10, comment_n=20, embeddings=build_embeddings())
        .build_vector_db(collection_name='reddit-data', persist_directory=get_repo_root()/'chroma')
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

def main():
    prepare_document_files()
    chatbot = build_chatbot()
    exit_kwords = ['exit', 'quit', 'terminate']
    print(
        """
        Welcome to 'Chat, what car to buy?', an AI Chatbot that helps you decide on the best used car for you.
        Chat is based on Gemini 2.5. Flash, a low-cost lightweight variant of Gemini with sufficient cognitive
        capabilities for general tasks. It leverages crowd knowledge by retrieving relevant online discussions 
        from Reddit to provide rich and diverse information based on lived experiences from owners themselves.
        
        To begin, just type what's on your mind and chat will do the rest.
        
        """
        )
    user_query = input("Chat, ")
    while True:
        if user_query.lower() in exit_kwords:
            break
        print(f"\n\n{chatbot.query(user_query)}\n\n")
        print(f"\n\nType {'/'.join(exit_kwords)} to exit the app.\n\n")
        user_query = input("Follow-up prompt: ")

if __name__ == '__main__':
    setup_logging(level=logging.DEBUG, file_prefix='chatbot-cli-app', output_to_file=True)
    load_dotenv(get_path(start_path=__file__, target='.env', subdir='config'))
    main()