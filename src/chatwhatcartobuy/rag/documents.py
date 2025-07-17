import re
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List
from chatwhatcartobuy.utils.wrangling import read_dataset, wrangle_dataset, pandas_to_parquet
from chatwhatcartobuy.config.logging_config import setup_logging
from chatwhatcartobuy.utils.getpath import get_latest_path
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def chunks_from_pandas(df, chunk_size: int=256, chunk_overlap: int=0, **kwargs):
    """
    Split text to chunks with specified token sizes to fit LLM context windows. Returns a list
    of chunked Document objects. Default splitter is TokenTextSplitter. 
    
    Args:
        df: Pandas DataFrame containing document and submission id.
        chunk_size: Number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens from contiguous chunks.
        kwargs: Keyword arguments for text splitter.
        
    Returns:
        documents: List of chunked Documents with metadata.
    """
    logger.debug('Initializing TokenTextSplitter with chunk_size=%d, chunk_overlap=%d', chunk_size, chunk_overlap)
    text_splitter = TokenTextSplitter(**kwargs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Retain original metadata from dataframe
    logger.debug('Building metadata from dataframe with columns: %s', ', '.join(df.columns))
    
    metadatas = []
    for _, record in df.iterrows():
        # Remove "document" from metadata and convert timestamp to unix seconds
        metadata = record.to_dict()
        metadata['timestamp'] = metadata['timestamp'].timestamp()
        metadata.pop('document')
        metadatas.append(metadata)
    
    # Generate langchain Docs in specified chunks 
    docs = text_splitter.create_documents(df.document, metadatas=metadatas)
    logger.debug('Returning %d chunked documents from source dataframe with %d rows', len(docs), len(df))
    return docs

def documents_from_pandas(df, text_col: str='document', include_metadata: bool=False):
    """
    Generate langchain Documents from a dataframe containing text and optional metadata.
    
    Args:
        df: Dataframe containing text (and metadata).
        text_col: Name of column containing documents. Default = "document".
        include_metadata: If True, all other columns in dataframe will be converted to metadata.
    
    Returns:
        documents: List of Documents.
    """
    if include_metadata:
        logger.debug('Building metadata from dataframe with columns: %s', ', '.join(df.columns))
        docs = []
        for _, record in df.iterrows():
            # Remove "document" from metadata and convert timestamp to unix seconds
            metadata = record.to_dict()
            metadata['timestamp'] = metadata['timestamp'].timestamp()
            metadata.pop('document')
            docs.append(Document(page_content=record[text_col], metadata=metadata))
    else:
        docs = [Document(page_content=text) for text in df.loc[:, text_col]]
    
    logger.debug('Returning %d documents from source dataframe with %d rows', len(docs), len(df))
    return docs
        
def generate_document_ids(docs):
    """
    Generates descriptive ids for submission and comment documents. Format:
    Submission: submission_id-000n, where n=chunk
    Comment: submission_id-comment_id-000n, where n=chunk
    
    Args:
        docs: List of langchain Documents.
        
    Returns:
        id: Descriptive id for the documents.
    """
    # Either {submission_id}, or [submission_id, comment_id] for submissions and comments respectively
    col_pat = re.compile(r'\w+_id$')
    id_keys = [col_pat.fullmatch(key).group() for key in docs[0].metadata if col_pat.fullmatch(key) is not None]
    logger.debug('Generating document IDs from %s', 'submissions' if len(id_keys)==2 else 'comments')
    
    id_set = set() # Store ids in a set for fast membership checks
    ids = [] # List of ids to return
    
    # For every document, assign a unique id depending on record type. 
    for doc in docs:
        chunk_n = 0
        id_prefix = '-'.join([doc.metadata[key] for key in id_keys])
        id = id_prefix + f'-{chunk_n:04d}'
        while id in id_set:
            chunk_n += 1
            id = id_prefix + f'-{chunk_n:04d}'
        # Update set with unique id
        id_set.add(id)
        ids.append(id)
    
    logger.debug('Successfully generated %d unique IDs from %d documents.', len(ids), len(docs))
    
    return ids

def preprocess_raw_parquet(path_or_paths: List[Path] | Path):
    """""
    Wrangle raw parquet files exported from the Reddit API to Local ETL pipeline.
    
    Args:
        path_or_paths: List of parquet file Paths or Path to dataset directory.
    
    Returns:
        df: Preprocessed dataframe    
    """
    try:
        # Parse multiple parquet files into a dataset
        logger.debug('Parsing parquet files from: %s', path_or_paths)
        # Merge all parquet files into a dataset
        dataset = read_dataset(path_or_paths)
        # Wrangle pyarrow dataset and return a pandas dataframe
        df = wrangle_dataset(dataset)
        return df
    
    except Exception as e:
        logger.critical('Error encountered: %s', e)
        raise e

def get_documents_and_ids(df: Optional[pd.DataFrame]=None, file_path: Optional[str]=None):
    """
    Takes either a preprocessed dataframe or parquet filepath and returns a tuple of 
    langchain Documents and unique document IDs. DataFrame must contain a 'document'
    column, as well as respective id columns. Only takes keyword arguments.
    
    Args:
        df: Preprocessed dataframe.
        file_path: PosixPath to preprocessed dataframe.
        
    Returns:
        tuple: Tuple of documents and document ids.
    """
    if file_path:
        logger.debug('Parsing parquet file from path: %s', file_path)
        df = pd.read_parquet(file_path, engine='pyarrow')
    docs = chunks_from_pandas(df)
    doc_ids = generate_document_ids(docs)
    return (docs, doc_ids)

if __name__ == '__main__':
    """
    If this script is run, the raw datasets will be parsed, preprocessed, and saved to disk
    path: project_root/data/processed/partition/.
    """
    setup_logging(level=logging.INFO, output_to_console=True)
    logger.info('Preprocessing raw datasets for document generation.')
    # Preprocess raw datasets
    submissions = preprocess_raw_parquet(get_latest_path('raw/submission-dataset/*/'))
    comments = preprocess_raw_parquet(get_latest_path('raw/comment-dataset/*/'))
    # Optional: Save to disk
    pandas_to_parquet(submissions, file_name='submissions.parquet', partition_by_date=True)
    pandas_to_parquet(comments, file_name='comments.parquet', partition_by_date=True)