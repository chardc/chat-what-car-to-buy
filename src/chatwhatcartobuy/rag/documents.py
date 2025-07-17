import re
import logging
import pandas as pd
from typing import Optional
from chatwhatcartobuy.utils.getpath import get_repo_root
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
    metadatas = [{col: record[col] for col in record.index if col != 'document'}
                 for _, record in df.iterrows()]
    
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
            metadata = {col: record[col] for col in record.index if col != text_col}
            docs.append(Document(page_content=record[text_col], metadata=metadata))
        return docs
    
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