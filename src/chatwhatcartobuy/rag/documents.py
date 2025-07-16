import pandas as pd
from typing import Optional
from chatwhatcartobuy.utils.getpath import get_repo_root
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document

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
    text_splitter = TokenTextSplitter(**kwargs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Retain original metadata from dataframe
    metadatas = [{col: record[col] for col in record.index if col != 'document'}
                 for _, record in df.iterrows()]
    
    # Generate langchain Docs in specified chunks 
    docs = text_splitter.create_documents(df.document, metadatas=metadatas)
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
        docs = []
        for _, record in df.iterrows():
            metadata = {col: record[col] for col in record.index if col != text_col}
            docs.append(Document(page_content=record[text_col], metadata=metadata))
        return docs
    
    return [Document(page_content=text) for text in df.loc[:, text_col]]

if __name__ == '__main__':
    """Simple tests to check functionality."""
    
    submissions = pd.read_parquet(get_repo_root() / 'data/processed/submissions.parquet')
    comments = pd.read_parquet(get_repo_root() / 'data/processed/comments.parquet')
    
    n = 3
    sub_docs = chunks_from_pandas(submissions[:n])
    
    # Test print
    for i, doc in enumerate(sub_docs[:n]):
        print(f'\nsubmission doc {i}:\n{'*'*100}\n{doc.page_content}\n{doc.metadata}\n\n')
        
    com_docs = documents_from_pandas(comments[:n], include_metadata=True)
    
    # Test print
    for i, doc in enumerate(com_docs[:n]):
        print(f'\ncomment doc {i}:\n{'*'*100}\n{doc.page_content}\n{doc.metadata}\n\n')