import os
import logging
import pandas as pd
import numpy as np
from typing import Literal, Iterable
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def build_embedding_model(
    model_name: str='sentence-transformers/all-MiniLM-L6-v2',
    device: Literal['cuda', 'cpu', 'mps', 'npu']='cpu',
    batch_size: int=32,
    multi_process: bool=False, 
    normalize_embeddings: bool=False, 
    show_progress: bool=False, **kwargs
    ):
    """
    Initialize a HuggingFaceEmbeddings object. Default is SBERT MiniLM model
    for faster encoding. Embeddings are normalized for cosine similarity search.
    
    Args:
        model_name: HuggingFace model. Default 'all-MiniLM-L6-v2'.
        device: Computation device. Default 'mps' for MacOS.
        batch_size: Number of documents to encode. Default 32.
        multi_process: If multiprocessing should be enabled. Default False.
        normalize_embeddings: If embeddings must be normalized 
        (e.g. for cosine similarity). Default False.
        show_progress: If progress bar should be shown. Default False.
        kwargs: HuggingFaceEmbeddings kwargs. See source code for more info.
        
    Returns:
        embedding_model: HuggingFaceEmbeddings object.
        
    Reference: 
        https://python.langchain.com/api_reference/_modules/langchain_huggingface
        /embeddings/huggingface.html
    """
    if multi_process:
        logger.debug('Multi-processing enabled.')
        # Disable warning for multiprocess encoding
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    logger.debug(
        'Initializing HuggingFaceEmbeddings\nmodel_name: %s\ndevice: %s\nbatch_size: %d'
        '\nnormalize_embeddings: %s\nmulti_process: %s\nshow_progress: %s\nkwargs: %r',
        model_name, device, batch_size, normalize_embeddings, multi_process, show_progress, kwargs
        )
    return HuggingFaceEmbeddings(
        **kwargs,
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={
            'batch_size': batch_size,
            'normalize_embeddings': normalize_embeddings
            },
        query_encode_kwargs={'normalize_embeddings': normalize_embeddings},
        multi_process=multi_process,
        show_progress=show_progress
        )

def normalize_embeddings(emb, to_list: bool=False):
    """
    Args:
        emb: Unnormalized embeddings.
        to_list: Export as list of lists. Default False.
        
    Returns:
        normalized_emb: Normalized embeddings. Either 2D array or List of lists.
    """
    logger.info('Normalizing embeddings. Return to list: %s', to_list)
    # Convert to 2D array to compute row-wise norms
    if isinstance(emb, pd.Series):
        unnormalized_emb = np.stack(emb)
    elif isinstance(emb, list):
        unnormalized_emb = np.asarray(emb)
        
    norms = np.linalg.norm(unnormalized_emb, axis=1, keepdims=True)
    
    if to_list:
        logger.debug('Returning normalized embeddings in list format.')
        return (unnormalized_emb / norms).to_list() if norms != 0 else unnormalized_emb.to_list()
    
    return (unnormalized_emb / norms) if norms != 0 else unnormalized_emb

def combine_embeddings_from_pandas(submissions, comments, emb_model, context_weight: float=None, comment_weight: float=None):    
    """
    Linear combination of submission and comment embeddings to adjust each comment by its
    parent submission (i.e. context).
    
    Args:
        submissions: Dataframe containing submission docs and submission id.
        comments: Dataframe containing comment docs and submission id.
        context_weight: Weight for submission embeddings.
        comment_weight: Weight for comment embeddings.
        
    Returns:
        contextualized_emb: Linear combination of both embeddings. List of lists.
    """
    # Default higher weight on comment embeddings compared to context since comments == answer
    if comment_weight:
        context_weight = 1 - comment_weight
    elif context_weight:
        comment_weight = 1 - context_weight
    else: 
        context_weight = 0.3
        comment_weight = 0.7
    
    # Embed the submissions and comments
    submission_emb = emb_model.embed_documents(submissions.document.values)
    comment_emb = emb_model.embed_documents(comments.document.values)
    
    # Convert to array for vectorized operations
    submission_emb_arrays = [np.asarray(emb) for emb in submission_emb]
    comment_emb_arrays = [np.asarray(emb) for emb in comment_emb]
    
    # Coerce context embeddings to shape of comment embeddings. Missing submission ids filled with NaN.
    # Pandas is used to leverage fast indexing and vectorized operations
    context_emb = (pd.Series(submission_emb_arrays, index=submissions.submission_id)
            .reindex(comments.submission_id)
            .reset_index(drop=True)
            )
    
    # Mask to track only comments with context
    with_context = ~context_emb.isna()
    
    # Comments with context are adjusted based on linear combination; otherwise, retain original embeddings
    combined_emb = pd.Series(comment_emb_arrays)
    combined_emb[with_context] = (
        comment_weight * combined_emb[with_context] 
        + context_weight * context_emb[with_context]
        )
    
    # Return as a list of embeddings
    return combined_emb.to_list()

def documents_to_embeddings(docs: Iterable, emb_model):
    """
    Embed a list of documents using the provided embedding model.
    
    Args:
        docs: List of Documents.
        emb_model: Langchain embedding model or any model with embed_documents method.
        
    Returns:
        embeddings: List of document embeddings.
    """
    texts = [doc.page_content for doc in docs]
    embeddings = emb_model.embed_documents(texts)
    return embeddings