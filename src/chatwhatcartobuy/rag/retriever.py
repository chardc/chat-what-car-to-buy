import logging
from pathlib import Path
from chatwhatcartobuy.rag.vector_db import build_vector_db, load_vector_db

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, submission_k: int, comment_n: int, embeddings):
        """
        Args:
            context_k: Number of relevant submissions to retrieve.
            comment_n: Number of comments to retrieve per relevant submission.
            embeddings: Embedding model for retrieval.
        """
        self.submission_k = submission_k
        self.comment_n = comment_n
        self.embeddings = embeddings
    
    def set_k(self, submission_k: int=None, comment_n: int=None):
        """Modify the k-number of returned documents."""
        if submission_k:
            self._k_input_check(submission_k)
            self.submission_k = submission_k
        if comment_n:
            self._k_input_check(comment_n)
            self.comment_n = comment_n
        return self
    
    @staticmethod
    def _k_input_check(k):
        """Utility function for k input validation."""
        if k:
            if not isinstance(k, int): raise TypeError('k must be an integer.')
            if k <= 0: raise ValueError('k must be a non-negative integer.')
    
    def retrieve(self, query: str):
        """Retrieve reddit posts and comments relevant to query."""
        # Retrieve relevant context documents first        
        submission_docs = self._vector_store.similarity_search(
            query, k=self.submission_k, filter={'record_type': 'submission'}
            )
        logger.debug('Retrieved %d relevant documents from submissions.', len(submission_docs))
        
        # For each relevant context, retrieve n relevant comments
        # Output format: source 1: <text> comment 1: <text> comment 2: <text> source n: <text> comment n: <text>...
        context = []
        for k, submission_doc in enumerate(submission_docs):
            context.append(f'source {k}:\n{submission_doc.page_content}')
            submission_id = submission_doc.metadata['submission_id']
            # MongoDB query language; Get comments from current submission id
            comment_docs = self._vector_store.similarity_search(
                query, 
                k=self.comment_n, 
                filter={'$and': [
                    {'record_type': {'$eq': 'comment'}}, 
                    {'submission_id': {'$eq': submission_id}}
                    ]}
                )
            logger.debug('Retrieved %d relevant documents from comments for submission id: %s.', len(comment_docs), submission_id)
            context.extend([f'comment {n}:\n{doc.page_content}' for n, doc in enumerate(comment_docs)])
        
        logger.debug('Retrieved a total of %d documents.', len(context))
        return '\n\n'.join(context)
    
    def build_vector_db(self, collection_name: str, persist_directory: str | Path, **kwargs):
        """Build a Chroma vector db and save to local directory."""        
        self._vector_store = build_vector_db(
            **kwargs, 
            embeddings=self.embeddings, 
            collection_name=collection_name, 
            persist_directory=persist_directory
            )
        return self
    
    def load_vector_db(self, collection_name: str, persist_directory: str | Path, **kwargs):
        """Load a Chroma vector db from local directory."""
        self._vector_store = load_vector_db(
            **kwargs, 
            embeddings=self.embeddings, 
            collection_name=collection_name, 
            persist_directory=persist_directory
            )
        return self
    
    def add_documents(self, docs: list, ids: list=None, **kwargs):
        """Add documents to ChromaDB collection."""
        for i, (batch_docs, batch_ids) in enumerate(zip(self._split_to_batches(docs), self._split_to_batches(ids))):
            logger.debug('Adding documents to vector database (Batch %d).', i)
            self._vector_store.add_documents(batch_docs, ids=batch_ids, **kwargs)
        return self
    
    @staticmethod
    def _split_to_batches(iterable, batch_size=5000):
        """Split documents or ids by batch to stay within db upsert limits."""
        for i in range(0, len(iterable), batch_size):
            yield(iterable[i: i+batch_size])