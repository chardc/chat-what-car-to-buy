#! /usr/bin/env python

import logging
from usedcaranalytics.pipeline import wrangling
from usedcaranalytics.utils.getpath import get_repo_root
from usedcaranalytics.config.logging_config import setup_logging

def main():
    """
    Parquet files exported by ETL pipeline are consolidated into PyArrow datasets,
    then converted to Pandas DataFrames for fast in-memory preprocessing.
    Submissions and comments are deduplicated, filtered, and cleaned to yield
    context-rich text records. Finally, text data (documents) are extracted from each
    dataset and stored to disk as parquet files under the root/data/processed directory
    to indicate data's readiness for downstream applications (Ex. document retriever).
    """
    
    # Parse multiple parquet files into a dataset
    logger.info('Parsing and preprocessing parquet files.')
    logger.debug('Parsing parquet files as PyArrow ParquetDataset.')
    submission_dataset = wrangling.read_dataset(get_repo_root() / 'data/raw/submission-dataset')
    comment_dataset = wrangling.read_dataset(get_repo_root() / 'data/raw/comment-dataset')
    
    submission_df = wrangling.wrangle_dataset(submission_dataset)
    comment_df = wrangling.wrangle_dataset(comment_dataset)
    
    # Create empty N,2 dataframes containing columns: document, submission_id
    logger.info('Extracting documents and saving to disk in parquet format.')
    logger.debug('Copying cleaned dataframes and dropping text columns.')
    submissions = submission_df.copy().drop(columns=['title', 'selftext'])
    comments = comment_df.copy().drop(columns=['body']).rename(columns={'parent_submission_id': 'submission_id'})
    
    # Merge submission title and selftext
    logger.debug('Merging submission title and selftext.')
    context_docs = (submission_df.title + ' ' + submission_df.selftext).values
    logger.debug('Inserting "document" column to submissions dataframe.')
    submissions.insert(0, 'document', context_docs)
    
    # Add comment tags to comments; tags can delineate data source from user prompt in RAG model
    logger.debug('Inserting "document" column to comments dataframe.')
    comments.insert(0, 'document', comment_df.body.values)
    
    # Export only the documents (text) to parquet under root/data/processed dir
    wrangling.pandas_to_parquet(submissions, 'submissions')
    wrangling.pandas_to_parquet(comments, 'comments')

if __name__ == '__main__':
    setup_logging(output_to_console=True, level=logging.INFO)
    logger = logging.getLogger(__name__)
    try:
        main()
    except Exception as e:
        logger.error('Error encountered: %s', e)
        raise e