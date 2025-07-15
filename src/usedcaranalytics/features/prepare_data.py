#! /usr/bin/env python

import logging
import pandas as pd
from typing import Literal
from usedcaranalytics.features import wrangling
from usedcaranalytics.utils.getpath import get_repo_root
from usedcaranalytics.config.logging_config import setup_logging

def wrangle_dataset(dataset):
    """
    Perform data cleaning and wrangling on an input PyArrow dataset.
    
    Args:
        dataset: PyArrow ParquetDataset.
        
    Returns:
        dataframe: Cleaned and wrangled Pandas DataFrame.
    """
    # Either submission or comment; Get primary keys and columns containing text
    data_source = dataset.schema.metadata[b'data_source'].decode()
    subset = 'submission_id' if data_source == 'submission' else 'comment_id'
    cols = ['title', 'selftext'] if data_source == 'submission' else ['body']
    
    # Convert dataset to pandas for in-memory transformations
    logger.debug('Converting ParquetDataset to Pandas DataFrame.')
    out_df = wrangling.dataset_to_pandas(dataset)

    # Deduplication based on column subset
    logger.debug('Deduplicating %s dataframe.', data_source)
    out_df = wrangling.deduplicate_pandas(out_df, subset)

    # Remove empty records (records without valid tokens)
    logger.debug('Removing empty rows from %s dataframe.', data_source)
    out_df = wrangling.remove_empty_rows_pandas(out_df, cols)
    
    # Filter unreliable / low-score comments and posts
    logger.debug('Removing low score records from %s dataframe.', data_source)
    out_df = wrangling.remove_low_score_pandas(out_df, threshold=-2)
    
    # Pad URLs with context tags <URL>
    logger.debug('Padding URLs in %s from %s dataframe.', ', '.join(cols), data_source)
    out_df = wrangling.replace_url_pandas(out_df, cols)
    
    # Remove extra whitespaces
    logger.debug('Removing extra whitespaces from %s dataframe.', data_source)
    out_df = wrangling.remove_extra_whitespace_pandas(out_df, cols)
    
    logger.info('Finished preprocessing %s dataset.', data_source) 
    return out_df

def pandas_to_parquet(df, data_source: Literal['comments', 'submissions'], dir_path=None):
    """
    Args:
        df: Pandas DataFrame.
        data_source: Specify either 'comments' or 'submissions'
        dir_path: Path of directory where parquet will be stored. Default is 'data/processed' when None.
    
    Notes:
        Export pandas dataframe to parquet.
    """
    if data_source not in ['comments', 'submissions']:
        raise ValueError('Data source can only be either "comments" or "submissions".')
    
    if dir_path is None:
        dir_path = get_repo_root() / 'data/processed'
    
    logger.debug('Exporting %s to disk path: %s', data_source, dir_path / f'{data_source}.parquet')
    
    df.to_parquet(dir_path / f'{data_source}.parquet', engine='pyarrow')
    logger.info('Finished exporting preprocessed Reddit post and comment data to %s', dir_path)

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
    
    submission_df = wrangle_dataset(submission_dataset)
    comment_df = wrangle_dataset(comment_dataset)
    
    # Create empty N,2 dataframes containing columns: document, submission_id
    logger.info('Extracting documents and saving to disk in parquet format.')
    logger.debug('Copying cleaned dataframes and dropping text columns.')
    submissions = submission_df.copy().drop(columns=['title', 'selftext'])
    comments = comment_df.copy().drop(columns=['body']).rename(columns={'parent_submission_id': 'submission_id'})
    
    # Merge submission title and selftext with context tags
    logger.debug('Merging submission title and selftext. Padding with context tags.')
    context_docs = (submission_df.loc[:, ['title', 'selftext']]
                     .apply(lambda row: '<title> ' + row.title + ' </title> <selftext> ' + row.selftext + ' </selftext>', axis=1)
                     )
    logger.debug('Inserting "document" column to submissions dataframe.')
    submissions.insert(0, 'document', context_docs)
    
    # Add comment tags to comments; tags can delineate data source from user prompt in RAG model
    logger.debug('Padding comments with comment tag.')
    comment_docs = comment_df.loc[:, 'body'].map(lambda row: '<comment> ' + row + ' </comment>')
    logger.debug('Inserting "document" column to comments dataframe.')
    comments.insert(0, 'document', comment_docs)
    
    # Export only the documents (text) to parquet under root/data/processed dir
    pandas_to_parquet(submissions, 'submissions')
    pandas_to_parquet(comments, 'comments')

if __name__ == '__main__':
    setup_logging(output_to_console=True, level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    try:
        main()
    except Exception as e:
        logger.error('Error encountered: %s', e)
        raise e