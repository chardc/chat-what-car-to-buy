#! /usr/bin/env python

"""
Parquet files exported by ETL pipeline are consolidated into PyArrow datasets,
then converted to Pandas DataFrames for fast in-memory preprocessing.
Submissions and comments are deduplicated, filtered, and cleaned to yield
context-rich text records. 

Finally, text data (documents) are extracted from each dataset and stored to disk 
as parquet files under the root/data/processed directory to indicate data's readiness f
or downstream applications (Ex. document retriever).
"""

import logging
from typing import Literal, Optional, Dict
from pathlib import Path
from chatwhatcartobuy.pipeline import wrangling
from chatwhatcartobuy.utils.getpath import get_latest_path
from chatwhatcartobuy.config.logging_config import setup_logging

def prepare_and_export_data(
    dataset_paths: Optional[Dict[Literal['submissions', 'comments'], Path]]=None, 
    export_dir_path: Optional[str]=None
    ):
    """
    By default, the latest raw data from the ETL pipeline is merged as a Dataset, preprocessed,
    and exported as 'submissions.parquet' and 'comments.parquet' for use in information retrieval.
    
    Args:
        dataset_paths: Dict containing dataset paths. Format: {'submissions': Path, 'comments': Path}.
        export_dir_path: Optional path where parquet files will be saved to. 
        Default is 'data/processed/export_date/*.parquet'
    
    Notes:
        Exports parquet files to data/processed/
    """
    logger = logging.getLogger(__name__)
    
    # Parse multiple parquet files into a dataset
    logger.info('Parsing and preprocessing parquet files.')
    
    # Get the directories containing latest scraped data
    logger.debug('Getting paths of most recently scraped data.')
    if dataset_paths is None:
        submission_dataset_dir = get_latest_path('data/raw/submission-dataset/*/')
        comment_dataset_dir = get_latest_path('data/raw/comment-dataset/*/')
    else:
        submission_dataset_dir = dataset_paths['submissions']
        comment_dataset_dir = dataset_paths['comments']
    
    # Merge all parquet files into a dataset
    logger.debug('Parsing submission parquet files as PyArrow ParquetDataset. Source path: %s', submission_dataset_dir)
    submission_dataset = wrangling.read_dataset(submission_dataset_dir)
    logger.debug('Parsing comment parquet files as PyArrow ParquetDataset. Source path: %s', comment_dataset_dir)
    comment_dataset = wrangling.read_dataset(comment_dataset_dir)
    
    # Wrangle pyarrow dataset and return a pandas dataframe
    submission_df = wrangling.wrangle_dataset(submission_dataset)
    comment_df = wrangling.wrangle_dataset(comment_dataset)
    
    # Modify dataframes to only contain 'document' col for all text data
    logger.info('Extracting documents and saving to disk in parquet format.')
    logger.debug('Copying clean submission dataframe, dropping selftext, and renaming title to document.')
    submissions = submission_df.copy().drop(columns=['selftext']).rename(columns={'title': 'document'})
    logger.debug('Copying clean comment dataframe and renaming body to document.')
    comments = comment_df.copy().rename(columns={'parent_submission_id': 'submission_id', 'body': 'document'})
    
    # Merge submission title and selftext into 'document' col
    logger.debug('Merging submission title and selftext into a series.')
    context_docs = (submission_df.title + ' ' + submission_df.selftext).values
    logger.debug('Replacing values in "document" column with merged submission text.')
    submissions.loc[:, 'document'] = context_docs
    
    # Export only the documents (text) to parquet under root/data/processed dir
    wrangling.pandas_to_parquet(submissions, data_source='submissions', partition_by_date=True, dir_path=export_dir_path)
    wrangling.pandas_to_parquet(comments, data_source='comments', partition_by_date=True, dir_path=export_dir_path)

if __name__ == '__main__':
    # For testing
    setup_logging(level=logging.INFO, output_to_console=True)
    prepare_and_export_data()