"""
This module provides reusable functions for basic data wrangling. Since the ETL pipeline
exports parquet files by batch, it has no visibility on previous and subsequent batches.
Thus, when merging all parquet files into a single dataset, there are cases of duplicate
entries. 

Additionally, the transformations done prior to exporting are quite basic (e.g.
deduplication, removal of [deleted] and [removed] text, removal of newlines and extra 
whitespaces). This module intends to provide helper functions that handle the merging
of all parquet files into singular dataset, deduplication of dataset, text cleaning and
basic record masking (e.g. filtering out low scores or non-token text records).
"""

import re
import logging
import datetime as dt
import pandas as pd
import pyarrow.parquet as pq
from typing import Hashable, List, Literal
from chatwhatcartobuy.utils.getpath import get_repo_root

logger = logging.getLogger(__name__)

def read_dataset(path_or_paths, **kwargs):
    """
    Parses multiple *.parquet files into a single dataset.
    
    Args:
        path: Filepath to the dataset directory that contains parquet files.
        kwargs: pyarrow.parquet.ParquetDataset keyword arguments.
        
    Returns:
        dataset: pyarrow.parquet ParquetDataset.
    """
    logger.debug('Parsing parquet files as PyArrow ParquetDataset. Source path(s): %s', path_or_paths)
    dataset = pq.ParquetDataset(path_or_paths, **kwargs)
    return dataset

def dataset_to_pandas(dataset):
    """
    Wrapper for dataset.read().to_pandas().
    
    Args:
        dataset: ParquetDataset object.
        
    Returns:
        df: Pandas DataFrame.
    """
    logger.debug('Converting ParquetDataset to Pandas DataFrame.')
    return dataset.read().to_pandas()

def deduplicate_pandas(df, subset: Hashable=None):
    """
    Args:
        df: Pandas dataframe.
    
    Returns:
        df: Deduplicated copy of the original dataframe.
    """
    logger.debug('Deduplicating dataframe.')
    return df.drop_duplicates(subset=subset)

def remove_empty_rows_pandas(df, cols: List[str], pat: str=r'\b\w{2,}'):
    """
    Performs naive tokenization of text records based on provided regex pattern
    and removes rows without tokens. Default token is a word char with len >= 2.
    
    Args:
        df: Pandas dataframe.
        cols: List of column names containing text data for tokenization.
        pat: Regex pattern for naive tokenization.
        
    Returns:
        df: Copy of original dataframe without empty or zero-token records.
    """
    out_df = df.copy()
    # Naive tokenization of dataframe, then mask rows to retain only those with valid records from all cols
    mask = (out_df.loc[:, cols]
            .apply(lambda col: col.str.findall(pat)) # Tokenize all rows; returns list of tokens
            .apply(lambda row: row.all(), axis=1) # Ensure both columns contain non-empty list of tokens
            )
    logger.debug('Removing empty rows from dataframe.')
    return out_df[mask]

def remove_low_score_pandas(df, threshold: int):
    """
    Args:
        df: Pandas dataframe.
        threshold: Minimum score of record to consider.
        
    Returns:
        df: Copy of original dataframe with low score records removed.
    """
    out_df = df.copy()
    logger.debug('Removing low score records from dataframe.')
    return out_df.loc[out_df.score >= threshold]

def replace_text_pandas(df, cols: List[str], pat: str, repl: str):
    """
    Remove or replace text with context tags.
    
    Args:
        df: Pandas dataframe.
        cols: List of column names containing text for replacement.
        pat: Regex pattern.
        repl: Replacement string.
    
    Returns:
        df: Copy of original dataframe with replaced or padded string.
    """
    out_df = df.copy()
    logger.debug('Replacing text in columns: %s with pattern: %s, and replacement: %s', ', '.join(cols), pat, repl)
    out_df.loc[:, cols] = (out_df.loc[:, cols]
                           .apply(lambda col: col.str.replace(pat, repl, regex=True))
                           )
    return out_df

def replace_url_pandas(df, cols: List[str], pat: str=r'[\[\(]?https?://[\S]+[\]\)]?', repl: str='<URL>'):
    """
    Wrapper for replace_text_pandas configured to replace URLs. Default pads links with <URL>.
    
    Args:
        df: Pandas dataframe.
        cols: List of column names containing text for replacement.
        pat: Regex pattern for URL.
        repl: Replacement string for URL. Default is <URL>.
        
    Returns:
        df: Copy of original dataframe with padded URLs.
    """
    logger.debug('Padding URLs in columns: %s from dataframe.', ', '.join(cols))
    return replace_text_pandas(df, cols, pat, repl)

def remove_extra_whitespace_pandas(df, cols: List[str]):
    """
    Wrapper for replace_text_pandas configured to remove contiguous extra whitespaces
    with a single space.
    
    Args:
        df: Pandas dataframe.
        cols: List of column names containing text for replacement.
        
    Returns:
        df: Copy of original dataframe with extra whitespaces removed.
    """
    # Only replace contiguous whitespaces with single space.
    logger.debug('Removing extra whitespaces in columns: %s.', ', '.join(cols))
    return replace_text_pandas(df, cols, pat=r'\s{2,}', repl=' ')

def lowercase_text_pandas(df, cols: list[str]):
    """
    Optional. ETL pipeline already converts text data to lowercase.
    
    Args:
        df: Pandas dataframe.
        cols: List of column names containing text records.
    
    Returns:
        df: Copy of original dataframe with text records in lowercase.
    """
    out_df = df.copy()
    logger.debug('Lowercasing text in columns: %s', ', '.join(cols))
    out_df.loc[:, cols] = (out_df.loc[:, cols]
                           .apply(lambda col: col.str.lower())
                           )
    return out_df

def assign_record_type_pandas(df, record_type: Literal['submission', 'comment']):
    """
    Assigns a new column to the current dataframe containing the data source
    from the pyarrow schema metadata. Useful for filtering records downstream.
    
    Args:
        df: Pandas DataFrame.
        record_type: 'submission' or 'comment' from the schema metadata.
    
    Returns:
        df: Copy of original dataframe with assigned record_type column.
    """
    logger.debug('Assigning "record_type" column for %s dataframe.', record_type)
    return df.assign(record_type=record_type)

def drop_and_rename_text_cols(df):
    """
    Drop all other text columns and assign values to a new 'document' column.
    
    Args:
        df: Pandas DataFrame.
    
    Returns:
        df: DataFrame with assigned 'document' column.
    """
    if 'title' in df.columns and 'selftext' in df.columns:
        logger.debug('Dropping "selftext" and renaming "title" to "document" in submission dataframe.')
        out_df = df.drop(columns=['selftext']).rename(columns={'title': 'document'})
        
        logger.debug('Replacing values in "document" column with merged submission title and selftext.')
        out_df.loc[:, 'document'] = (df.title + ' ' + df.selftext).values
        return out_df
    
    elif 'body' in df.columns:
        logger.debug('Renaming "body" to "document" in comment dataframe.')
        return df.rename(columns={'parent_submission_id': 'submission_id', 'body': 'document'})

def wrangle_dataset(dataset):
    """
    Perform data cleaning and wrangling on an input PyArrow dataset.
    
    Args:
        dataset: PyArrow ParquetDataset.
        
    Returns:
        dataframe: Cleaned and wrangled Pandas DataFrame.
    """
    # Either submission or comment; Get primary keys and columns containing text
    record_type = dataset.schema.metadata[b'record_type'].decode()
    subset = 'submission_id' if record_type == 'submission' else 'comment_id'
    cols = ['title', 'selftext'] if record_type == 'submission' else ['body']
    logger.info('Parsing raw %s dataset.', record_type) 
    
    # Convert dataset to pandas for in-memory transformations
    out_df = dataset_to_pandas(dataset)

    # Deduplication based on column subset
    out_df = deduplicate_pandas(out_df, subset)

    # Remove empty records (records without valid tokens)
    out_df = remove_empty_rows_pandas(out_df, cols)
    
    # Filter unreliable / low-score comments and posts
    out_df = remove_low_score_pandas(out_df, threshold=-2)
    
    # Pad URLs with context tags <URL>
    out_df = replace_url_pandas(out_df, cols)
    
    # Remove extra whitespaces
    out_df = remove_extra_whitespace_pandas(out_df, cols)
    
    # Retain only 'document' column for all text data
    out_df = drop_and_rename_text_cols(out_df)
    
    # Assign record type
    out_df = assign_record_type_pandas(out_df, record_type)
    
    logger.info('Finished preprocessing %s dataset.', record_type) 
    return out_df

def pandas_to_parquet(df, file_name: str, dir_path=None, partition_by_date:bool=False):
    """
    Args:
        df: Pandas DataFrame.
        file_name: Specify either 'comments' or 'submissions'
        dir_path: Path of directory where parquet will be stored. Default is 'data/processed' when None.
        partition_by_date: If True, output file will be saved to 'dir_path/current_date'. 
    
    Notes:
        Export pandas dataframe to parquet.
    """
    # Append .parquet to file_name if it doesn't have extension
    if not re.match(r'\w+\.parquet$', file_name):
        file_name += '.parquet'
    
    if dir_path is None:
        dir_path = get_repo_root() / 'data/processed'
        
    if partition_by_date:
        logger.debug('Partition by date set to True. Modifying target directory path.')
        dir_path /= f'{dt.datetime.now():%Y-%m-%d}'
    
    # Create directory if necessary
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / file_name
    
    logger.debug('Exporting %s to path: %s', file_name, file_path)
    
    df.to_parquet(file_path, engine='pyarrow')
    logger.info('Finished exporting preprocessed Reddit post and comment data to %s', dir_path)