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
import logging
import datetime as dt
import pandas as pd
import pyarrow.parquet as pq
from typing import Hashable, List, Literal
from chatwhatcartobuy.utils.getpath import get_repo_root

logger = logging.getLogger(__name__)

def read_dataset(path, **kwargs):
    """
    Parses multiple *.parquet files into a single dataset.
    
    Args:
        path: Filepath to the dataset directory that contains parquet files.
        kwargs: pyarrow.parquet.ParquetDataset keyword arguments.
        
    Returns:
        dataset: pyarrow.parquet ParquetDataset.
    """
    dataset = pq.ParquetDataset(path, **kwargs)
    return dataset

def dataset_to_pandas(dataset):
    """
    Wrapper for dataset.read().to_pandas().
    
    Args:
        dataset: ParquetDataset object.
        
    Returns:
        df: Pandas DataFrame.
    """
    return dataset.read().to_pandas()

def deduplicate_pandas(df, subset: Hashable=None):
    """
    Args:
        df: Pandas dataframe.
    
    Returns:
        df: Deduplicated copy of the original dataframe.
    """
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
    out_df.loc[:, cols] = (out_df.loc[:, cols]
                           .apply(lambda col: col.str.lower())
                           )
    return out_df

def assign_data_source_pandas(df, data_source):
    """
    Assigns a new column to the current dataframe containing the data source
    from the pyarrow schema metadata. Useful for filtering records downstream.
    
    Args:
        df: Pandas DataFrame.
        data_source: 'submission' or 'comment' from the schema metadata.
    
    Returns:
        df: Copy of original dataframe with assigned data_source column.
    """
    return df.assign(data_source=data_source)

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
    out_df = dataset_to_pandas(dataset)

    # Deduplication based on column subset
    logger.debug('Deduplicating %s dataframe.', data_source)
    out_df = deduplicate_pandas(out_df, subset)

    # Remove empty records (records without valid tokens)
    logger.debug('Removing empty rows from %s dataframe.', data_source)
    out_df = remove_empty_rows_pandas(out_df, cols)
    
    # Filter unreliable / low-score comments and posts
    logger.debug('Removing low score records from %s dataframe.', data_source)
    out_df = remove_low_score_pandas(out_df, threshold=-2)
    
    # Pad URLs with context tags <URL>
    logger.debug('Padding URLs in %s from %s dataframe.', ', '.join(cols), data_source)
    out_df = replace_url_pandas(out_df, cols)
    
    # Remove extra whitespaces
    logger.debug('Removing extra whitespaces from %s dataframe.', data_source)
    out_df = remove_extra_whitespace_pandas(out_df, cols)
    
    logger.info('Finished preprocessing %s dataset.', data_source) 
    return out_df

def pandas_to_parquet(df, data_source: Literal['comments', 'submissions'], dir_path=None, partition_by_date:bool=False):
    """
    Args:
        df: Pandas DataFrame.
        data_source: Specify either 'comments' or 'submissions'
        dir_path: Path of directory where parquet will be stored. Default is 'data/processed' when None.
        partition_by_date: If True, output file will be saved to 'dir_path/current_date'. 
    
    Notes:
        Export pandas dataframe to parquet.
    """
    if data_source not in ['comments', 'submissions']:
        raise ValueError('Data source can only be either "comments" or "submissions".')
    
    if dir_path is None:
        dir_path = get_repo_root() / 'data/processed'
        
    if partition_by_date:
        logger.debug('Partition by date set to True. Modifying target directory path.')
        dir_path = dir_path / f'{dt.datetime.now():%Y-%m-%d}'
    
    # Create directory if necessary
    dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.debug('Exporting %s to disk path: %s', data_source, dir_path / f'{data_source}.parquet')
    
    df.to_parquet(dir_path / f'{data_source}.parquet', engine='pyarrow')
    logger.info('Finished exporting preprocessed Reddit post and comment data to %s', dir_path)