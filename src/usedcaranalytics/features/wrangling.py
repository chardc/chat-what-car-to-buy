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

from typing import Hashable
import pyarrow.parquet as pq
import pandas as pd
from usedcaranalytics.utils.getpath import get_repo_root

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

def remove_empty_rows_pandas(df, cols: list, pat: str=r'\b\w{2,}'):
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

def replace_text_pandas(df, col: str, pat: str, repl: str):
    """
    Remove or replace text with context tags.
    
    Args:
        df: Pandas dataframe.
        col: Column name containing text for replacement.
        pat: Regex pattern.
        repl: Replacement string.
    
    Returns:
        df: Copy of original dataframe with replaced or padded string.
    """
    out_df = df.copy()
    out_df.loc[:, col] = out_df.loc[:, col].str.replace(pat, repl, regex=True)
    return out_df

def replace_url_pandas(df, col: str, pat: str=r'[\[\(]?https?://[\S]+[\]\)]?', repl: str='<URL>'):
    """
    Wrapper for replace_text_pandas configured to replace URLs. Default pads links with <URL>.
    
    Args:
        df: Pandas dataframe.
        col: Column name containing text records.
        pat: Regex pattern for URL.
        repl: Replacement string for URL. Default is <URL>.
        
    Returns:
        df: Copy of original dataframe with padded URLs.
    """
    return replace_text_pandas(df, col, pat, repl)

def remove_extra_whitespace_pandas(df, col: str):
    """
    Wrapper for replace_text_pandas configured to remove contiguous extra whitespaces
    with a single space.
    
    Args:
        df: Pandas dataframe.
        col: Column name containing text records.
        
    Returns:
        df: Copy of original dataframe with extra whitespaces removed.
    """
    # Only replace contiguous whitespaces with single space.
    return replace_text_pandas(df, col, pat=r'\s{2,}', repl=' ')

def lowercase_text_pandas(df, cols: list):
    """
    Args:
        df: Pandas dataframe.
        cols: List of column names containing text records.
    
    Returns:
        df: Copy of original dataframe with text records in lowercase.
    """
    out_df = df.copy()
    out_df.loc[:, cols] = out_df.loc[:, cols].apply(lambda col: col.str.lower())
    return out_df