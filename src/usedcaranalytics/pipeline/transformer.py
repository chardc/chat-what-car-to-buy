import re
import pandas as pd
import logging
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from typing import Union, Tuple, List, Literal

logger = logging.getLogger(__name__)

class DataTransformer():
    def transform(self, table: pa.Table):
        """
        Perform cleaning and preprocessing on current batch's PyArrow Table before writing to disk.
        
        Args:
            table: PyArrow table containing either submission or comment data.
        
        Returns:
            table: Transformed PyArrow table.
        """
        logger.info("Starting DataTransformer.transform on batch with %d rows.", table.num_rows)
        try:
            # Duplicates mainly come from duplicated submissions when searching
            out_table = self.remove_duplicates(table) 
            # Filter out [removed] and [deleted] text
            out_table = self.remove_matching_text( 
                out_table, regex_patterns=(r"\[deleted\]",r"\[removed\]")
                ) 
            # Clean the text data and convert to lowercase
            out_table = self.remove_newlines_nonascii_lowercase(out_table)
            # Remove extremely short text data
            out_table = self.remove_short_text(out_table) 
            logger.info('DataTransformer.transform finished. Final row count: %d', out_table.num_rows)
        
        except Exception as e:
            logger.exception('Exception in DataTransformer.transform (returning original table): %s', e)
            return table
        
        else:
            return out_table
    
    def __call__(self, *args, **kwargs):
        """Alias for self.transform. Takes a PyArrow table and returns a transformed copy."""
        return self.transform(*args **kwargs)
    
    def remove_duplicates(table: pa.Table):
        """
        Remove duplicate records by converting PyArrow Table to Pandas and leveraging 
        built-in methods. 
        
        Args:
            table: PyArrow table.
        
        Returns:
            table: Deduplicated PyArrow table.
        """
        logger.debug('remove_duplicates called on data_source: %s', table.schema.metadata[b'data_source'])
        input_rows = table.num_rows
        # PyArrow currently has no API for convenient deduplication but has
        # good integration with Pandas. Conversion to DataFrame is quick and easy.
        if table.schema.metadata[b'data_source'] == b'submission':
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            df = df.drop_duplicates('submission_id')
        elif table.schema.metadata[b'data_source'] == b'comment':
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            df = df.drop_duplicates('comment_id')
        output_rows = df.shape[0]
        logger.debug('Removed %d duplicates (%d -> %d)', input_rows - output_rows, input_rows, output_rows)
        # Convert DataFrame back to PyArrow Table with schema from the original table
        return pa.Table.from_pandas(df, schema=table.schema)
    
    @staticmethod
    def remove_matching_text(
        table: pa.Table, regex_patterns: Union[Tuple[str], List[str]], 
        match_logic: Literal['union', 'intersect']='union'
        ):
        """
        Filter out matching text. By default, deleted and removed text are filtered out from 
        the submission title, submission selftext, and comment body.
        
        Args:
            table: PyArrow Table for either submission or comment data.
            regex_patterns: Tuple or list of regex patterns for matching.
            match_logic: Union if text can match either of the patterns (OR), Intersect 
            if text must match all (AND).
        
        Returns:
            table: PyArrow table with matching text records removed.
        """
        logger.debug(
            'remove_matching_text called for %s using patterns %s with logic "%s"',
            table.schema.metadata[b'data_source'], regex_patterns, match_logic
            )
        input_rows = table.num_rows
        # Build the regex pattern; Union (OR) by default
        regexp = (r"&".join([pat for pat in regex_patterns]) 
                  if match_logic == 'intersect' 
                  else r"|".join([pat for pat in regex_patterns]))
        # Build the masking expression based on current dataset
        # Match all records that satisfy the regex expression
        data_source = table.schema.metadata[b'data_source']
        if data_source == b'submission':
            mask_expr = (
                pc.match_substring_regex(pc.field("title"), regexp)
                | pc.match_substring_regex(pc.field("selftext"), regexp)
            )
        elif data_source == b'comment':
            mask_expr = (
                pc.match_substring_regex(pc.field("body"), regexp)
            )
        out_table = table.filter(~mask_expr)
        output_rows = out_table.num_rows
        logger.debug('Rows after remove_matching_text: %d (removed %d)', output_rows, input_rows - output_rows)
        # Return the table without matching text
        return out_table
    
    @staticmethod
    def remove_newlines_nonascii_lowercase(table: pa.Table):
        """
        Convert text to lowercase, substitute newlines and non-ascii characters with spaces, 
        then collapse all contiguous spaces into a single space character. This approach
        mitigates the risk of word concatenation from character replacements.
        
        Args:
        table: PyArrow table.
        
        Returns:
        table: PyArrow table with ASCII-only text data.
        """
        logger.debug('remove_newlines_nonascii_lowercase called for %s', table.schema.metadata[b'data_source'])
        input_rows = table.num_rows
        df = table.to_pandas(types_mapper=pd.ArrowDtype)
        logger.debug('Converting text to lowercase, substituting newlines and non-ASCII characters, and collapsing spaces...')
        if table.schema.metadata[b'data_source'] == b'submission':
            df.loc[:,['title','selftext']] = df.loc[:,['title','selftext']].apply(lambda col: col.str.lower())
            df.loc[:,['title','selftext']] = df.loc[:,['title','selftext']].apply(lambda col: col.str.replace(r'\n', ' ', regex=True))
            df.loc[:,['title','selftext']] = df.loc[:,['title','selftext']].apply(lambda col: col.str.replace(r'[^\x00-\x7F]+', ' ', regex=True))
            df.loc[:,['title','selftext']] = df.loc[:,['title','selftext']].apply(lambda col: col.str.replace(r'\s+', ' ', regex=True))
        elif table.schema.metadata[b'data_source'] == b'comment':
            df.loc[:,'body'] = df.loc[:,'body'].str.lower()
            df.loc[:,'body'] = df.loc[:,'body'].str.replace(r'\n', ' ', regex=True)
            df.loc[:,'body'] = df.loc[:,'body'].str.replace(r'[^\x00-\x7F]+', ' ', regex=True)
            df.loc[:,'body'] = df.loc[:,'body'].str.replace(r'\s+', ' ', regex=True)
        out_table = pa.Table.from_pandas(df, schema=table.schema)
        output_rows = out_table.num_rows
        logger.debug('Rows after remove_newlines_nonascii_lowercase: %d (removed %d)', output_rows, input_rows - output_rows)
        # Return the table with cleaned text data
        return out_table
    
    @staticmethod
    def remove_short_text(
        table: pa.Table, min_title_length: int=15, 
        min_selftext_length: int=50, min_body_length: int=15
        ):
        """
        Filter out short text from submission and comment dataset to retain only 
        contextually rich textual data.
        
        Args:
            table: PyArrow Table for either submission or comment data.
            min_title_length: Int >= 0 for minimum title text length (UTF-8).
            min_selftext_length: Int >= 0 for minimum post selftext length (UTF-8).
            min_body_length: Int >= 0 for minimum comment body length (UTF-8).
        
        Returns:
            table: PyArrow table with short text records removed.
        """
        logger.debug('remove_short_text called for %s', table.schema.metadata[b'data_source'])
        input_rows = table.num_rows
        # Build the masking expression based on current dataset
        # Match all records that pass minimum character count for relevant columns
        data_source = table.schema.metadata[b'data_source']
        if data_source == b'submission':
            mask_expr = (
                (pc.utf8_length(pc.field('title')) < min_title_length) | 
                (pc.utf8_length(pc.field('selftext')) < min_selftext_length)
                )
        elif data_source == b'comment':
            mask_expr = (pc.utf8_length(pc.field('body')) < min_body_length)
        out_table = table.filter(~mask_expr)
        output_rows = out_table.num_rows
        logger.debug('Rows after remove_short_text: %d (removed %d)', output_rows, input_rows - output_rows)
        # Return the table without short text
        return out_table