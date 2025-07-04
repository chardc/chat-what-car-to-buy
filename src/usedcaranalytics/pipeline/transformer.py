import pandas as pd
import logging
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from typing import Union, Tuple, List, Literal

logger = logging.getLogger(__name__)

class DataTransformer():
    def transform(self, table: pa.Table):
        """
        Perform cleaning and preprocessing on current batch's PyArrow Table before writing to disk.
        Converts PyArrow table to Pandas DataFrame to perform quick in-memory transformations,
        and then convert back to PyArrow table for final masking.
        
        Args:
            table: PyArrow table containing either submission or comment data.
        
        Returns:
            table: Transformed PyArrow table.
        """
        logger.info("Starting DataTransformer.transform on batch with %d rows.", table.num_rows)
        
        try:
            # Store current table's schema and record type
            self.current_schema = table.schema
            self.current_record_type = table.schema.metadata[b'data_source'].decode()
            
            # Simple masking transformations made via pyarrow.compute to reduce rows prior to pandas ops
            # Remove all [deleted] and [removed] text
            out_table = self.remove_match_from_table(table)
            # Remove all short text from table
            out_table = self.remove_short_text_from_table(out_table)
            
            # Regex replacements, deduplication, and regex row removals made via pandas
            # Convert table to dataframe
            out_df = self._table_to_pandas(out_table)
            
            # Remove duplicates
            out_df = self.remove_duplicates_from_df(out_df)
            # Replace newlines and nonascii with spaces, and collapse all contiguous spaces
            out_df = self.replace_newlines_nonascii_from_df(out_df)
            # Remove empty rows (empty string or space only)
            out_df = self.remove_empty_rows_from_df(out_df)
            # Lowercase text and id columns
            out_df = self.lowercase_text_from_df(out_df)
            
            # Convert back to PyArrow Table with original schema
            out_table = self._pandas_to_table(out_df)
            
            logger.info('DataTransformer.transform finished. Final row count: %d', out_table.num_rows)
        
        except Exception as e:
            logger.exception('Exception in DataTransformer.transform (returning original table): %s', e)
            return table
        
        else:
            return out_table
    
    def __call__(self, *args, **kwargs):
        """Alias for self.transform. Takes a PyArrow table and returns a transformed copy."""
        return self.transform(*args, **kwargs)
    
    def _pandas_to_table(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe: Pandas DataFrame
        
        Returns:
            table: PyArrow Table with original schema retained.
        """
        logger.debug('Converting Pandas DataFrame to PyArrow Table with schema: %s.', self.current_schema)
        return pa.Table.from_pandas(dataframe, schema=self.current_schema)
    
    def _table_to_pandas(self, table: pa.Table):
        """
        Args:
            table: PyArrow Table.
        
        Returns:
            dataframe: Pandas DataFrame            
        """
        logger.debug('Converting PyArrow Table to Pandas DataFrame from source %s.', self.current_record_type)
        return table.to_pandas(types_mapper=pd.ArrowDtype)
    
    def remove_duplicates_from_df(self, dataframe: pd.DataFrame):
        """
        Remove duplicate records using Pandas. PyArrow currently has no API for deduplication
        unlike Pandas. Duplicate records are determined by primary key = comment / submission id.
        
        Args:
            dataframe: Pandas DataFrame.
        
        Returns:
            dataframe: Deduplicated Pandas DataFrame.
        """
        logger.debug('remove_duplicates_from_df called on data_source: %s', self.current_record_type)
        input_rows = dataframe.shape[0]
        
        if self.current_record_type == 'submission':
            out_df = dataframe.drop_duplicates('submission_id')
        
        elif self.current_record_type == 'comment':
            out_df = dataframe.drop_duplicates('comment_id')
        
        output_rows = out_df.shape[0]
        logger.debug('Rows after remove_duplicates_from_df: %d, (removed %d)', output_rows, input_rows - output_rows)
        return out_df
    
    def lowercase_text_from_df(self, dataframe: pd.DataFrame):
        """
        Converts all text columns and id columns to lowercase. Subreddit is excluded since
        case-sensitive.
        
        Args:
            dataframe: Pandas DataFrame.
        
        Returns:
            dataframe: Pandas DataFrame with all text data in lowercase.
        """
        logger.debug('lowercase_text_from_df called for %s', self.current_record_type)
        out_df = dataframe.copy()
        
        if self.current_record_type == 'submission':
            out_df.loc[:,['submission_id','title','selftext']] = (out_df
                                                                  .loc[:,['submission_id','title','selftext']]
                                                                  .apply(lambda col: col.str.lower())
                                                                  )
        
        elif self.current_record_type == 'comment':
            out_df.loc[:,['comment_id','body','parent_submission_id']] = (out_df
                                                                          .loc[:,['comment_id','body','parent_submission_id']]
                                                                          .apply(lambda col: col.str.lower())
                                                                          )
        
        logger.debug('Finished converting text and id columns to lowercase.')
        return out_df
    
    def replace_newlines_nonascii_from_df(self, dataframe: pd.DataFrame):
        """
        Substitute newlines and non-ASCII characters with spaces, then collapse all contiguous 
        spaces into a single space character. This approach mitigates the risk of word 
        concatenation from character replacements. 
        
        Args:
        dataframe: Pandas DataFrame.
        
        Returns:
        dataframe: Pandas DataFrame with ASCII-only text and no newlines.
        """
        logger.debug('replace_newlines_nonascii_from_df called for %s', self.current_record_type)
        out_df = dataframe.copy()       
        
        # 1) Replace all newlines with a space
        # 2) Replace all non-ASCII characters with a space
        # 3) Replace all contiguous spaces with just one space
        if self.current_record_type == 'submission':
            out_df.loc[:,['title','selftext']] = (out_df
                                                  .loc[:,['title','selftext']]
                                                  .apply(
                                                      lambda col: col.str.replace(r'\n', ' ', regex=True)
                                                      ))
            out_df.loc[:,['title','selftext']] = (out_df
                                                  .loc[:,['title','selftext']]
                                                  .apply(
                                                      lambda col: col.str.replace(r'[^\x00-\x7F]+', ' ', regex=True)
                                                      ))
            out_df.loc[:,['title','selftext']] = (out_df
                                                  .loc[:,['title','selftext']]
                                                  .apply(
                                                      lambda col: col.str.replace(r'\s+', ' ', regex=True)
                                                      ))
        
        elif self.current_record_type == 'comment':
            out_df.loc[:,'body'] = (out_df
                                    .loc[:,'body']
                                    .str.replace(r'\n', ' ', regex=True)
                                    )
            out_df.loc[:,'body'] = (out_df
                                    .loc[:,'body']
                                    .str.replace(r'[^\x00-\x7F]+', ' ', regex=True)
                                    )
            out_df.loc[:,'body'] = (out_df
                                    .loc[:,'body']
                                    .str.replace(r'\s+', ' ', regex=True)
                                    )
            
        logger.debug('Finished replacing newlines "\\n" and non-ASCII characters with empty string.')
        return out_df
    
    def remove_empty_rows_from_df(self, dataframe: pd.DataFrame):
        """
        Removes empty rows from a Pandas DataFrame. Empty rows are defined as rows containing
        only contiguous space characters \s+ or containing empty string ''. Should be done
        after all pattern replacements.
        
        Args:
            dataframe: Pandas DataFrame.
            
        Returns:
            dataframe: Pandas DataFrame without empty rows (either '' or '\s+').
        """
        logger.debug('remove_empty_rows_from_df called for %s', self.current_record_type)
        input_rows = dataframe.shape[0]
        out_df = dataframe.copy()
        
        # Match all empty text rows based on regex
        if self.current_record_type == 'submission':
            empty_rows = (out_df
                          .loc[:,['title','selftext']]
                          .apply(
                              lambda col: col.str.match(r'^\s*$')).any(axis=1)
                          )
        
        elif self.current_record_type == 'comment':
            empty_rows = out_df.loc[:,'body'].str.match(r'^\s*$')
        
        # Filter out the matching text from output dataframe
        out_df = out_df.loc[~empty_rows]
        output_rows = out_df.shape[0]
        logger.debug('Rows after remove_empty_rows_from_df: %d (removed %d)', output_rows, input_rows - output_rows)
        return out_df
    
    def remove_short_text_from_table(
        self,
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
            table: PyArrow table with short text removed.
        """
        logger.debug('remove_short_text_from_table called for %s', self.current_record_type)
        input_rows = table.num_rows
        # Build the masking expression based on current dataset
        # Match all records that pass minimum character count for relevant columns
        if self.current_record_type == 'submission':
            mask_expr = (
                (pc.utf8_length(pc.field('title')) < min_title_length)
                | (pc.utf8_length(pc.field('selftext')) < min_selftext_length)
                )
        elif self.current_record_type == 'comment':
            mask_expr = (pc.utf8_length(pc.field('body')) < min_body_length)
        # Filter out short text from output table
        out_table = table.filter(~mask_expr)
        output_rows = out_table.num_rows
        logger.debug('Rows after remove_short_text_from_table: %d (removed %d)', output_rows, input_rows - output_rows)
        return out_table
    
    def remove_match_from_table(
        self,
        table: pa.Table, regex_patterns: Union[Tuple[str], List[str]]=(r"\[deleted\]",r"\[removed\]"), 
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
        logger.debug('remove_match_from_table called for %s using patterns %s with logic "%s"', self.current_record_type, regex_patterns, match_logic)
        input_rows = table.num_rows
        # Build the regex pattern; Union (OR) by default
        regexp = (r"&".join([pat for pat in regex_patterns]) 
                  if match_logic == 'intersect' 
                  else r"|".join([pat for pat in regex_patterns]))
        # Build the masking expression based on current dataset
        # Match all records that satisfy the regex expression
        if self.current_record_type == 'submission':
            mask_expr = (
                pc.match_substring_regex(pc.field("title"), regexp)
                | pc.match_substring_regex(pc.field("selftext"), regexp)
            )
        elif self.current_record_type == 'comment':
            mask_expr = (
                pc.match_substring_regex(pc.field("body"), regexp)
            )
        # Filter out matching text from output table
        out_table = table.filter(~mask_expr)
        output_rows = out_table.num_rows
        logger.debug('Rows after remove_match_from_table: %d (removed %d)', output_rows, input_rows - output_rows)
        return out_table