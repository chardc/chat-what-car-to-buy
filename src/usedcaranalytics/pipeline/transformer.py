import pyarrow as pa, pyarrow.compute as pc
from typing import Union, Tuple, List, Literal

class DataTransformer():
    def transform(self, table: pa.Table):
        """
        Perform cleaning and preprocessing on current batch's PyArrow Table before writing to disk.
        Args:
            table: PyArrow table containing either submission or comment data.
        Returns:
            Wrangled PyArrow table.
        """
        try:
            out_table = self.remove_short_text(table)
            out_table = self.remove_matching_text(out_table)
        except Exception as e:
            print(f'Exception ignored: {e}. Returning untampered PyArrow table.')
            return table
        else:
            return out_table
    
    def __call__(self, *args, **kwargs):
        """Alias for self.transform. Takes a PyArrow table and returns a transformed copy."""
        return self.transform(*args **kwargs)
    
    @staticmethod
    def remove_short_text(
        table: pa.Table, 
        min_title_length: int=20, 
        min_selftext_length: int=100, 
        min_body_length: int=20
        ):
        """
        Filter out short text from submission and comment dataset to retain only contextually rich textual data.
        Args:
            table: PyArrow Table for either submission or comment data
            min_title_length: Int >= 0 for minimum title text length (UTF-8)
            min_selftext_length: Int >= 0 for minimum post selftext length (UTF-8)
            min_body_length: Int >= 0 for minimum comment body length (UTF-8)
        Returns:
            PyArrow table with short text records removed.
        """
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
        # Return the table without short text
        return table.filter(~mask_expr)
    
    @staticmethod
    def remove_matching_text(
        table: pa.Table, 
        regex_patterns: Union[Tuple[str], List[str]]=(r"\[deleted\]",r"\[removed\]"), 
        match_logic: Literal['union', 'intersect']='union'
        ):
        """
        FIlter out matching text. By default, deleted and removed text are filtered out from the submission title,
        submission selftext, and comment body.
        Args:
            table: PyArrow Table for either submission or comment data.
            regex_patterns: Tuple or list of regex patterns for matching.
            match_logic: Union if text can match either of the patterns (OR), Intersect if text must match all (AND).
        Returns:
            PyArrow table with matching text records removed.
        """
        # Build the regex pattern; Union (OR) by default
        regexp = r"&".join([pat for pat in regex_patterns]) if match_logic == 'intersect' else r"|".join([pat for pat in regex_patterns])
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
        # Return the table without matching text
        return table.filter(~mask_expr)