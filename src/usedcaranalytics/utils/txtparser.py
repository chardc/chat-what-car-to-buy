from typing import Union, Tuple, List, Optional
from usedcaranalytics.utils.getpath import get_path

def txt_to_list(target_file: str, subdir: Optional[str]=None):
    """
    Utility function for parsing a multi-line text file where each item 
    is separated by a newline.
    Args:
        - target_file: Name of the *.txt file
        - subdir: Optional path for subdirectories containing target file.
    Returns:
        - List of elements (e.g. search queries, subreddit names).
    """
    # Get absolute file path
    file_path = get_path(__file__, target_file, subdir)
    with open(file_path, 'r') as f:
        # Ignore comments and empty lines
        txt_list = [
            line.rstrip("\n") for line in f 
            if not (line.startswith('#') or line.startswith("\n"))
            ]
    return txt_list
    