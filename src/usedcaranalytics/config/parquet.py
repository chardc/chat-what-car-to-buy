from pathlib import Path
from collections import namedtuple
from typing import Optional, Union, Tuple, List
import pyarrow as pa
import json

## Initialize namedtuple for Parquet Config for ease of access and immutability
ParquetConfig = namedtuple('ParquetConfig',['record_type','dataset_path','schema'])

def _load_schema(path: Union[str, Path], record_type: str) -> pa.Schema:
    """
    Returns a pyarrow schema parsed from a JSON file containing the schema.
    Args:
        - Path to JSON schema. Parent object must be either 'submission' or 'comment'
        and child object must be a dictionary containing column name keys and pyarrow datatype values
        Ex. 'submission' : {'submission_id' : 'pa.string()'}
    Returns:
        - PyArrow Schema object
    """
    if (isinstance(path, Path) and not path.exists()) or (not Path(path).exists()):
        raise ValueError(f"{path} doesn't exist.")
        
    with open(path, 'r') as f:
        schemas = json.load(f)
        # lowercase keys and column names for safety
        schemas = {key.lower() : {col.lower() : dtype for col, dtype in value.items()} 
            for key, value in schemas.items()}
        return pa.schema(
            [(col, eval(datatype)) for col, datatype in schemas[record_type].items()], 
            metadata={'data_source':record_type}
            )

def get_submission_schema(schema_path:Optional[str]=None) -> pa.Schema:
    """
    Returns a boilerplate pyarrow schema for submission data. An optional schema path
    can be provided to return a predefined schema.
    Args:
        - [Optional] path to JSON schema. Parent object must be either 'submission' or 'comment'
        and child object must be a dictionary containing column name keys and pyarrow datatype values
        Ex. 'submission' : {'submission_id' : 'pa.string()'}
    Returns:
        - PyArrow Schema object
    """
    if schema_path:
        return _load_schema(schema_path, 'submission')
    return pa.schema(
        [
            ("submission_id", pa.string()),
            ("title", pa.string()),
            ("selftext", pa.string()),
            ("score", pa.int64()),
            ("upvote_ratio", pa.float64()),
            ("timestamp", pa.timestamp("s")),
            ("subreddit", pa.string()),
            ("num_comments", pa.int64())
        ],
        metadata={'data_source':'submission'}
        )
    
def get_comment_schema(schema_path:Optional[str]=None) -> pa.Schema:
    """
    Returns a boilerplate pyarrow schema for submission data. An optional schema path
    can be provided to return a predefined schema.
    Args:
        - [Optional] path to JSON schema. Parent object must be either 'submission' or 'comment'
        and child object must be a dictionary containing column name keys and pyarrow datatype values
        Ex. 'submission' : {'submission_id' : 'pa.string()'}
    Returns:
        - PyArrow Schema object
    """
    if schema_path:
        return _load_schema(schema_path, 'comment')
    return pa.schema(
        [
            ("comment_id", pa.string()),
            ("body", pa.string()),
            ("score", pa.int64()),
            ("timestamp", pa.timestamp("s")),
            ("subreddit", pa.string()),
            ("parent_submission_id", pa.string())
        ],
        metadata={'data_source':'comment'}
        )

def get_parquet_configs(
    root: Union[str, Path]=None, 
    subdir: Union[str, Path]=None, 
    dataset_dirs: Union[Tuple[str, str], List[str]]=('submission-dataset','comment-dataset'),
    **schema_kwargs
    ):
    """
    Utility function to generate tuple of ParquetConfig namedtuples for submission and comment
    datasets respectively.
    Args:
        - Root path for repo
        - Data subdirectory path (default = data/processed).
        - Dataset directory names in order: 1) submission dataset, 2) comment dataset 
        (default = ('submission-dataset','comment-dataset')).
    Returns:
        - Tuple of namedtuples (sub_cfg, com_cfg) containing record_type, dataset_path, and
        schema attribtues
    """
    
    # Default root/subdir path: UsedCarAnalytics/data/processed
    if root is None:
        root = get_repo_root()
    elif isinstance(root, str):
        root = Path(root)
    
    if subdir is None:
        subdir = Path('data') / 'processed'
    if isinstance(subdir, str):
        subdir = Path(subdir)
    
    # Input validation for dataset directory names; 
    # non-list or tuple args will be catched by builtin exceptions 
    if len(dataset_dirs) != 2:
        raise ValueError("dataset_dirs must be a tuple or list of 2 directory names for submission and comment data, respectively.")
    
    # Build paths for submission dataset and comment dataset directories
    sub_path, com_path = tuple(
        root / subdir / dataset_dir
        for dataset_dir in dataset_dirs
    )
    
    # Get the submission and content data Arrow & Parquet schemas
    sub_schema = get_submission_schema(**schema_kwargs)
    com_schema = get_comment_schema(**schema_kwargs)

    # Build the ParquetConfig files and return as tuple
    return tuple(
        ParquetConfig(record_type, path, schema) 
        for record_type, path, schema 
        in zip(
            ('submission','comment'),
            (sub_path, com_path),
            (sub_schema, com_schema)
            )
        )

def get_repo_root(
    start_path: Union[str, Path]=None, 
    sentinels: Tuple[str]=('.git','pyproject.toml','setup.py','README.md')
    ):
    """Finds the project root by walking up each parent directory until sentinel is found."""
    if start_path is None:
        start_path = Path(__file__)
    elif isinstance(start_path, str):
        start_path = Path(start_path)
    # Get absolute path of current file
    current_path = start_path.resolve()
    # Iterate through all parents until marker is found
    for parent in [current_path, *current_path.parents]:
        # Return parent path if any of the sentinels is matched
        if any((parent / sentinel).exists() for sentinel in sentinels):
            return parent
    raise RuntimeError('Project root not found. Try specifying a different sentinel (marker).')