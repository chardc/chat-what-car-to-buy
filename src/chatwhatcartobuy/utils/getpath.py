from pathlib import Path
from typing import Callable, Optional, Union, Tuple

# Full recursion implementation
def get_path(start_path: Union[str, Path], target: str, subdir: Optional[str]=None):
    """
    Recursive search for a given file and returns the absolute path.
    Args:
        - start_path: PosixPath object. If run in main.py, arg == Path(__file__).
        - target: Name of target file/directory (e.g. ".env", "subreddits.txt").
        - subdir: [Optional] Name of directory containing the file (e.g. "config").
    Returns:
        - Absolute path of target file as PosixPath object.
    """
    if isinstance(start_path, str):
        start_path = Path(start_path)
    
    if subdir is None:
        subdir = ''
    
    # Convert to absolute path
    current_path = start_path.resolve()
    
    # Base case: If .env in current path, return
    target_path = current_path / target
    if target_path.exists():
        return target_path
    
    # If subdir containing target path exists, search within that dir
    subdir_path = current_path / subdir
    if subdir_path != current_path and subdir_path.exists():
        return get_path(subdir_path, target, subdir)
    
    # If we're at the root, raise an Exception
    if current_path == current_path.parent:
        # Handle infinite recursion for when all parent paths exhausted
        raise RuntimeError(
            f'"{target}" file not found. Ensure {target} is stored in project repo.'
            )
    
    # If we're at the subdirectory and file not there, raise an Exception
    if current_path == current_path.parent / subdir:
        raise RuntimeError(
            f'"{target}" file not found.{f' in "{subdir}"' if subdir else ''}.'
            f'Ensure {target} is stored in specified sub-directory.'
            )
    
    # Otherwise, move up to parent and repeat search
    return get_path(current_path.parent, target, subdir)

def get_repo_root(start_path: Union[str, Path]=__file__, sentinels: Tuple[str]=('.git','pyproject.toml','setup.py','README.md')):
    """
    Wrapper for get_path configured to always return project root by determining parent
    path of any of the sentinel files usually contained directly in root directory.
    Args:
        - start_path: Default = __file__ or current file path.
        - sentinels: Tuple of file names or directories immediately inside root directory.
    Returns:
        - Project root path.
    """
    # Return the file path of the first successful call
    for sentinel in sentinels:        
        try:
            file_path = get_path(start_path, sentinel)
        except RuntimeError:
            continue
        else:
            return file_path.parent
    # If interpreter reaches here, then raise an error
    raise RuntimeError('Project root not found due to missing markers. Retry with updated markers.')
    
def get_latest_path(search_pat: str):
    """
    Returns the path of the latest file or directory inside the target directory.
    
    Args:
        search_pat: PosixPath pattern for the container directory / file itself.
        
    Returns:
        latest_fp: Path of the latest directory or file within the search directory.
    """
    # Return only the first element (i.e. latest path)
    paths = list(get_repo_root().rglob(search_pat))
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)[0]

def get_earliest_path(search_pat: str):
    """
    Returns the path of the earliest file or directory inside the target directory.
    
    Args:
        search_pat: PosixPath pattern for the container directory / file itself.
        
    Returns:
        earliest_fp: Path of the earliest directory or file within the search directory.
    """
    # Return only the last element (i.e. latest path)
    paths = list(get_repo_root().rglob(target_pat))
    return sorted(paths, key=lambda path: path.stat().st_mtime)[0]