from pathlib import Path
from typing import Optional, Union, Tuple

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

def get_repo_root(
    start_path: Union[str, Path]=__file__, 
    sentinels: Tuple[str]=('.git','pyproject.toml','setup.py','README.md')
    ):
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
    
# # Recursive iteration implementation, only works when target file provided 
# # and in immediate parent. Else, requires a subdir to work
# def get_path(start_path: Path, target_file: str, subdir: str=None):
#     """
#     Recursive search for a given file and returns the absolute path.
#     Args:
#         - start_path: PosixPath object. If run in main.py, arg == Path(__file__).
#         - target_file: Name of target file (e.g. ".env", "subreddits.txt").
#         - subdir: Name of directory containing the file (e.g. "config").
#     Returns:
#         - Absolute path of target file as PosixPath object.
#     """
#     if subdir is None:
#         subdir = ''
    
#     if target_file is None:
#         target_file = ''
    
#     # Convert to absolute path
#     current_path = start_path.resolve()
    
#     # Base case: If .env in current path, return
#     target_path = current_path / target_file
#     if target_path.exists():
#         return target_path
    
#     # If config directory exists in parent path, search within config
#     for parent in [current_path, *current_path.parents]:
#         # Handle infinite recursion when file not found in subdir
#         if current_path == parent.parent / subdir:
#             subdir = current_path
#             break
#         subdir_path = parent / subdir
#         if subdir_path.exists():
#             return get_path(subdir_path, target_file, subdir)
    
#     # Handle infinite recursion for when all parent paths exhausted
#     raise RuntimeError(
#         f'"{target_file}" file not found.{f' in "{subdir}"' if subdir else ''}\n'
#         f'Ensure {target_file} is stored in project repo and inside specified sub-directory.'
#         )  