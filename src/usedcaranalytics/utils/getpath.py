from pathlib import Path
from typing import Optional

# Full recursion implementation
def get_path(start_path: Path, target_file: str, subdir: Optional[str]=None):
    """
    Recursive search for a given file and returns the absolute path.
    Args:
        - start_path: PosixPath object. If run in main.py, arg == Path(__file__).
        - target_file: Name of target file (e.g. ".env", "subreddits.txt").
        - subdir: [Optional] Name of directory containing the file (e.g. "config").
    Returns:
        - Absolute path of target file as PosixPath object.
    """
    if subdir is None:
        subdir = ''
    
    # Convert to absolute path
    current_path = start_path.resolve()
    
    # Base case: If .env in current path, return
    target_path = current_path / target_file
    if target_path.exists():
        return target_path
    
    # If subdir containing target path exists, search within that dir
    subdir_path = current_path / subdir
    if subdir_path != current_path and subdir_path.exists():
        return get_path(subdir_path, target_file, subdir)
    
    # If we're at the root, raise an Exception
    if current_path == current_path.parent:
        # Handle infinite recursion for when all parent paths exhausted
        raise RuntimeError(
            f'"{target_file}" file not found. Ensure {target_file} is stored in project repo.'
            )
    
    # If we're at the subdirectory and file not there, raise an Exception
    if current_path == current_path.parent / subdir:
        raise RuntimeError(
            f'"{target_file}" file not found.{f' in "{subdir}"' if subdir else ''}.'
            f'Ensure {target_file} is stored in specified sub-directory.'
            )
    
    # Otherwise, move up to parent and repeat search
    return get_path(current_path.parent, target_file, subdir)

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