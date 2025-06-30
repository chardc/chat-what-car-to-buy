from pathlib import Path

def get_path(start_path: Path, target_file: str, subdir: str=None):
    """
    Recursive search for a given file and returns the absolute path.
    Args:
        - start_path: PosixPath object. If run in main.py, arg == Path(__file__).
        - target_file: Name of target file (e.g. ".env", "subreddits.txt").
        - subdir: Name of directory containing the file (e.g. "config").
    Returns:
        - Absolute path of target file.
    """
    if subdir is None:
        subdir = ''
    
    if target_file is None:
        target_file = ''
    
    # Convert to absolute path
    current_path = start_path.resolve()
    
    # Base case: If .env in current path, return
    target_path = current_path / target_file
    if target_path.exists():
        return target_path
    
    # If config directory exists in parent path, search within config
    for parent in [current_path, *current_path.parents]:
        # Handle infinite recursion when file not found in subdir
        if current_path == parent.parent / subdir:
            subdir = current_path
            break
        subdir_path = parent / subdir
        if subdir_path.exists():
            return get_path(subdir_path, target_file, subdir)
    
    # Handle infinite recursion for when all parent paths exhausted
    raise RuntimeError(
        f'"{target_file}" file not found.{f' in "{subdir}"' if subdir else ''}\n'
        f'Ensure {target_file} is stored in project repo and inside specified sub-directory.'
        )  