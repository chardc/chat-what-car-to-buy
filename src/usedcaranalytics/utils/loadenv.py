from usedcaranalytics.utils.getpath import get_path
from dotenv import load_dotenv
from pathlib import Path

def load_env():
    """
    Abstraction of loading environment variables in main ETL script.
    Args:
        - None.
    Returns:
        - Tuple of client id, client secret, reddit password, user agent,
        and reddit username.
    Side effect:
        - Loads environment variables from .env file.
    """
    # Load environment variables using absolute .env path
    env_path = get_path(Path(__file__), subdir='config', target_file='.env')
    load_dotenv(env_path)
    # Import credentials initialized by config.api module
    from usedcaranalytics.config.api import PRAW_ID, PRAW_SECRET, PRAW_PASSWORD, PRAW_USER_AGENT, PRAW_USERNAME 
    # Return tuple in same arrangement as praw.Reddit constructor kwargs
    return PRAW_ID, PRAW_SECRET, PRAW_PASSWORD, PRAW_USER_AGENT, PRAW_USERNAME 