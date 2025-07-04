from usedcaranalytics.utils.getpath import get_path
from dotenv import load_dotenv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_env():
    """
    Abstraction of loading environment variables in main ETL script.
    
    Returns:
        tuple: Contains (client id, client secret, reddit password, user agent,
        reddit username).
    
    Notes:
        Loads environment variables from .env file.
    """
    # Load environment variables using absolute .env path
    env_path = get_path(start_path=__file__, subdir='config', target='.env')
    
    logger.info('Loading environment variables containing Reddit API keys and OAuth login credentials.')
    logger.debug('.env loaded from %s', str(env_path))
    
    load_dotenv(env_path)
    
    # Import credentials initialized by config.api module
    from usedcaranalytics.config.api_config import PRAW_ID, PRAW_SECRET, PRAW_USER_AGENT, PRAW_USERNAME, PRAW_PASSWORD
    
    # Returns in the same order as it's used in praw.Reddit() instantiation
    return PRAW_ID, PRAW_SECRET, PRAW_PASSWORD, PRAW_USER_AGENT, PRAW_USERNAME