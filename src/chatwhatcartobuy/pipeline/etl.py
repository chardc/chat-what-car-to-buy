#! /usr/bin/env python3

import logging
from praw import Reddit
from dotenv import load_dotenv
from chatwhatcartobuy.pipeline.streamer import DataStreamer
from chatwhatcartobuy.pipeline.transformer import DataTransformer
from chatwhatcartobuy.pipeline.loader import ParquetDataLoader
from chatwhatcartobuy.config.parquet_config import get_parquet_configs
from chatwhatcartobuy.config.logging_config import setup_logging
from chatwhatcartobuy.utils.txtparser import txt_to_list
from chatwhatcartobuy.utils.getpath import get_path

logger = logging.getLogger(__name__)

def load_api_keys():
    """Load API keys from .env file in config directory."""
    env_path = get_path(__file__, '.env', 'config')
    logger.debug('.env loaded from %s', str(env_path))
    load_dotenv(env_path)
    logger.debug('PRAW API keys successfully loaded.')

def main(**stream_kwargs):
    """
    Main ETL pipeline script. Extract step involves streaming data from Reddit API
    as submission and comment records in dictionary format. Transform step mainly
    involves cleaning records (i.e. filtering out short/removed/deleted text).
    Load step stores the streamed data into buffers, and batch exports data as *.parquet
    whenever buffer size threshold reached.
    
    Args:
        stream_kwargs (Optional): Keyword arguments to control the stream method call.
        Useful for limiting the results per subreddit search or enabling/disabling
        progress bars.
        
    Notes:
        Writes *.parquet files to project_root/data/raw/**-dataset directories.
    """
    # Setup logger with file handler
    setup_logging(level=logging.INFO, file_prefix='etl-pipeline', output_to_file=True)
    
    load_api_keys()
    
    from chatwhatcartobuy.config.reddit_api_config import PRAW_ID, PRAW_SECRET, PRAW_USER_AGENT, PRAW_USERNAME, PRAW_PASSWORD
    
    # Try getting schema from schemas.json in config dir, otherwise, return default.
    # Save the data to root/data/raw/**-dataset
    try:
        loader_config = get_parquet_configs(
            schema_path=get_path(start_path=__file__, target='schemas.json', subdir='config')
            )
    except:
        loader_config = get_parquet_configs()
    
    # Initialize reddit using API Keys and Reddit login credentials
    reddit = Reddit(
        client_id=PRAW_ID,
        client_secret=PRAW_SECRET,
        password=PRAW_PASSWORD,
        user_agent=PRAW_USER_AGENT,
        username=PRAW_USERNAME,
    )
    
    # Initialize the ETL objects
    streamer = DataStreamer(reddit)
    transformer = DataTransformer()
    loader = ParquetDataLoader(loader_config, target_MB=128, transformer=transformer)
    
    # Get the list of subreddits and queries
    subreddits = txt_to_list(target_file='subreddits.txt', subdir='data/queries')
    queries = txt_to_list(target_file='search_queries.txt', subdir='data/queries')
    
    # Assign generator to variable, default submission limit = 50
    # Yield: Submissions <= 10*10*50==5000; Comments >= ~5*Submissions 
    stream = streamer.stream(**stream_kwargs, subreddits=subreddits, queries=queries, progress_bar=True)
    
    # ETL to disk as *.parquet
    loader.load(stream, partition_by_date=True)
    
if __name__ == "__main__":
    main()
