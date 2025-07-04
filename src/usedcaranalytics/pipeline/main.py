#! /usr/bin/env python3

from tracemalloc import start
from dotenv import load_dotenv
from praw import Reddit
from usedcaranalytics.pipeline.streamer import DataStreamer
from usedcaranalytics.pipeline.transformer import DataTransformer
from usedcaranalytics.pipeline.loader import ParquetDataLoader
from usedcaranalytics.config.parquet_config import get_parquet_configs
from usedcaranalytics.config.logging_config import setup_logging
from usedcaranalytics.utils.txtparser import txt_to_list
from usedcaranalytics.utils.loadenv import load_env
from usedcaranalytics.utils.getpath import get_path

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
        Writes *.parquet files to project_root/data/processed/**-dataset directories.
    """
    # Setup logger, default=logging.DEBUG
    setup_logging()
    
    # Run config scripts; generate loader config 
    PRAW_ID, PRAW_SECRET, PRAW_PASSWORD, PRAW_USER_AGENT, PRAW_USERNAME = load_env()
    
    # Try getting schema from schemas.json in config dir, otherwise, return default.
    try:
        loader_config = get_parquet_configs(
            schema_path=get_path(start_path=__file__, target='schemas.json')
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
    loader = ParquetDataLoader(loader_config, target_MB=16, transformer=transformer) # Smaller target MB for frequent writes
    
    # Get the list of subreddits and queries
    subreddits = txt_to_list(target_file='subreddits.txt', subdir='data/raw')
    queries = txt_to_list(target_file='search_queries.txt', subdir='data/raw')
    
    # Assign generator to variable, default submission limit = 50
    # Yield: Submissions <= 10*10*50==5000; Comments >= ~5*Submissions 
    stream = streamer.stream(**stream_kwargs, subreddits=subreddits, queries=queries, progress_bar=True)
    
    # ETL to disk as *.parquet
    loader.load(stream)
    
if __name__ == "__main__":
    main(limit=10)
