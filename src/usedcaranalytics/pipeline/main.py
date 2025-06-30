#! /usr/bin/env python3

from dotenv import load_dotenv
from praw import Reddit
from usedcaranalytics.pipeline.streamer import DataStreamer
from usedcaranalytics.pipeline.transformer import DataTransformer
from usedcaranalytics.pipeline.loader import ParquetDataLoader
from usedcaranalytics.config.parquet import get_parquet_configs
from usedcaranalytics.utils.txtparser import txt_to_list
from usedcaranalytics.utils.loadenv import load_env

def main():
    """
    Main ETL pipeline script. Extract step involves streaming data from Reddit API
    as submission and comment records in dictionary format. Transform step mainly
    involves cleaning records (i.e. filtering out short/removed/deleted text).
    Load step stores the streamed data into buffers, and batch exports data as *.parquet
    whenever buffer size threshold reached.
    """
    # Run config scripts; generate loader config 
    PRAW_ID, PRAW_SECRET, PRAW_USER_AGENT, PRAW_USERNAME, PRAW_PASSWORD = load_env()
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
    loader = ParquetDataLoader(loader_config, target_MB=256, transformer=transformer)
    
    # Get the list of subreddits and queries
    subreddits = txt_to_list(target_file='subreddits.txt', subdir='data/raw')
    queries = txt_to_list(target_file='search_queries.txt', subdir='data/raw')
    
    # Assign generator to variable
    # Expected 10x10x50 == 5000 submissions, and < 3x comments 
    stream = streamer.stream(subreddits, queries, progress_bar=True, limit=50)
    
    # ETL to disk as *.parquet
    loader.load(stream)
    
if __name__ == "__main__":
    main()