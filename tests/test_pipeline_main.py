import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
from dataclasses import dataclass
from usedcaranalytics.pipeline.main import main, get_parquet_configs
from usedcaranalytics.pipeline.transformer import DataTransformer
import pyarrow as pa, pyarrow.compute as pc

@dataclass
class Submission:
    """Mock class for PRAW Submission."""
    id: str
    title: str
    selftext: str
    score: int
    upvote_ratio: float
    created_utc: int # unix time in seconds
    subreddit_name_prefixed: str
    num_comments: int
    
    class Comments:
        """
        Mocks the submission.comments.replace_more method and iterator implem
        for 'yield from'.
        """
        def __init__(self, comments: list=None):
            self.values = comments if comments else []
        
        def __iter__(self):
            return iter(self.values)
        
        @staticmethod
        def replace_more(**kwargs):
            return None
    
    def __post_init__(self):
        self.comments = self.Comments()
    
    @classmethod
    def from_pandas(cls, dataframe: pd.DataFrame):
        """Return a list of Submission objects from a Pandas DataFrame."""
        # Convert timestamp from dt to int 's'
        df = dataframe.copy()
        df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        df['timestamp'] = df['timestamp'].astype('int64')
        return [cls(*record) for record in df.itertuples(index=False, name=None)]
    
    def add_comments(self, comments: list):
        """Appends comment objects to self.comments instance variable."""
        self.comments.values.extend(self.Comments(comments))
        return self
    
@dataclass
class Comment:
    """Mock class for PRAW Comment."""
    id: str
    body: str
    score: int
    created_utc: int # unix time in seconds
    subreddit_name_prefixed: str
    _: str # throwaway variable for submission id
    
    @classmethod
    def from_pandas(cls, dataframe: pd.DataFrame):
        """Return a list of Submission objects from a Pandas DataFrame."""
        # Convert timestamp from dt to int 's'
        df = dataframe.copy()
        df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        df['timestamp'] = df['timestamp'].astype('int64')
        return [cls(*record) for record in df.itertuples(index=False, name=None)]

def generate_submissions(sub_df, com_df):
    """
    Generate the Submissions objects, each with a list of Comment objects
    originating from the particular submission.
    """
    # For each unique sid, generate the Submission object and list
    # of Comment objects. Update each Submission with the list of
    # Comments originating from it.
    submissions = []
    sids = []
    for sid, comment in com_df.groupby('parent_submission_id'):
        sids.append(sid)
        sid_df = sub_df[sub_df.submission_id == sid]
        sid_submissions = Submission.from_pandas(sid_df)
        comments = Comment.from_pandas(comment)
        for submission in sid_submissions:
            submission.add_comments(comments)
        # Update return list; should have same # elements as sub_df # rows
        submissions.extend(sid_submissions)
    # Get the submission id from submission df without comments
    excluded_sid_df = sub_df[~sub_df.submission_id.isin(sids)]
    submissions.extend(Submission.from_pandas(excluded_sid_df))
    return submissions

# 1049 comments (935 unique) and 35 submissions (33 unique)
TEST_DIR = Path(__file__).parent
COMMENT_DF = pd.read_parquet(TEST_DIR / 'test-data/sample-comments.parquet')
SUBMISSION_DF = pd.read_parquet(TEST_DIR / 'test-data/sample-submissions.parquet')
# Remove duplicates to ensure only unique is counted at the end
COMMENT_DF = COMMENT_DF.drop_duplicates(['comment_id'])
SUBMISSION_DF = SUBMISSION_DF.drop_duplicates(['submission_id'])
# Only retain comments with corresponding parent submissions from submission df (876 comments)
COMMENT_DF = COMMENT_DF[
    COMMENT_DF.parent_submission_id
    .isin(SUBMISSION_DF.submission_id)
    ]

# Mock submission objects from test datasets; 35 submission records
SUBMISSION_OBJECTS = generate_submissions(SUBMISSION_DF, COMMENT_DF)

def stub_authenticate_reddit(self, reddit):
    """Bypass Reddit authentication to create a fake RateLimiter object using mock_reddit."""
    self.reddit = reddit
    return self

class StubSubredditSearch:
    """Iterator that yields a list of Submissions when subreddit.search() is called."""
    def __init__(self):
        # Yields a fixed list of Submission objects; set to 4 partitions
        self.return_vals = iter([
            SUBMISSION_OBJECTS[:5],
            SUBMISSION_OBJECTS[5:10],
            SUBMISSION_OBJECTS[10:20],
            SUBMISSION_OBJECTS[20:]
            ])
    
    def __call__(self, *args, **kwargs):
        """Returns a partition of submission objects from the data samples."""
        yield from next(self.return_vals)
    
class StubTextParser:
    """Stubs the txt_to_list func with fixed return values per call."""
    return_vals = iter([['subreddit1', 'subreddit2'], ['query1', 'query2']])
    def __call__(self, *args, **kwargs):
        return next(self.return_vals)

def fake_transform(table):
    """Fake transformation will convert all text data to uppercase."""
    if table.schema.metadata[b'data_source'] == b'submission':
        out_table = (table.
                     set_column(
                         table.schema.get_field_index('title'),
                         'title', 
                         pc.utf8_upper(table.column('title')))
                     .set_column(
                         table.schema.get_field_index('selftext'),
                         'selftext',
                         pc.utf8_upper(table.column('selftext'))
                     ))
    else:
        out_table = (table.
                     set_column(
                         table.schema.get_field_index('body'),
                         'body', 
                         pc.utf8_upper(table.column('body')))
                     )
    return out_table

@patch('usedcaranalytics.pipeline.streamer.logger')
@patch('usedcaranalytics.pipeline.transformer.logger')
@patch('usedcaranalytics.pipeline.loader.logger')
@patch('usedcaranalytics.pipeline.main.DataTransformer')
@patch('usedcaranalytics.pipeline.main.txt_to_list')
@patch('usedcaranalytics.pipeline.streamer.RateLimiter')
@patch('usedcaranalytics.pipeline.main.get_parquet_configs')
@patch('usedcaranalytics.pipeline.main.load_env')
@patch('usedcaranalytics.pipeline.main.Reddit')
def test_main(
    mock_reddit, stub_load_env, stub_parquet_cfg, 
    fake_ratelimiter, stub_txt_to_list, 
    mock_transformer, mock_logging_loader, mock_logging_transformer,
    mock_logging_streamer, tmp_path_factory
    ):
    """Integration test for the ETL pipeline script."""
    # Patch dependencies
    ## Create fake attributes to mock reddit
    mock_reddit.return_value.auth.limits = {'remaining': 1000, 'reset_timestamp': None, 'used':0}
    ## Fix subreddit to a single mock object for all calls
    mock_subreddit = MagicMock()
    stub_subreddit_search = StubSubredditSearch()
    mock_subreddit.search.side_effect = lambda *args, **kwargs: stub_subreddit_search()
    mock_reddit.return_value.subreddit.return_value = mock_subreddit
    ## Patches subreddit.search to return fixed list of submissions per call
    stub_load_env.return_value = (None, None, None, None, None)
    ## Ensure parquet files are written to temp directory; otherwise, default args
    temp_root = tmp_path_factory.mktemp('test_etl')
    stub_parquet_cfg.return_value = get_parquet_configs(root=temp_root)
    ## Patch a fake RateLimiter class
    fake_ratelimiter.return_value.evaluate.return_value = None
    fake_ratelimiter.return_value._authenticate_reddit.side_effect = (
        lambda *args: stub_authenticate_reddit(*args)
        )
    ## Patch txt_to_list function to return subreddits on first call, 
    ## then queries on second call. Total = 4 search pairs.
    stub_text_parser = StubTextParser()
    stub_txt_to_list.return_value = stub_text_parser()
    ## Patch Transformer.transform and instance __call__ to only remove [removed] or [deleted] text
    mock_transformer.return_value.transform.side_effect = lambda table: fake_transform(table)
    mock_transformer.return_value.side_effect = lambda table: fake_transform(table)
    
    # Run main script
    main()
    
    submission_pqt = list(temp_root.glob('data/processed/submission-dataset/*.parquet'))
    comment_pqt = list(temp_root.glob('data/processed/comment-dataset/*.parquet'))
    
    # Assert that new files were created
    assert submission_pqt
    assert comment_pqt
    
    test_sub_df = pd.read_parquet(submission_pqt[0])
    test_cmt_df = pd.read_parquet(comment_pqt[0])
    
    # Since all duplicates removed, should only return 34 subs, and 1010 cmts
    assert len(test_sub_df) == len(SUBMISSION_DF)
    assert len(test_cmt_df) == len(COMMENT_DF)
    
    # Check if transformer worked as intended
    assert test_sub_df.loc[:, 'title'].head().str.isupper().all()
    assert test_sub_df.loc[:, 'selftext'].head().str.isupper().all()
    assert test_cmt_df.loc[:, 'body'].head().str.isupper().all()
    