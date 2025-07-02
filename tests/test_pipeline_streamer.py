import pytest
from unittest.mock import MagicMock, patch
from usedcaranalytics.pipeline.streamer import DataStreamer

@pytest.fixture
def mock_reddit(mock_subreddit):
    """Creates a mock Reddit instance with minimal subreddit and submission support."""
    mock_reddit = MagicMock()
    # When self.Reddit.subreddit() called by stream_search_results()
    mock_reddit.subreddit.side_effect = lambda *args: mock_subreddit
    return mock_reddit

@pytest.fixture
def mock_subreddit(mock_submission):
    mock_subreddit = MagicMock()
    # When Subreddit.search() called by _fetch_submission()
    mock_subreddit.search.side_effect = lambda *args, **kwargs: mock_submission
    return mock_subreddit

@pytest.fixture
def mock_submission(mock_comment):
    mock_submission = MagicMock()
    # Submission attributes accessed during stream_search_results()
    mock_submission.id = 'sid1'
    mock_submission.title = 'test title'
    mock_submission.selftext = 'test selftext'
    mock_submission.score = 100
    mock_submission.upvote_ratio = 0.50
    mock_submission.created_utc = 1000000
    mock_submission.subreddit_name_prefixed = 'r/cars'
    mock_submission.num_comments = 1
    mock_submission.comments = MagicMock()
    # When submission.comments.replace_more() called by _fetch_comments()
    mock_submission.comments.replace_more.side_effect = lambda *args, **kwargs: None
    # Enable yield from submission.comments in _fetch_comments()
    mock_submission.comments.__iter__.return_value = iter([mock_comment])
    return mock_submission

@pytest.fixture
def mock_comment():
    mock_comment = MagicMock()
    # Comment attributes accessed during _stream_comments()
    mock_comment.id = 'cid1'
    mock_comment.body = 'test comment body'
    mock_comment.score = 10
    mock_comment.created_utc = 1000001
    mock_comment.subreddit_name_prefixed = 'r/cars'
    return mock_comment

@pytest.fixture()
def mock_streamer(mock_reddit):
    """Returns a DataStreamer with a mocked Reddit API."""
    mock_streamer = DataStreamer(mock_reddit)
    return mock_streamer

@pytest.fixture
def mock_stream_search_results():
    """Mocks the stream_search_results() method."""
    fake_records_per_query = [('submission', {'title': 'fake title'}), 
                              ('comment', {'body': 'fake body'})]
    mock_stream_search_results = MagicMock()
    mock_stream_search_results.side_effect = lambda *args, **kwargs: fake_records_per_query
    return mock_stream_search_results

def test_initialization(mock_reddit):
    """Tests DataStreamer constructor correctly assigns Reddit and RateLimiter."""
    reddit = mock_reddit
    streamer = DataStreamer(reddit)
    # Reddit object passed to constructor remains the same
    assert streamer.reddit is reddit
    # Ensure that rate_limiter attr has evaluate() method
    assert getattr(streamer.rate_limiter, 'evaluate', False)

@patch('usedcaranalytics.pipeline.streamer.RateLimiter')
def test_fetch_submissions(mock_ratelimiter, mock_subreddit, mock_reddit):
    """Tests _fetch_submissions method if it returns an iterator of Submission obj."""
    mock_ratelimiter.return_value.evaluate = MagicMock()
    streamer = DataStreamer(mock_reddit)
    subreddit = mock_subreddit
    # Args orthogonal to stubbed output
    streamer._fetch_submissions(subreddit, query="test query", limit=1)
    # Assert that the ratelimiter called evaluate() method before API call
    streamer.rate_limiter.evaluate.assert_called_once()
    # subreddit.search() method should be called with args
    subreddit.search.assert_called_with(query='test query', limit=1)

@patch('usedcaranalytics.pipeline.streamer.RateLimiter')
def test_fetch_comments(mock_ratelimiter, mock_reddit, mock_submission, mock_comment):
    """Tests _fetch_comments yields if it yields comment objects."""
    mock_ratelimiter.return_value.evaluate = MagicMock()
    streamer = DataStreamer(mock_reddit)
    # Make sure to call the generator function
    mocked_comment = mock_comment
    comments = list(streamer._fetch_comments(mock_submission, limit=0))
    # Assert that the ratelimiter called evaluate() method before API call
    streamer.rate_limiter.evaluate.assert_called_once()
    # replace_more() should be called with kwargs
    mock_submission.comments.replace_more.assert_called_with(limit=0)
    # When iterator converted to list, should contain mock_comment
    assert comments == [mocked_comment]

def test_stream_comments(monkeypatch, mock_streamer, mock_submission, mock_comment):
    """Tests _stream_comments if it yields tuple of record type and dict."""
    streamer = mock_streamer
    comment = mock_comment
    # Patch _fetch_comments to always return the comments list
    monkeypatch.setattr(streamer, '_fetch_comments', lambda *args, **kwargs: [comment])
    # Get the list of records, should be 1 comment record
    records = list(streamer._stream_comments(mock_submission, limit=0))
    record_type, data = records[0]
    assert len(records) == 1
    assert record_type == 'comment'
    assert data['comment_id'] == comment.id
    assert data['body'] == comment.body

def test_stream_search_results(
    monkeypatch, mock_streamer, 
    mock_submission, mock_comment
    ):
    """Tests stream_search_results if it yields both comment and submission records."""
    # Mock the DataStreamer
    streamer = mock_streamer
    # Patch _fetch_submissions and _fetch_comments to always return mock objects
    monkeypatch.setattr(streamer, '_fetch_submissions', lambda *args, **kwargs: [mock_submission])
    monkeypatch.setattr(streamer, '_fetch_comments', lambda *args, **kwargs: [mock_comment])
    # Get records from the streamer
    records = list(mock_streamer.stream_search_results(subreddit_name='test_sub', query='test_query', limit=1))
    # Should yield one comment and one submission per submission
    record_types = [record[0] for record in records]
    # Dict of submission and comment data
    submission_data = [data for record_type, data in records if record_type == 'submission'][0]
    comment_data = [data for record_type, data in records if record_type == 'comment'][0]
    # 1 submission and 1 comment
    assert len(records) == 2
    # First element should be record type
    assert 'submission' in record_types
    assert 'comment' in record_types
    # Check if records at least have the same data as origination
    assert submission_data['submission_id'] == mock_submission.id
    assert comment_data['comment_id'] == mock_comment.id

def test_stream(monkeypatch, mock_streamer, mock_stream_search_results):
    """Tests stream yields all data for combinations of subreddits and queries."""
    subreddits = ['subreddit1', 'subreddit2']
    queries = ['query1', 'query2']
    # Mock the DataStreamer
    streamer = mock_streamer
    stream_search_results_fn = mock_stream_search_results
    # Patch stream_search_results with a mock function that returns fixed records
    monkeypatch.setattr(streamer, 'stream_search_results', stream_search_results_fn)
    # Get records from the streamer
    records = list(streamer.stream(subreddits=subreddits, queries=queries))
    # Fake records are yielded as individual records per call to 
    # stream_search_results() generator. Results should be 4*2 = 8
    assert len(records) == 8
    assert all([record_type in ('submission', 'comment') for record_type, _ in records])
    # Should be called 4 times based on product(subreddits, queries)
    assert stream_search_results_fn.call_count == 4

@patch('usedcaranalytics.pipeline.streamer.tqdm')
def test_stream_with_progress_bar(stub_tqdm, monkeypatch, mock_streamer, mock_stream_search_results):
    """Tests stream and stream_search_results handle progress_bar argument."""
    # Mock the DataStreamer
    streamer = mock_streamer
    # Patch tqdm to pass through its iterable unchanged
    stub_tqdm.side_effect = lambda iter, **kwargs: iter
    # Patch stream_search_results to avoid inner tqdm
    monkeypatch.setattr(streamer, 'stream_search_results', mock_stream_search_results)
    subreddits = ['subreddit']
    queries = ['query1', 'query2']
    records = list(mock_streamer.stream(subreddits=subreddits, queries=queries, progress_bar=True))
    # Should return 2*2 = 4 records based on patched fake_records_per_query
    assert len(records) == 4
    # Should call twice because of 1x2 = 2 search pairs
    assert mock_streamer.stream_search_results.call_count == 2