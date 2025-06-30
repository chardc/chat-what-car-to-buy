from prawcore import OAuthException
from usedcaranalytics.utils.ratelimiter import RateLimiter, backoff_on_rate_limit
from unittest.mock import MagicMock, patch
from collections import deque
import pytest

class LimitsDict(dict):
    """Sentinel class to avoid infinite recursion for test_evaluate_sleep()."""
    REMAINING_VALS = [5, 1000]
    
    def get_remaining(self):
        return self.REMAINING_VALS.pop(0) if self.REMAINING_VALS else 1000
        
    def __getitem__(self, key):
        if key == 'remaining':
            return self.get_remaining()
        return super().__getitem__(key)

# Mock reddit object
@pytest.fixture
def mock_reddit():
    reddit = MagicMock()
    reddit.auth.limits = {'remaining': None, 'reset_timestamp': None, 'used': None}
    return reddit

def invalid_oauth(self):
    """Stub for Reddit.user.me() method. Always raises OAuthException."""
    raise OAuthException

def test_initialization(mock_reddit):
    """Tests constructor to check whether instance attributes are initialized properly."""
    rate_limiter = RateLimiter(mock_reddit, buffer_range=[0,1])
    assert rate_limiter.buffer_range == [0,1]
    assert rate_limiter.total_requests == 0
    assert isinstance(rate_limiter.requests_in_window, deque)

def test_invalid_initialization(mock_reddit):
    """Tests if error was raised when reddit object with invalid keys passed to constructor."""
    with pytest.raises(Exception):
        reddit = mock_reddit
        reddit.user.me = invalid_oauth
        reddit.read_only = False
        rate_limiter = RateLimiter(reddit)
    
def test_evaluate_deque_implem(mock_reddit, monkeypatch):
    """
    Tests sliding window counter implementation as fallback when reddit.auth.limits
    is initially set to None (During first evaluate call when no API call made yet).
    """
    rate_limiter = RateLimiter(mock_reddit)
    
    # Test if total requests and requests in window updated 
    # and sliding window counter works when auth.limits is None
    rate_limiter.evaluate()
    assert rate_limiter.total_requests == 1
    assert len(rate_limiter.requests_in_window) == 1

def test_evaluate_sleep(mock_reddit, monkeypatch):
    """
    Tests if time.sleep() is called with calculated delay whenever the expected
    requests dip into random buffer (for remaining API requests). 
    """
    reddit = mock_reddit
    reddit.auth.limits['remaining'] = 5
    rate_limiter = RateLimiter(reddit)
    
    # Patch ensures buffer is 5, and remaining - 1 < buffer
    monkeypatch.setattr("random.randint", lambda *args: 5)
    
    # delay = earliest request + default window period - current time + jitter
    # Patches delay as 1 + 600 - 1 + 0.01 = 600.01; 
    rate_limiter.requests_in_window = [1]
    monkeypatch.setattr("time.time", lambda: 1)
    monkeypatch.setattr("random.randrange", lambda *args: 0.01)
    
    # Test if sleep called when dipping into buffer, mocking when 
    # auth.limits has no reset timestamp
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)
        
    reddit.auth.limits = LimitsDict(reddit.auth.limits)
    
    # First evaluate() call dips into buffer and sleeps via time.sleep()
    # Succeeding calls avoid recursion
    rate_limiter.evaluate()
    mock_sleep.assert_called_with(600.01)

def test_log_request(mock_reddit, monkeypatch):
    """Tests if a timestamp is logged to deque whenever method is called."""
    rate_limiter = RateLimiter(mock_reddit)
    monkeypatch.setattr("time.time", lambda: 1)
    # Check that time.time()=1 was appended to deque
    rate_limiter._log_request()
    assert rate_limiter.requests_in_window[0] == 1
    assert len(rate_limiter.requests_in_window) == 1

def test_refresh_window(mock_reddit, monkeypatch):
    """Tests if timestamps outside sliding window are removed."""
    rate_limiter = RateLimiter(mock_reddit)
    # Removes mock timestamps < 2-1, retaining only [2] in deque
    monkeypatch.setattr("time.time", lambda: 2)
    rate_limiter.PERIOD = 1
    rate_limiter.requests_in_window = deque([0, 1, 2])
    rate_limiter._refresh_window()
    assert rate_limiter.requests_in_window == deque([1, 2])