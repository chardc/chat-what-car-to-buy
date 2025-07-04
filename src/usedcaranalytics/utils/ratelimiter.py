import datetime as dt, time, random, functools
from collections import deque
from typing import Tuple, List, Union
from prawcore.exceptions import ResponseException, OAuthException
from praw import Reddit

# RateLimiter class implementation for counting requests in 10-minute sliding window and
# throttling requests to maintain < 100 requests/min average.
class RateLimiter:
    """
    Rate limits the API requests made when streaming data from Reddit API. Ensures requests stay under
    1000 requests per 10-minute sliding window as per Reddit policies. Fallback logic if PRAW's auth.limits
    is unavailable is manual sliding window counter.
    """
    MAX_REQUESTS = 1000
    PERIOD = 600.0
    
    def __init__(self, reddit: Reddit, buffer_range: Union[Tuple[int, int], List[int]]=(50, 100)):
        """
        Args:
            - reddit: PRAW Reddit object.
            - buffer_range: Tuple or list containing the minimum and maximum value of API request buffers.
        Returns:
            - RateLimiter object.
        """
        self._authenticate_reddit(reddit)
        self.buffer_range = buffer_range
        self.total_requests = 0
        self.requests_in_window = deque()
    
    def _authenticate_reddit(self, reddit: Reddit):
        """Checks if reddit object passed to constructor contains valid credentials."""
        self.reddit = reddit
        if reddit.read_only:    
            print('Access is in read-only mode. Limiting requests to 10 calls/min.')
            # Create instance attribute denoting new MAX_REQUESTS: 10 requests/min
            self.MAX_REQUESTS = 100
            return self
        # Otherwise, check if OAuth passes
        try:
            reddit.user.me()
        except OAuthException as e:
            print(e)
            raise Exception('Cannot construct RateLimiter with invalid Reddit object due to invalid API keys.')
        return self
    
    def evaluate(self):
        """Checks if current request can be accommodated based on current limits."""        
        remaining_requests = self.reddit.auth.limits['remaining']
        
        # If remaining requests from praw.Reddit.auth.limits unavailable (no requests yet),
        # return manually counted results from sliding window counter
        if remaining_requests is None:
            self._refresh_window()
            remaining_requests = self.MAX_REQUESTS - len(self.requests_in_window)
        
        buffer = random.randint(*self.buffer_range)
        
        # If we dip into the buffer, sleep until limits reset
        if remaining_requests - 1 < buffer:
            # Calculate time left until limits refresh and add jitter
            reset_time = self.reddit.auth.limits['reset_timestamp']
            
            # If Praw returns no information, then manually compute request reset time
            if reset_time is None:
                # Manually get the reset time, which is at least 600 seconds from earliest request in window
                reset_time = self.requests_in_window[0] + self.PERIOD
            
            delay = max(reset_time - time.time(), 0)
            delay += random.randrange(0.01, 5.0)
            time.sleep(delay)
            
            # Re-evaluate if API call can proceed
            return self.evaluate()
        
        # Update sliding window counter
        self._refresh_window()._log_request()
        
        # Tally API call
        self.total_requests += 1
    
    def _log_request(self):
        """
        Failsafe for when REDDIT.auth.limits is unavailable. Tracks the requests_in_window attribute, which
        is a deque containing timestamps of API requests made in the current sliding window (600 seconds).
        
        Usage: 
            - Before every API call, refresh the sliding window and append current timestamp to mark 
            outbound request via "self._refresh_window()._log_request()".
        """
        self.requests_in_window.append(time.time())
        return self
    
    def _refresh_window(self):
        """
        Updates self.requests_in_window by evaluating the number of requests in current window. Removes
        request logs outside current 10-minute sliding window.
        """
        # Remove requests outside 600 second sliding window (i.e. earlier than 600s ago)
        while self.requests_in_window and self.requests_in_window[0] < time.time() - self.PERIOD:
            self.requests_in_window.popleft()
        return self
    
    def print_total_requests(self):
        return f'{self.total_requests} total requests as of {dt.datetime.now():%Y-%m-%d %H:%M:%S}.'
    
    def print_remaining_requests(self):
        """
        Prints the number of remaining requests in current window either based on PRAW reddit.auth.limits
        or manual sliding window counter.
        """
        limits = self.reddit.auth.limits
        # If limits hasn't refreshed yet, return remaining requests. Otherwise, return total limit
        if all(limits['reset_timestamp'], limits['remaining'], limits['used']):
            return f'{limits['remaining']}' \
                if time.time() < limits['reset_timestamp'] \
                else f'{limits['remaining'] + limits['used']}'
        # Refresh current window and return difference between max requests and # requests in window
        else:
            self._refresh_window()
            return f'{self.MAX_REQUESTS - len(self.requests_in_window)}'
    
    def __str__(self):
        return f'RateLimiter object: {self.print_remaining_requests()} available requests in current window. \
            {self.print_total_requests()}'

# Backoff algorithm with full jitter for handling transient failures from HTTP 429 response
def backoff_on_rate_limit(
    max_retries: int=5, 
    base_delay: float=1.0, 
    cap_delay: float=60.0
    ):
    """
    Decorator factory that applies exponential backoff (with optional jitter) when Reddit API
    rate limits (HTTP 429) or server errors occur. Stops after max_retries and re-raises the exception.
    
    Args:
        - max_retries: Integer value for max retries. When attempts exceed this number, an Exception is raised.
        - base_delay: Float for base delay in seconds (i.e. Delay at first failed attempt).
        - cap_delay: Float for maximum delay in seconds.
        - jitter: Bool on whether to implement full jitter or not.
    Returns:
        - Decorator to be applied to an PRAW API request wrapper.
    """
    def decorator(func):# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start with base delay, then exponentially scale by attempt
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except ResponseException as e:
                    if attempt > max_retries:
                        raise Exception("Max retries exceeded with Reddit API.")
                    delay = random.uniform(
                        0, 
                        min(
                            cap_delay, 
                            base_delay * 2 ** attempt
                            )
                        )
                    print(f"[WARNING] {e.__class__.__name__} on attempt {attempt + 1},\
                        retrying after {delay:.2f}s.")
                    time.sleep(delay)
                    attempt += 1
        return wrapper
    return decorator