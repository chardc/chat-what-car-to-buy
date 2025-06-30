from praw import Reddit
from praw.models import Subreddit, Submission
from tqdm.auto import tqdm
from typing import List
from itertools import product
from usedcaranalytics.utils.ratelimiter import RateLimiter, backoff_on_rate_limit

class DataStreamer:
    def __init__(self, reddit: Reddit):
        """
        Args:
            - reddit: Praw Reddit object.
        Returns:
            - DataStreamer object.
        """
        # If invalid reddit passed to constructor, RateLimiter will catch and raise Exception
        self.reddit = reddit
        self.rate_limiter = RateLimiter(reddit)
    
    @backoff_on_rate_limit(max_retries=5)
    def _fetch_submissions(self, subreddit: Subreddit, query: str, **kwargs):
        """
        Implements TooManyRequests exception handling by retrying comment retrieval 
        request after a certain backoff period. Exponentially increases backoff by 2**N
        after every failed request until max retries is reached.
        Args:
            - subreddit: Praw Subreddit object.
        Returns:
            - Generator of Submission objects.
        """
        # Evaluate if current request can be accommodated with remaining limits
        self.rate_limiter.evaluate()
        return subreddit.search(**kwargs, query=query)

    @backoff_on_rate_limit(max_retries=5)
    def _fetch_comments(self, submission: Submission, **kwargs):
        """
        Implements TooManyRequests exception handling by retrying comment retrieval 
        request after a certain backoff period. Exponentially increases backoff by 2**N
        after every failed request until max retries is reached.
        Args:
            - submission: Praw Submission object.
        Returns:
            - Generator of Comment objects.
        """
        # Evaluate if current request can be accommodated with remaining limits
        self.rate_limiter.evaluate()
        # Replace 'more comments' with specified limit (default = 0 or retain top-level comments only)
        submission.comments.replace_more(**kwargs)
        yield from submission.comments
    
    def _stream_comments(self, submission: Submission, **kwargs):
        """
        Fetches comments from a Submission objects then parses each comment into 
        a dictionary record. Each entry is streamed for efficient memory footprint 
        when handling larger CommentForests.
        Args:
            - submission: Praw Submission object.
        Returns: 
            - Generator for comment data in dictionary format.
        """
        comments = self._fetch_comments(submission, **kwargs)
        # Update comments dict with info dict 
        for comment in comments:            
            # Stream comment data when slot available in current window
            yield "comment", {
                'comment_id':comment.id,
                'body':comment.body,
                'score':comment.score,
                'timestamp':int(comment.created_utc),
                'subreddit':comment.subreddit_name_prefixed,
                'parent_submission_id':submission.id
                }
    
    def stream_search_results(
        self, subreddit_name: str, query:str, limit: int=50, 
        progress_bar: bool=None, **search_kwargs
        ):
        """
        Fetches submissions then fetches the comments for each submission. Data is then 
        repackaged into a dictionary and streamed as a tuple of (record, data)
        Args:
            - subreddit_name: Name of subreddit to search from.
            - query: Search keywords or query.
            - limit: Number of maximum submissions to return for the search.
            - progress_bar: If True, display progress bars in console.
            - search_kwargs: Kwargs for praw.subreddit.search() method.
        Returns:
            - Generator for submission data and comment data in dictionary format.
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        
        # Iterator
        submissions = self._fetch_submissions(
            **search_kwargs, subreddit=subreddit, query=query, limit=limit
            )
        
        if progress_bar:
            submissions = tqdm(
                submissions, total=limit, position=1, colour='red', 
                desc=f"fetching submissions...", unit='posts', leave=False
            )
        
        # Fetch submissions, and for every submission, fetch the comments
        for submission in submissions:
            # Stream comment data from current submission ("submission", Dict[str, Any])
            yield from self._stream_comments(submission, limit=0)
            # Stream submission data when slot available in current window
            yield "submission", {
                'submission_id':submission.id,
                'title':submission.title,
                'selftext':submission.selftext,
                'score':submission.score,
                'upvote_ratio':submission.upvote_ratio,
                'timestamp':int(submission.created_utc),
                'subreddit':submission.subreddit_name_prefixed,
                'num_comments':submission.num_comments
                }
            
    def stream(
        self, subreddits:List[str], queries:List[str], progress_bar:bool=None, 
        **search_kwargs
        ):
        """
        Wrapper for streaming functions. Takes a list of subreddits and queries, then calls the 
        stream_search_results method for each combination of subreddit and query. 
        Args:
            - subreddits: List of subreddit names.
            - queries: List of search queries or keywords:
            - progress_bar: If true, display progress bars in the console.
        Returns:
            - Generator of submission data and comment data.
        """
        # Generator of combinations
        search_pairs = product(subreddits, queries)
        
        # List
        if progress_bar:
            search_pairs = list(search_pairs)
            search_pairs = tqdm(
                search_pairs, total=len(search_pairs), 
                desc=f'streaming subreddit search results...',
                unit='queries', colour='orange'
                )
        
        # Parse submission and comment data with jittered API calls
        for subreddit, query in search_pairs:    
            # Stream submission and comment records (str(record_type), Dict[str(col_name), Any])
            yield from self.stream_search_results(
                **search_kwargs, subreddit_name=subreddit, query=query, progress_bar=progress_bar
                )