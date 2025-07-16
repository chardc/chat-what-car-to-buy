import pytest
import pyarrow as pa
from chatwhatcartobuy.pipeline.transformer import DataTransformer
from unittest.mock import patch

@pytest.fixture
def mock_submission_table():
    """Creates a synthetic PyArrow table simulating submissions."""
    return pa.table({
        "submission_id": pa.array(["id1", "id2", "id3"]),
        "title": pa.array([
            "short 1", 
            "this will pass 20 character minimum length", 
            "this will also pass the 20 character minimum length"
            ]),
        "selftext": pa.array([
            "this self-text is pretty long and should pass the minimum selftext length of one hundred characters.", 
            "this selftext is sufficiently long and should pass the minimum selftext length of one hundred characters", 
            "[deleted]"
            ]),
        "score": pa.array([-1, 0, 1]),
        "upvote_ratio": pa.array([0.8, 0.9, 0.7]),
        "timestamp": pa.array([1000000, 1000001, 1000002], type=pa.timestamp("s")),
        "subreddit": pa.array(["cars", "stickshift", "automatic"]),
        "num_comments": pa.array([0, 15, 3]),
        }, schema=pa.schema(
            [
                ("submission_id", pa.string()),
                ("title", pa.string()),
                ("selftext", pa.string()),
                ("score", pa.int64()),
                ("upvote_ratio", pa.float64()),
                ("timestamp", pa.timestamp("s")),
                ("subreddit", pa.string()),
                ("num_comments", pa.int64())
            ], metadata={b'record_type':b'submission'}
        )
    )

@pytest.fixture
def mock_comment_table():
    """Creates a synthetic PyArrow table simulating comments."""
    return pa.table({
        "comment_id": pa.array(["cid1", "cid2", "cid3"]),
        "body": pa.array([
            "below minimum", 
            "[removed]", 
            "this is above 20 character minimum length"
            ]),
        "score": pa.array([-1, 0, 1]),
        "timestamp": pa.array([1000000, 1000001, 1000002], type=pa.timestamp("s")),
        "subreddit": pa.array(["cars", "stickshift", "automatic"]),
        "parent_submission_id": pa.array(["id1", "id2", "id3"]),
        }, schema=pa.schema(
            [
                ("comment_id", pa.string()),
                ("body", pa.string()),
                ("score", pa.int64()),
                ("timestamp", pa.timestamp("s")),
                ("subreddit", pa.string()),
                ("parent_submission_id", pa.string())
            ], metadata={b'record_type':b'comment'}
        )
    )

TRANSFORMER = DataTransformer()

def test_remove_short_text_submissions(mock_submission_table):
    """Tests filtering short text from submissions."""
    TRANSFORMER.current_record_type = 'submission'
    results = TRANSFORMER.remove_short_text_from_table(mock_submission_table)
    # Row 1 (short title) and Row 3 (short selftext) removed
    assert results.num_rows == 1
    # Only row 2 should remain
    assert results.column('submission_id').to_pylist() == ['id2']

def test_remove_short_text_comments(mock_comment_table):
    """Tests filtering short text from comments."""
    TRANSFORMER.current_record_type = 'comment'
    results = TRANSFORMER.remove_short_text_from_table(mock_comment_table)
    # Row 1 and row 2 removed
    assert results.num_rows == 1
    # Only row 3 should remain
    assert results.column('comment_id').to_pylist() == ['cid3']

def test_remove_custom_submission_length(mock_submission_table):
    """Tests filtering of comments if custom minimum title and selftext length was passed."""
    # Should remove all since excessive minimum character length
    TRANSFORMER.current_record_type = 'submission'
    results_long_ttl = TRANSFORMER.remove_short_text_from_table(mock_submission_table, 
                                                     min_title_length=50)
    assert results_long_ttl.num_rows == 0
    # Should retain rows 2 and 3 since they pass selftext length, 
    # but row 1 fails title length check
    results_short_stxt = TRANSFORMER.remove_short_text_from_table(mock_submission_table, 
                                                       min_selftext_length=5)
    assert results_short_stxt.num_rows == 2

def test_remove_custom_comment_length(mock_comment_table):
    """Tests filtering of comments if custom minimum body length was passed."""
    # Should remove all since excessive minimum character length
    TRANSFORMER.current_record_type = 'comment'
    results_long_body = TRANSFORMER.remove_short_text_from_table(mock_comment_table, 
                                                      min_body_length=50)
    assert results_long_body.num_rows == 0
    # Should retain all since no minimum length
    results_short_body = TRANSFORMER.remove_short_text_from_table(mock_comment_table, 
                                                       min_body_length=0)
    assert results_short_body.num_rows == 3

def test_remove_matching_submission(mock_submission_table):
    """Tests regex-based filtering on submission texts, primarily [deleted] and [removed]."""
    TRANSFORMER.current_record_type = 'submission'
    results = TRANSFORMER.remove_match_from_table(mock_submission_table)
    # Only row 3 is removed since selftext == '[deleted]'
    assert results.num_rows == 2
    assert results.column('submission_id').to_pylist() == ['id1', 'id2']

def test_remove_matching_comment(mock_comment_table):
    """Tests regex-based filtering on comment texts, primarily [deleted] and [removed]."""
    TRANSFORMER.current_record_type = 'comment'
    results = TRANSFORMER.remove_match_from_table(mock_comment_table)
    # Only row 2 is removed since body == '[removed]'
    assert results.num_rows == 2
    assert results.column('comment_id').to_pylist() == ['cid1', 'cid3']

def test_edge_case_empty_table():
    """Tests behavior with an empty PyArrow table."""
    empty_schema = pa.schema([], metadata={b'record_type': b'submission'})
    empty_table = pa.Table.from_batches([], schema=empty_schema)
    transformer = DataTransformer()
    result = transformer.transform(empty_table)
    assert result.num_rows == 0
    assert result.schema == empty_table.schema
    
def test_edge_case_no_metadata():
    """Tests behavior when metadata is missing."""
    # Table with no metadata
    table = pa.table({
            "title": ["Valid title"], 
            "selftext": ["Valid selftext."]
            }
        )  
    transformer = DataTransformer()
    result = transformer.transform(table)
    # Transformer should handle and ignore exception, then return untampered table
    assert result.num_rows == 1
    assert result.schema.metadata is None
    
def test_exception_handling_invalid_columns(mock_submission_table):
    """Tests graceful handling of missing columns."""
    # Drop required fields intentionally
    table = mock_submission_table.drop(["title", "selftext"])  
    transformer = DataTransformer()
    result = transformer.transform(table)
    # Original table returned due to exception handling
    assert result.equals(table)
    
# # Commented out these test cases as they don't scale to additional transformations in transformer
# def test_transform_full_pipeline_submissions(mock_submission_table):
#     """Tests the full transformation pipeline on submissions."""
#     transformer = DataTransformer()
#     result = transformer.transform(mock_submission_table)
#     # Only row 2 passes both length and regex checks
#     assert result.num_rows == 1
#     assert result.column('submission_id').to_pylist() == ['id2']

# def test_transform_full_pipeline_comments(mock_comment_table):
#     """Tests the full transformation pipeline on comments."""
#     transformer = DataTransformer()
#     result = transformer.transform(mock_comment_table)
#     # Only row 3 passes both checks
#     assert result.num_rows == 1
#     assert result.column('comment_id').to_pylist() == ['cid3']
