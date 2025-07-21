import pytest
import pandas as pd
import datetime as dt
from chatwhatcartobuy.utils import wrangling
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.fixture
def stub_dataframe():
    return pd.DataFrame(
        {
            'submission_id': [100, 200, 300],
            'title': ['this is an acceptable title', 
                      'a', # Should be removed
                      'another acceptable title'
                      ],
            'selftext': ['sufficient selftext', 
                         'an acceptable selftext', 
                         '[!//d' # Should be removed
                         ] ,
            'score': [-2, # should be removed
                      1, 
                      0]
            }
        )
    
@pytest.fixture
def stub_dataframe_whitespaces():
    return pd.DataFrame(
        {
            'text1': ['many     spaces     here',
                      'normal text here'
                      ],
            'text2': ['normal text here',
                      'a    lot   of spaces   here'
                      ]
        }
    )

def test_deduplicate_pandas():
    df = pd.DataFrame(
        {
            'id': ['duplicate', 'duplicate', 'unique']
            }
        )
    out_df = wrangling.deduplicate_pandas(df, 'id')
    assert len(out_df) == 2
    assert out_df.id.to_list() == ['duplicate','unique']
    
def test_remove_empty_rows_pandas(stub_dataframe):
    out_df = wrangling.remove_empty_rows_pandas(stub_dataframe, ['title','selftext'])
    print(out_df)
    assert len(out_df) == 1
    assert out_df.submission_id.to_list() == [100] # only first row is retained
    
def test_remove_low_score_pandas(stub_dataframe):
    out_df = wrangling.remove_low_score_pandas(stub_dataframe, 0)
    assert len(out_df) == 2
    assert out_df.submission_id.to_list() == [200, 300] # Remove row with -2 score
    
def test_replace_text_pandas(stub_dataframe_whitespaces):
    out_df = wrangling.replace_text_pandas(stub_dataframe_whitespaces, ['text1', 'text2'], r'\s+', ' ')
    assert out_df.text1.to_list() == ['many spaces here', 'normal text here']
    assert out_df.text2.to_list() == ['normal text here', 'a lot of spaces here']
    
def test_replace_url_pandas():
    df = pd.DataFrame(
        {
            'text1': ['sample url here: https://www.example.com',
                      'no url here: '
                      ],
            'text2': ['no url here: ',
                      'basic url here: (https://test.html)'
                      ]
        }
    )
    out_df = wrangling.replace_url_pandas(df, ['text1', 'text2'])
    assert out_df.text1.to_list() == ['sample url here: <URL>', 'no url here: ']
    assert out_df.text2.to_list() == ['no url here: ', 'basic url here: <URL>']
    
def test_remove_extra_whitespace_pandas(stub_dataframe_whitespaces):
    out_df = wrangling.remove_extra_whitespace_pandas(stub_dataframe_whitespaces, ['text1', 'text2'])
    assert out_df.text1.to_list() == ['many spaces here', 'normal text here']
    assert out_df.text2.to_list() == ['normal text here', 'a lot of spaces here']
    
def test_lowercase_text_pandas():
    df = pd.DataFrame(
        {'text' : ['UPPERCASE TEXT HERE', 'CamelCaseTextHere']}
    )
    out_df = wrangling.lowercase_text_pandas(df, ['text'])
    assert out_df.text.to_list() == ['uppercase text here', 'camelcasetexthere']
    
def test_assign_record_type_pandas(stub_dataframe):
    out_df = wrangling.assign_record_type_pandas(stub_dataframe, 'submission')
    assert 'record_type' in out_df.columns
    assert (out_df.record_type == 'submission').all()

def test_drop_and_rename_text_cols(stub_dataframe):
    submission_df = stub_dataframe
    out_submission_df = wrangling.drop_and_rename_text_cols(submission_df)
    assert 'title' not in out_submission_df.columns
    assert 'text' not in out_submission_df.columns
    assert 'document' in out_submission_df.columns
    assert (out_submission_df.document == (submission_df.title + ' ' + submission_df.selftext)).all()
    
    body_df = pd.DataFrame(
        {
            'comment_id': [0, 1, 2],
            'body' : ['first comment', 'second comment', 'third comment'],
            'parent_submission_id': [100, 200, 300]
        }
    )
    out_body_df = wrangling.drop_and_rename_text_cols(body_df)
    assert 'body' not in out_body_df
    assert 'parent_submission_id' not in out_body_df
    assert 'document' in out_body_df
    assert 'submission_id' in out_body_df
    assert (out_body_df.document == body_df.body).all()
    assert (out_body_df.submission_id == body_df.parent_submission_id).all()

@patch('chatwhatcartobuy.utils.wrangling.assign_record_type_pandas')
@patch('chatwhatcartobuy.utils.wrangling.drop_and_rename_text_cols')
@patch('chatwhatcartobuy.utils.wrangling.remove_extra_whitespace_pandas')
@patch('chatwhatcartobuy.utils.wrangling.replace_url_pandas')
@patch('chatwhatcartobuy.utils.wrangling.remove_low_score_pandas')
@patch('chatwhatcartobuy.utils.wrangling.remove_empty_rows_pandas')
@patch('chatwhatcartobuy.utils.wrangling.deduplicate_pandas')
@patch('chatwhatcartobuy.utils.wrangling.dataset_to_pandas')
def test_wrangle_dataset(
    mock_read_dataset,
    mock_deduplicate,
    mock_remove_empty,
    mock_remove_score,
    mock_replace_url,
    mock_remove_space,
    mock_drop_rename,
    mock_assign_type
    ):
    
    mock_dataset = MagicMock()
    mock_dataset.schema.metadata = {b'record_type' : b'submission'}
    stub_df = pd.DataFrame({
        'submission_id': [],
        'title': [],
        'selftext': [],
        'score': []
    })
    
    # Mimic sequential transformations
    mock_read_dataset.return_value = stub_df
    mock_deduplicate.return_value = mock_read_dataset()
    mock_remove_empty.return_value = mock_deduplicate()
    mock_remove_score.return_value = mock_remove_empty()
    mock_replace_url.return_value = mock_remove_score()
    mock_remove_space.return_value = mock_replace_url()
    mock_drop_rename.return_value = mock_remove_space()
    mock_assign_type.return_value = mock_drop_rename()
    
    # Only verify that the modular transformation fns are called by orchestrator
    wrangling.wrangle_dataset(mock_dataset)
    mock_read_dataset.assert_called_with(mock_dataset)
    mock_deduplicate.assert_called_with(mock_read_dataset(), 'submission_id')
    mock_remove_empty.assert_called_with(mock_deduplicate(), ['title', 'selftext'])
    mock_remove_score.assert_called_with(mock_remove_empty(), threshold=-2)
    mock_replace_url.assert_called_with(mock_remove_score(), ['title', 'selftext'])
    mock_remove_space.assert_called_with(mock_replace_url(), ['title', 'selftext'])
    mock_drop_rename.assert_called_with(mock_remove_space())
    mock_assign_type.assert_called_with(mock_drop_rename(), 'submission')

@patch('chatwhatcartobuy.utils.wrangling.dt')
@patch('chatwhatcartobuy.utils.wrangling.get_repo_root')
def test_pandas_to_parquet(mock_root, mock_dt, stub_dataframe, monkeypatch, tmp_path):
    mock_dt.datetime.now.return_value = dt.datetime(2000,1,1) # Y-m-d == 2000-01-01
    mock_write = MagicMock()
    monkeypatch.setattr(type(stub_dataframe), 'to_parquet', mock_write)
    mock_root.return_value = tmp_path
    
    # Test if .parquet is appended and if current date partitioning works
    wrangling.pandas_to_parquet(stub_dataframe, 'sample', partition_by_date=True)
    export_path = mock_root() / 'data/processed/2000-01-01/sample.parquet'
    mock_write.assert_called_with(export_path, engine='pyarrow')
    
    # Test if .parquet is not duplicated when filename ext provided
    wrangling.pandas_to_parquet(stub_dataframe, 'sample.parquet', partition_by_date=False)
    mock_write.assert_called_with(mock_root() / 'data/processed/sample.parquet', engine='pyarrow')