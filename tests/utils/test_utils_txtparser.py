import pytest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from chatwhatcartobuy.utils.txtparser import txt_to_list
from pathlib import Path

def generate_txt_tempfile(lines: list):
    with TemporaryDirectory() as tempdir:
        abs_path = Path(tempdir) / 'subreddits.txt'
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        # Yield temporary file path's absolute path
        yield abs_path

@pytest.fixture
def mock_txt_file():
    """Mocks a valid *.txt file with comments, empty newlines, and valid lines."""
    lines = [
        '# This is a sample comment\n',
        'cars\n',
        '\n' # This is an empty line in multiline txt file
        'vehicles\n',
        '# Another sample comment\n'
    ]
    yield from generate_txt_tempfile(lines)

@pytest.fixture
def mock_empty_txt_file():
    """Mocks an empty *.txt file."""
    lines = []
    yield from generate_txt_tempfile(lines)

@patch('chatwhatcartobuy.utils.txtparser.get_path')
def test_default_txt_to_list(mock_get_path, mock_txt_file):
    """Tests basic functionality of text to list func."""
    mock_get_path.return_value = mock_txt_file
    # Input othogonal to output; output is stubbed
    subreddits = txt_to_list('mock_subreddits.txt') 
    assert subreddits == ['cars', 'vehicles']

@patch('chatwhatcartobuy.utils.txtparser.get_path')
def test_empty_txt_to_list(mock_get_path, mock_empty_txt_file):
    """Tests handling of empty file objects. Should return empty list"""
    mock_get_path.return_value = mock_empty_txt_file
    subreddits = txt_to_list('mock_empty.txt')
    assert subreddits == []