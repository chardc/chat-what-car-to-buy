import pytest

@pytest.fixture
def patch_env(monkeypatch):
    monkeypatch.setenv('PRAW_ID', 'test_client_key')
    monkeypatch.setenv('PRAW_SECRET', 'test_secret_key')
    monkeypatch.setenv('PRAW_USER_AGENT', 'test_agent')
    monkeypatch.setenv('PRAW_USERNAME', 'test_usr')
    monkeypatch.setenv('PRAW_PASSWORD', 'test_pw')

def test_load_env(patch_env, monkeypatch):
    """Tests if environment variables are properly loaded."""
    # Patch env 
    patch_env
    # Ensure all env variables are returned with corresponding values
    from chatwhatcartobuy.config.api_config import PRAW_ID, PRAW_SECRET, PRAW_USER_AGENT, PRAW_USERNAME, PRAW_PASSWORD
    
    assert PRAW_ID == 'test_client_key'
    assert PRAW_SECRET == 'test_secret_key'
    assert PRAW_USER_AGENT == 'test_agent'
    assert PRAW_USERNAME == 'test_usr'
    assert PRAW_PASSWORD == 'test_pw'