import pytest
import os
import time
from pathlib import Path
from chatwhatcartobuy.utils import getpath
from chatwhatcartobuy.utils.getpath import get_path, get_repo_root

@pytest.fixture
def mock_path():
    return Path('/root/src/module/submodule/main.py')

def stub_config_dotenv_exists(self):
    """
    Stub for Path.exists() to mimic when reaching config dir in search
    of .env file.
    """
    # First path ensures subdir_exists will evaluate to True when in root/src
    # Second path ensures target_exists will evaluate to True when in root/src/config
    valid_paths = ['/root/src/module/config', '/root/src/module/config/.env']
    if str(self) in valid_paths:
        return True
    return False

def stub_root_sentinel_exists(self):
    """
    Stub for Path.exists() to mimic when reaching repo root folder by
    matching sentinel/marker files.
    """
    sentinels = ('.git', 'pyproject.toml', 'setup.py', 'README.md')
    if str(self) in [f'/root/{s}' for s in sentinels]:
        return True
    return False

def test_get_dotenv(mock_path, monkeypatch):
    """
    Test implementation of get_path function when searching for .env
    file in config directory.
    """
    __mockfile__ = mock_path
    monkeypatch.setattr(Path, 'exists', stub_config_dotenv_exists)
    dotenv_path = get_path(__mockfile__, target='.env', subdir='config')
    # Func should go up to /root/src, then move search to config
    assert dotenv_path == Path('/root/src/module/config/.env')

def test_get_path_root_target(mock_path):
    """Tests when root path passed as input. Should return /."""
    assert get_path(start_path=__file__, target='/') == Path('/')
    
def test_get_path_dot_target(mock_path):
    """Tests when dot passed as input. Should return current file path."""
    assert get_path(start_path=__file__, target='.') == Path(__file__)

def test_get_path_invalid_file(mock_path):
    """
    Tests if RuntimeError raised whenever file not found, regardless
    if directory is invalid or file is invalid.
    """
    __mockfile__ = mock_path
    with pytest.raises(RuntimeError):
        get_path(start_path=__mockfile__, target='fakemain.py', subdir='fakedir.py')

def test_get_repo_root(mock_path, monkeypatch):
    """Tests implementation when valid args passed."""
    # Patch the mock_path_exists to Path.exists()
    __mockfile__ = mock_path
    monkeypatch.setattr(Path, 'exists', stub_root_sentinel_exists)
    root = get_repo_root(__mockfile__)
    assert root == Path('/root')
    
def test_no_root_raised():
    """Tests if runtime error raised when no repo root found."""
    with pytest.raises(RuntimeError):
        # Fake path; naturally, no sentinel will be found
        get_repo_root('/fakeroot/fakemodule/fakesubmodule/fake_main.py')
        
@pytest.fixture
def create_test_files(tmp_path):
    # Make three files with a slight time difference
    files = []
    for i, name in enumerate(['file1.txt', 'file2.txt', 'file3.txt']):
        f = tmp_path / name
        f.write_text(f"content {i}")
        files.append(f)
        # Make sure files have different mtimes
        time.sleep(0.001)
    # Set mtimes manually to guarantee ordering
    now = time.time()
    for i, f in enumerate(files):
        f.touch()
        f.write_text(f"content {i}")  # update content for fun
        mtime = now + i
        os.utime(f, (mtime, mtime))
    return files

def test_get_latest_path(create_test_files, monkeypatch, tmp_path):
    files = create_test_files
    # Patch get_repo_root to return tmp_path
    monkeypatch.setattr(getpath, "get_repo_root", lambda: tmp_path)
    # Should return the file with the highest mtime (last one)
    latest = getpath.get_latest_path("file*.txt")
    assert latest == files[-1]

def test_get_earliest_path(create_test_files, monkeypatch, tmp_path):
    files = create_test_files
    monkeypatch.setattr(getpath, "get_repo_root", lambda: tmp_path)
    # Should return the file with the lowest mtime (first one)
    earliest = getpath.get_earliest_path("file*.txt")
    assert earliest == files[0]