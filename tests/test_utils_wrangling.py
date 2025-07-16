import pytest
import pandas as pd
from chatwhatcartobuy.utils import wrangling

def test_deduplicate_pandas():
    df = pd.DataFrame(
        {
            'id': ['duplicate', 'duplicate', 'unique']
            }
        )
    out_df = wrangling.deduplicate_pandas(df, 'id')
    assert len(out_df) == 2
    assert out_df.id.to_list() == ['duplicate','unique']
    
def test_remove_empty_rows_pandas():
    df = pd.DataFrame(
        {
            'id': [1,2,3],
            'title': ['this is an acceptable title', 
                      'a', # Should be removed
                      'another acceptable title'
                      ],
            'selftext': ['sufficient selftext', 
                         'an acceptable selftext', 
                         '[!//d' # Should be removed
                         ] 
            }
        )
    out_df = wrangling.remove_empty_rows_pandas(df, ['title','selftext'])
    print(out_df)
    assert len(out_df) == 1
    assert out_df.id.to_list() == [1] # only first row is retained