import pandas as pd
from io import BytesIO

def load_csv(buffer: bytes):
    """
    Load CSV data and return as list of dictionaries.
    Each row becomes a dict with column names as keys.
    """
    df = pd.read_csv(BytesIO(buffer))
    return df.to_dict('records')
