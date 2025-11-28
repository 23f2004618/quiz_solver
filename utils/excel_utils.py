# excel_utils.py

import pandas as pd
from io import BytesIO

def load_excel(buffer: bytes, sheet_name=None) -> pd.DataFrame:
    """
    Load Excel data into a pandas DataFrame.
    If sheet_name is None → loads the first sheet.
    """
    try:
        excel_file = BytesIO(buffer)
        if sheet_name:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
        else:
            df = pd.read_excel(excel_file, engine='openpyxl')
        return df
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")
