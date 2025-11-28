"""
Data processing utilities for quiz solver.
Supports filtering, aggregation, transformation, statistical analysis.
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union


def filter_data(df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter DataFrame based on conditions.
    
    Args:
        df: pandas DataFrame
        conditions: dict like {"column": value} or {"column": {"operator": "gt", "value": 5}}
    
    Returns:
        Filtered DataFrame
    """
    result = df.copy()
    for col, condition in conditions.items():
        if isinstance(condition, dict):
            op = condition.get("operator", "eq")
            val = condition.get("value")
            if op == "eq":
                result = result[result[col] == val]
            elif op == "ne":
                result = result[result[col] != val]
            elif op == "gt":
                result = result[result[col] > val]
            elif op == "lt":
                result = result[result[col] < val]
            elif op == "gte":
                result = result[result[col] >= val]
            elif op == "lte":
                result = result[result[col] <= val]
        else:
            result = result[result[col] == condition]
    return result


def aggregate_data(df: pd.DataFrame, group_by: List[str], agg_func: Dict[str, str]) -> pd.DataFrame:
    """
    Aggregate DataFrame by grouping.
    
    Args:
        df: pandas DataFrame
        group_by: list of columns to group by
        agg_func: dict like {"column": "sum", "column2": "mean"}
    
    Returns:
        Aggregated DataFrame
    """
    return df.groupby(group_by).agg(agg_func).reset_index()


def sort_data(df: pd.DataFrame, by: Union[str, List[str]], ascending: bool = True) -> pd.DataFrame:
    """
    Sort DataFrame.
    
    Args:
        df: pandas DataFrame
        by: column name or list of column names
        ascending: sort order
    
    Returns:
        Sorted DataFrame
    """
    return df.sort_values(by=by, ascending=ascending)


def calculate_statistics(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Calculate basic statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: column name
    
    Returns:
        dict with statistics
    """
    data = df[column]
    return {
        "count": len(data),
        "sum": data.sum(),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "min": data.min(),
        "max": data.max(),
        "q25": data.quantile(0.25),
        "q75": data.quantile(0.75),
    }


def reshape_data(df: pd.DataFrame, operation: str, **kwargs) -> pd.DataFrame:
    """
    Reshape DataFrame (pivot, melt, etc.).
    
    Args:
        df: pandas DataFrame
        operation: 'pivot', 'melt', 'transpose'
        kwargs: operation-specific arguments
    
    Returns:
        Reshaped DataFrame
    """
    if operation == "pivot":
        return df.pivot(**kwargs)
    elif operation == "melt":
        return df.melt(**kwargs)
    elif operation == "transpose":
        return df.transpose()
    return df


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: input text
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Basic cleaning
    text = text.strip()
    
    return text


def parse_numeric(value: Any) -> Union[int, float, None]:
    """
    Parse value to numeric.
    
    Args:
        value: input value
    
    Returns:
        numeric value or None
    """
    try:
        # Try int first
        if isinstance(value, (int, float)):
            return value
        
        # Handle string
        value_str = str(value).strip().replace(",", "")
        
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except:
        return None


def find_correlations(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Find correlations between numeric columns.
    
    Args:
        df: pandas DataFrame
        threshold: correlation threshold
    
    Returns:
        DataFrame with correlations above threshold
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Get pairs with correlation above threshold
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                correlations.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": corr_val
                })
    
    return pd.DataFrame(correlations)
