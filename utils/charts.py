import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
from io import BytesIO
import pandas as pd
import numpy as np

def generate_chart(data, chart_type="line", title="", xlabel="", ylabel="", figsize=(10, 6)):
    """
    Generate various types of charts and return as base64 data URI.
    
    Args:
        data: pandas DataFrame or dict
        chart_type: 'line', 'bar', 'scatter', 'pie', 'histogram', 'box'
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
    
    Returns:
        base64 data URI string
    """
    plt.figure(figsize=figsize)
    
    if isinstance(data, dict):
        # Convert dict to Series for proper bar charts
        data = pd.Series(data)
    
    try:
        if chart_type == "line":
            data.plot(kind='line', ax=plt.gca())
        elif chart_type == "bar":
            data.plot(kind='bar', ax=plt.gca())
        elif chart_type == "scatter":
            if len(data.columns) >= 2:
                plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
            else:
                data.plot(kind='scatter', ax=plt.gca())
        elif chart_type == "pie":
            if len(data.columns) >= 1:
                data.iloc[:, 0].plot(kind='pie', ax=plt.gca(), autopct='%1.1f%%')
        elif chart_type == "histogram":
            data.plot(kind='hist', ax=plt.gca(), bins=20)
        elif chart_type == "box":
            data.plot(kind='box', ax=plt.gca())
        else:
            # Default to line
            data.plot(ax=plt.gca())
        
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        plt.close()
        
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    
    except Exception as e:
        plt.close()
        raise Exception(f"Chart generation error: {e}")


def generate_simple_plot(x_data, y_data=None, title="", xlabel="", ylabel=""):
    """
    Generate a simple line or scatter plot.
    
    Args:
        x_data: list or array of x values
        y_data: list or array of y values (if None, uses index)
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
    
    Returns:
        base64 data URI string
    """
    plt.figure(figsize=(10, 6))
    
    if y_data is None:
        plt.plot(x_data)
    else:
        plt.plot(x_data, y_data)
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

