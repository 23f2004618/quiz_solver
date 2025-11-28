import json
import numpy as np
import pandas as pd

def parse_answer(raw: str):
    s = raw.strip()
    print(f"[PARSER INPUT]: {repr(s)}")
    
    # 0. Check for LaTeX boxed answer (common in reasoning models)
    # Pattern: \boxed{answer}
    import re
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", s)
    if boxed_match:
        extracted = boxed_match.group(1)
        print(f"[PARSER]: Found boxed answer: {extracted}")
        return parse_answer(extracted) # Recursively parse the content inside the box

    # JSON object (try first to handle explicit JSON)
    try:
        if s.startswith("{") and s.endswith("}"):
            parsed = json.loads(s)
            
            # CRITICAL: Detect if LLM returned example documentation instead of answer
            # If the JSON contains the submission structure (email/secret/url/answer),
            # it's likely returning the example format - extract just the 'answer' field
            if isinstance(parsed, dict) and all(k in parsed for k in ['email', 'secret', 'url', 'answer']):
                print(f"[PARSER WARNING]: LLM returned example payload structure!")
                print(f"[PARSER]: Extracting 'answer' field from example")
                # Extract the actual answer from the example structure
                actual_answer = parsed.get('answer', '')
                # Recursively parse the extracted answer
                if isinstance(actual_answer, str):
                    return parse_answer(actual_answer)
                else:
                    return convert_to_json_serializable(actual_answer)
            
            print(f"[PARSER OUTPUT]: {repr(parsed)} (JSON object)")
            return convert_to_json_serializable(parsed)
    except:
        pass

    # Boolean (check before string processing)
    if s.lower() in ["true", "false"]:
        result = s.lower() == "true"
        print(f"[PARSER OUTPUT]: {repr(result)} (boolean)")
        return result
    
    # Number (check before string processing)
    try:
        if "." in s:
            result = float(s)
            print(f"[PARSER OUTPUT]: {repr(result)} (float)")
            return result
        result = int(s)
        print(f"[PARSER OUTPUT]: {repr(result)} (int)")
        return result
    except:
        pass

    # String: Remove surrounding quotes if present
    # Handle both single and double quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    
    print(f"[PARSER OUTPUT]: {repr(s)} (string)")
    return s


def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle pandas types
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    
    # Handle dict (recursively convert values)
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    
    # Handle list (recursively convert items)
    if isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    
    # Already JSON-serializable (str, int, float, bool, None)
    return obj
