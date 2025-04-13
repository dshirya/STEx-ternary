import pandas as pd
import re

def parse_elements(cell_value):
    
    s = str(cell_value).strip()
    if not s:
        return []
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
    matches = re.findall(pattern, s)
    results = []
    for element, count_str in matches:
        count = float(count_str) if count_str != "" else 1.0
        if element:
            results.append((element, count))
    return results

def combine_elements(row, columns):
    """
    Combines element counts from specified columns in a row.
    
    Parameters:
        row (pd.Series): A row from the DataFrame.
        columns (list): List of column names to combine.
    
    Returns:
        str: A concatenated string of elements and their summed counts.
    """
    combined_counts = {}
    
    for col in columns:
        if col not in row or pd.isna(row[col]):
            continue
        parsed_items = parse_elements(row[col])
        for element, count in parsed_items:
            combined_counts[element] = combined_counts.get(element, 0) + count
    
    combined_str = ""
    for element in sorted(combined_counts.keys()):
        total_count = combined_counts[element]
        if total_count == 1:
            combined_str += f"{element}"
        else:
            combined_str += f"{element}{total_count}"
    return combined_str

def process_csv(df, rename_map=None):
    """
    Processes the CSV file and combines user-specified columns.
    
    Parameters:
        filepath (str): Path to the CSV file.
        rename_map (dict): Mapping from new column names to list of old columns to combine.
                           Example: {"R": ["2a", "6h (1)", "12k"], "X": ["6h (2)"], "M": ["2c"]}
    
    Returns:
        pd.DataFrame: Processed DataFrame with renamed/combined columns.
    """
    if rename_map is None:
        rename_map = {"R": ["2a", "6h (1)", "12k"], "X": ["6h (2)"], "M": ["2c"]}
    
    df = df.drop(index=1)
    
    # Drop unnecessary columns if they exist
    cols_to_drop = ['Num Elements', 'combined_RE10_l', 'combined_RE10']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Combine specified columns
    for new_col, old_cols in rename_map.items():
        df[new_col] = df.apply(lambda row: combine_elements(row, old_cols), axis=1)
        df = df.drop(columns=[col for col in old_cols if col in df.columns])
    
    return df