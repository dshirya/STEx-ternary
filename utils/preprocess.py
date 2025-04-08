import pandas as pd
import re

def parse_elements(cell_value):
    """
    Parses a string that contains one or more element symbols along with their numeric counts.
    Uses the regex pattern: r'([A-Z][a-z]*)(\d*\.?\d*)'
    
    Examples:
      - "Tm" returns [('Tm', 1.0)]
      - "Tm0.886Cd0.114" returns [('Tm', 0.886), ('Cd', 0.114)]
    
    If the numeric part is missing, defaults to a count of 1.0.
    
    Parameters:
        cell_value (str): The cell content from one of the element columns.
        
    Returns:
        list of tuples: A list containing (element, count) pairs.
    """
    s = str(cell_value).strip()
    if not s:
        return []
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
    matches = re.findall(pattern, s)
    results = []
    for element, count_str in matches:
        count = float(count_str) if count_str != "" else 1.0
        if element:  # Avoid empty matches
            results.append((element, count))
    return results

def combine_elements_from_row(row):
    """
    Combines element counts from the '2a', '6h (1)', and '12k' columns of a given row.
    For each column the function uses `parse_elements` to extract element-count pairs. If an element
    appears multiple times across these columns, their counts are summed.
    
    The resulting string is built from alphabetically sorted elements, 
    with the count omitted if it is exactly 1.
    
    Parameters:
        row (pd.Series): A row from the DataFrame.
        
    Returns:
        str: A concatenated string of elements and their summed counts.
    """
    columns_to_combine = ['2a', '6h (1)', '12k']
    combined_counts = {}
    
    for col in columns_to_combine:
        cell_value = row[col]
        parsed_items = parse_elements(cell_value)
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

def process_csv(filepath):
    """
    Reads and processes the CSV file with the following steps:
      - Reads a comma-delimited CSV file while skipping the second row.
      - Drops the columns 'Notes', 'Num Elements', 'combined_RE10_l', and 'combined_RE10'.
      - Creates a new column 'combined_elements' by combining and summing the values in
        the '2a', '6h (1)', and '12k' columns.
      - Removes the now redundant '2a', '6h (1)', and '12k' columns.
    
    Parameters:
        filepath (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The final processed DataFrame.
    """
    # Read the CSV (comma-separated) and skip the second row (index=1)
    df = pd.read_csv(filepath, sep=',', skiprows=[1])
    # Drop unwanted columns
    df = df.drop(columns=['Notes', 'Num Elements', 'combined_RE10_l', 'combined_RE10'])
    # Create a new column with combined elements from selected columns
    df["RE"] = df.apply(combine_elements_from_row, axis=1)
    # Remove the original columns used for combining
    df = df.drop(columns=['2a', '6h (1)', '12k'])
    return df

