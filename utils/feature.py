import pandas as pd
import re

def parse_elements(cell_value):
    """
    Parses a string that contains one or more element symbols with their numeric counts.
    Uses the regex pattern: r'([A-Z][a-z]*)(\d*\.?\d*)'
    
    Examples:
      - "Tm" returns [('Tm', 1.0)]
      - "Tm0.886Cd0.114" returns [('Tm', 0.886), ('Cd', 0.114)]
    
    If the numeric part is missing, a count of 1.0 is assumed.
    
    Parameters:
        cell_value (str): The string from one of the element columns.
        
    Returns:
        list of tuples: Each tuple is (element, count).
    """
    s = str(cell_value).strip()
    if not s:
        return []
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
    matches = re.findall(pattern, s)
    results = []
    for element, count_str in matches:
        count = float(count_str) if count_str != "" else 1.0
        if element:  # avoid empty matches
            results.append((element, count))
    return results

def compute_site_features(site_formula, properties_df):
    """
    Computes the weighted (atomic percent) average of each property feature for a given site.
    
    For a site formula (e.g., "Cd0.114Tm2.886"):
      - Parse the string into element-count pairs.
      - Calculate total atoms.
      - For each element, its atomic fraction is count/total.
      - For each feature (column in properties_df except "Symbol"), compute:
            weighted_feature += (atomic fraction) * (property value for that element)
    
    Parameters:
        site_formula (str): A string representing the site composition.
        properties_df (pd.DataFrame): A DataFrame containing element features 
          with one row per element and a column "Symbol" for element labels.
          
    Returns:
        dict: A dictionary where keys are feature names and values are the weighted feature values.
    """
    parsed = parse_elements(site_formula)
    total_atoms = sum(count for _, count in parsed)
    features_weighted = {}
    # Get the list of feature columns (exclude 'Symbol')
    feature_cols = list(properties_df.columns)
    feature_cols.remove("Symbol")
    
    # Initialize feature totals to zero
    for feat in feature_cols:
        features_weighted[feat] = 0.0
    
    if total_atoms == 0:
        # Avoid division by zero; return zeros.
        return features_weighted
    
    # Calculate weighted average for each feature
    for element, count in parsed:
        atomic_fraction = count / total_atoms
        # Lookup the element in the properties file
        elem_row = properties_df[properties_df["Symbol"] == element]
        if not elem_row.empty:
            for feat in feature_cols:
                features_weighted[feat] += atomic_fraction * elem_row.iloc[0][feat]
        else:
            print(f"Warning: Element {element} not found in properties data.")
    
    return features_weighted

def process_element_features(df_final, excel_file):
    """
    Processes the properties Excel file and computes the site features for each entry in df_final.
    
    Steps:
      - Load the Excel file (which contains element features) and drop any columns with NaN values.
      - For each row in df_final, process each of the site composition columns: "RE", "2c", and "6h (2)".
      - For each nonempty site composition, compute weighted features using `compute_site_features`.
      - The Site_Label is set to the name of the column that provided the composition.
      - Return a new DataFrame with columns: 'Filename', 'Formula', 'Site', 'Site_Label', and the computed feature columns.
    
    Parameters:
        df_final (pd.DataFrame): The DataFrame containing sites. It must include:
            - "Filename" and "Formula"
            - Site composition columns (e.g., "RE", "2c", and "6h (2)").
        excel_file (str): The path to the Excel file with element features.
        
    Returns:
        pd.DataFrame: A DataFrame with the site feature calculations.
    """
    # Load the element features from Excel and drop columns with NaN values.
    properties_df = pd.read_excel(excel_file)
    properties_df = properties_df.dropna(axis=1)
    
    # Get the feature columns (all columns except 'Symbol')
    feature_cols = list(properties_df.columns)
    feature_cols.remove("Symbol")
    
    output_rows = []
    # Define the site composition columns to process.
    site_columns = ["RE", "2c", "6h"]
    
    # Process each row from df_final.
    for _, row in df_final.iterrows():
        # Loop through each site composition column.
        for site_col in site_columns:
            # Check that the site column exists and has a nonempty value.
            if site_col in row and pd.notna(row[site_col]) and str(row[site_col]).strip() != "":
                site_formula = row[site_col]
                feat_dict = compute_site_features(site_formula, properties_df)
                out_row = {
                    "Filename": row["Filename"],
                    "Formula": row["Formula"],
                    "Site": site_formula,
                    "Site_Label": site_col
                }
                out_row.update(feat_dict)
                output_rows.append(out_row)
    
    # Create and order the output DataFrame.
    out_df = pd.DataFrame(output_rows)
    ordered_cols = ["Filename", "Formula", "Site", "Site_Label"] + feature_cols
    out_df = out_df[ordered_cols]
    return out_df

