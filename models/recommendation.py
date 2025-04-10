import re
import pandas as pd
import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull

def recommendation_engine(fixed_elements, sites_df, coordinate_file, output_file="recommendations.xlsx"):
    """
    Recommends candidate elements for the missing site based on distances from observed 
    missing-site elements in sites_df.
    
    Candidate pools:
      - Rec Type 1: All elements from the coordinate file that are NOT present in the 
                    overall missing-site set.
      - Rec Type 2: The unique missing-site elements (overall_set) that are NOT in 
                    the observed set (rows where fixed sites match the fixed values).
      - Rec Type 3: All elements from the coordinate file whose coordinates lie 
                    inside the convex hull of overall missing-site values, excluding 
                    those already observed.
    
    For each candidate, the Euclidean distance (based on scaled coordinates) from its 
    coordinate to each observed missing-site element is computed and the minimum distance 
    is taken. Let d_min be the smallest nonzero distance among all candidates in that pool.
    Each candidate is then assigned a score of:
         score = 1    if candidate_distance == 0 
         score = d_min / candidate_distance   otherwise.
         
    The function outputs an Excel file (padded to 10 recommendations per type) with columns:
         Element_rec1, Value_rec1, Element_rec2, Value_rec2, Element_rec3, Value_rec3.
    
    Parameters:
      fixed_elements : dict
          Dictionary with exactly two keys (from {"2c", "6h", "RE"}) with their fixed element symbols.
          For example: {"2c": "Ru", "6h": "Cd"}.
      sites_df : pandas.DataFrame
          DataFrame with site information. Must contain columns "2c", "6h", and "RE".
      coordinate_file : str
          Path to the Excel file with coordinates. Expected columns: "Symbol", "x", "y".
      output_file : str, optional
          The Excel filename to save the recommendations (default "recommendations.xlsx").
    
    Returns:
      final_df : pandas.DataFrame with columns:
         Element_rec1, Value_rec1, Element_rec2, Value_rec2, Element_rec3, Value_rec3.
    """
    # Global constants
    COORD_MULTIPLIER = 5
    SITE_LABELS = {"2c", "6h", "RE"}
    
    # --- Helper: extract primary element from a cell (ignore numbers) ---
    def get_primary(cell_value):
        s = str(cell_value).strip()
        if not s:
            return ""
        pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
        matches = re.findall(pattern, s)
        if not matches:
            return s
        primary, _ = max(matches, key=lambda x: float(x[1]) if x[1] != "" else 1.0)
        return primary
    
    # --- Helper: compute scores given a candidate pool and a reference set ---
    def compute_scores(candidate_pool, element_coords, reference_set):
        rec_list = []
        for candidate in candidate_pool:
            if candidate not in element_coords:
                continue
            cand_coord = np.array(element_coords[candidate])
            dists = []
            for ref in reference_set:
                if ref in element_coords:
                    ref_coord = np.array(element_coords[ref])
                    dists.append(np.linalg.norm(cand_coord - ref_coord))
            if not dists:
                continue
            candidate_distance = min(dists)
            rec_list.append((candidate, candidate_distance, None))
        if not rec_list:
            return []
        nonzero = [d for (_, d, _) in rec_list if d > 0]
        d_min = min(nonzero) if nonzero else 0.0
        scored = []
        for candidate, dist, _ in rec_list:
            score = 1.0 if dist == 0 else d_min / dist
            scored.append((candidate, dist, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored
    
    # --- Helper: pad the list to n items ---
    def pad_recommendations(rec_list, n=10):
        padded = rec_list[:n]
        if len(padded) < n:
            padded.extend([("", np.nan, np.nan)] * (n - len(padded)))
        return padded
    
    # --- Load and scale coordinates ---
    coord_df = pd.read_excel(coordinate_file)
    coord_df["x"] = coord_df["x"] * COORD_MULTIPLIER
    coord_df["y"] = coord_df["y"] * COORD_MULTIPLIER
    element_coords = {}
    for idx, row in coord_df.iterrows():
        element_coords[row["Symbol"]] = (row["x"], row["y"])
    
    # --- Determine the missing site ---
    fixed_sites = set(fixed_elements.keys())
    if len(fixed_sites) != 2:
        raise ValueError("Please provide exactly two fixed sites.")
    missing_site = list(SITE_LABELS - fixed_sites)[0]
    
    # --- Process missing-site values from sites_df ---
    # overall_set: all unique missing-site values (extracted with get_primary)
    # observed_set: missing-site values from rows where the fixed sites match fixed_elements
    overall_set = set()
    observed_set = set()
    for idx, row in sites_df.iterrows():
        match_fixed = True
        for site_key, fixed_el in fixed_elements.items():
            if str(row[site_key]).strip().lower() != fixed_el.lower():
                match_fixed = False
                break
        ms_val = get_primary(row[missing_site])
        if ms_val:
            overall_set.add(ms_val)
            if match_fixed:
                observed_set.add(ms_val)
    
    # --- Define candidate pools ---
    # Rec Type 1: All elements from the coordinate file that are NOT already present in overall_set.
    pool_rec1 = set(element_coords.keys()) - observed_set
    
    # Rec Type 2: The overall missing-site elements that are NOT observed (observed_set).
    pool_rec2 = overall_set - observed_set
    
    # Rec Type 3: Candidate pool = all elements from the coordinate file whose coordinates lie inside 
    # the convex hull of overall_set. We then remove any element already in observed_set.
    if len(overall_set) >= 3:
        pts = np.array([element_coords[el] for el in overall_set if el in element_coords])
        unique_pts = np.unique(pts, axis=0)
        if len(unique_pts) >= 3:
            hull = ConvexHull(unique_pts)
            hull_pts = unique_pts[hull.vertices]
            hull_path = Path(hull_pts)
            pool_convex = {el for el in element_coords.keys() if hull_path.contains_point(element_coords[el])}
            # For rec_3, we want to score candidates (from the convex-hull space) that are new,
            # i.e. not already observed.
            pool_rec3 = pool_convex - observed_set
        else:
            pool_rec3 = set()
    else:
        pool_rec3 = set()
    
    # --- Compute recommendations for each candidate pool ---
    rec1 = compute_scores(pool_rec1, element_coords, observed_set)
    rec2 = compute_scores(pool_rec2, element_coords, observed_set)
    rec3 = compute_scores(pool_rec3, element_coords, observed_set)
    
    rec1 = pad_recommendations(rec1, 10)
    rec2 = pad_recommendations(rec2, 10)
    rec3 = pad_recommendations(rec3, 10)
    
    # --- Build final DataFrame and output Excel file ---
    data = {
        "Element_rec1": [item[0] for item in rec1],
        "Value_rec1":   [round(item[2], 3) if not np.isnan(item[2]) else np.nan for item in rec1],
        "Element_rec2": [item[0] for item in rec2],
        "Value_rec2":   [round(item[2], 3) if not np.isnan(item[2]) else np.nan for item in rec2],
        "Element_rec3": [item[0] for item in rec3],
        "Value_rec3":   [round(item[2], 3) if not np.isnan(item[2]) else np.nan for item in rec3],
    }
    final_df = pd.DataFrame(data)
    final_df.to_excel(output_file, index=False)
    print(f"Recommendations saved to '{output_file}'.")
    return final_df

