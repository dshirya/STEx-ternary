import re
import pandas as pd
import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from utils.class_object import SiteElement
from utils.preprocess import parse_elements

# Define the standard ordering for the three sites.
STANDARD_ORDER = ["R", "M", "X"]
COORD_MULTIPLIER = 5

def recommendation_engine(fixed_elements, sites_df, coordinate_file, output_file="outputs/recommendations.xlsx"):
    """
    Recommends candidate elements for the missing site based on distances from observed 
    missing-site elements in sites_df.

    Candidate pools:
      - Rec Type 1: All elements from the coordinate file that are NOT present in the 
                    overall missing-site set.
      - Rec Type 2: The unique missing-site elements (overall_set) that are NOT in 
                    the observed set (rows where fixed sites match fixed_elements).
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
          Dictionary with exactly two keys (chosen from STANDARD_ORDER) with their fixed element symbols.
          For example: {"R": "Tb", "M": "Ru"}.
      sites_df : pandas.DataFrame
          DataFrame with site information. Must contain columns for all three sites in STANDARD_ORDER.
      coordinate_file : str
          Path to the Excel file with coordinates. Expected columns: "Symbol", "x", "y".
      output_file : str, optional
          The Excel filename to save the recommendations (default "outputs/recommendations.xlsx").

    Returns:
      final_df : pandas.DataFrame with columns:
         Element_rec1, Value_rec1, Element_rec2, Value_rec2, Element_rec3, Value_rec3.
    """
    # Multiply coordinates by a constant factor.
    
    all_sites = set(STANDARD_ORDER)


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
    element_coords = {row["Symbol"]: (row["x"], row["y"]) for idx, row in coord_df.iterrows()}

    # --- Determine the missing site ---
    fixed_keys = set(fixed_elements.keys())
    if len(fixed_keys) != 2:
        raise ValueError("Please provide exactly two fixed sites.")
    missing_site = (all_sites - fixed_keys).pop()

    # --- Process missing-site values using SiteElement ---
    overall_set  = set()
    observed_set = set()

    for _, row in sites_df.iterrows():
        se = SiteElement(row)
        ms_val = getattr(se, f"site_{missing_site}")
        if not ms_val:
            continue
        overall_set.add(ms_val)

        # check fixed‐site match
        if all(
            getattr(se, f"site_{k}") 
            and getattr(se, f"site_{k}").lower() == v.lower()
            for k, v in fixed_elements.items()
        ):
            observed_set.add(ms_val)

    # fallbacks
    if not overall_set:
        overall_set = set(element_coords.keys())
    if not observed_set:
        observed_set = overall_set

    # --- Define candidate pools ---
    pool_rec1 = set(element_coords.keys()) - observed_set
    pool_rec2 = overall_set - observed_set

    if len(overall_set) >= 3:
        pts = np.array([element_coords[el] for el in overall_set if el in element_coords])
        unique_pts = np.unique(pts, axis=0)
        if len(unique_pts) >= 3:
            hull = ConvexHull(unique_pts)
            hull_pts = unique_pts[hull.vertices]
            hull_path = Path(hull_pts)
            pool_convex = {el for el in element_coords.keys() if hull_path.contains_point(element_coords[el])}
            pool_rec3 = pool_convex - observed_set
        else:
            pool_rec3 = set()
    else:
        pool_rec3 = set()

    # --- Compute recommendations for each candidate pool ---
    rec1 = pad_recommendations(compute_scores(pool_rec1, element_coords, observed_set), 10)
    rec2 = pad_recommendations(compute_scores(pool_rec2, element_coords, observed_set), 10)
    rec3 = pad_recommendations(compute_scores(pool_rec3, element_coords, observed_set), 10)

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

def best_recommendation(final_df):
    """
    Given the recommendations DataFrame with columns:
      Element_rec1, Value_rec1, Element_rec2, Value_rec2, Element_rec3, Value_rec3,
    returns a list of tuples (candidate, avg_score) for all candidate elements that appear
    in all three recommendation lists.
    """
    rec1_elements = set(final_df["Element_rec1"][final_df["Element_rec1"] != ""])
    rec2_elements = set(final_df["Element_rec2"][final_df["Element_rec2"] != ""])
    rec3_elements = set(final_df["Element_rec3"][final_df["Element_rec3"] != ""])
    
    common_candidates = rec1_elements & rec2_elements & rec3_elements
    if not common_candidates:
        print("No common element found in all recommendations")
        return []
    
    candidate_scores = []
    for candidate in common_candidates:
        score1 = final_df.loc[final_df["Element_rec1"] == candidate, "Value_rec1"].iloc[0]
        score2 = final_df.loc[final_df["Element_rec2"] == candidate, "Value_rec2"].iloc[0]
        score3 = final_df.loc[final_df["Element_rec3"] == candidate, "Value_rec3"].iloc[0]
        avg_score = (score1 + score2 + score3) / 3.0
        candidate_scores.append((candidate, avg_score))
    
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Common candidates with average scores:")
    for cand, score in candidate_scores:
        print(f"  {cand}: {score}")
        
    return candidate_scores

def calculate_stoichiometry_from_df(df, fixed_elements, candidate, site_order=None):
    """
    Determines the average stoichiometric indices for the compound based on a template of similar compounds.

    The function:
      1. Determines the missing site (the one not present in fixed_elements) from STANDARD_ORDER.
      2. Filters df for rows where the fixed site columns have primary elements matching fixed_elements.
      3. For each similar row, it parses the Formula and collects the count for each site's primary element.
      4. It averages the counts (if any rows are found) and rounds to the nearest natural number.
      5. If no similar compounds are found, default stoichiometric values of 1 are used.

    Returns:
      A dictionary mapping each site (from site_order) to a natural number.
    """
    # Use STANDARD_ORDER if no ordering is provided.
    if site_order is None:
        site_order = STANDARD_ORDER

    def row_matches(row):
        se = SiteElement(row)
        # Only check fixed sites
        for site, fixed_el in fixed_elements.items():
            row_el = se.get_primary_element(row[site])
            if row_el is None or row_el.lower() != fixed_el.lower():
                return False
        return True

    similar_df = df[df.apply(row_matches, axis=1)]
    
    site_counts = {site: [] for site in site_order}

    for idx, row in similar_df.iterrows():
        parsed = parse_elements(row["Formula"])  # list of (element, count)
        formula_dict = {el: count for el, count in parsed}
        se = SiteElement(row)
        for site in site_order:
            site_el = se.get_primary_element(row[site])
            if site_el is None:
                continue
            if site_el in formula_dict:
                site_counts[site].append(formula_dict[site_el])
    
    stoich = {}
    for site in site_order:
        if site_counts[site]:
            avg_value = np.mean(site_counts[site])
            stoich[site] = int(round(avg_value))
            if stoich[site] < 1:
                stoich[site] = 1

    for site in site_order:
        if site not in stoich:
            stoich[site] = 1

    return stoich

def add_candidate(df, fixed_elements, candidates, filename="candidate.cif"):
    """
    Adds new rows to the DataFrame for each candidate element from best_recommendation.

    For each candidate:
      - The candidate element is inserted into the missing site (determined from STANDARD_ORDER).
      - The two fixed sites are set using fixed_elements.
      - Stoichiometry is computed based on similar compounds (using calculate_stoichiometry_from_df).
      - The compound formula is constructed using the ordering in STANDARD_ORDER.
      - The Notes column is set to "candidate" and the Filename column to the provided filename.

    Returns:
      The updated DataFrame with the new candidate rows appended.
    """
    site_order = STANDARD_ORDER

    new_rows = []
    for cand_item in candidates:
        candidate = cand_item[0] if isinstance(cand_item, tuple) else cand_item
        new_row = {"Filename": filename, "Notes": "candidate"}
        
        # Assign fixed or candidate element for each site
        for site in site_order:
            new_row[site] = fixed_elements.get(site, candidate)

        # Calculate stoichiometry.
        stoich = calculate_stoichiometry_from_df(df, fixed_elements, candidate, site_order=site_order)
        for site in site_order:
            stoich.setdefault(site, 1)

        # Construct the formula using the standard ordering.
        parts = [
            f"{new_row[site]}" if stoich[site] == 1 else f"{new_row[site]}{int(stoich[site])}"
            for site in site_order
        ]
        new_row["Formula"] = "".join(parts)

        new_rows.append(new_row)

    # Append new rows to the DataFrame
    new_candidates_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_candidates_df], ignore_index=True)

    # Print each formula on its own line
    formulas = [r["Formula"] for r in new_rows]
    print("New candidate formulas:\n" + "\n".join(formulas))

    return df