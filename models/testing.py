import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial import ConvexHull
from utils.class_object import SiteElement
import random, math

# ----------------- Helper functions for Minimum Enclosing Circle -----------------

def dist(p, q):
    """Compute Euclidean distance between points p and q."""
    return math.hypot(p[0] - q[0], p[1] - q[1])

def is_in_circle(p, center, radius):
    """Return True if point p is within circle defined by center and radius (with a tolerance)."""
    return dist(p, center) <= radius + 1e-8

def circle_from_two_points(p, q):
    """Return the circle defined by points p and q."""
    center = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
    radius = dist(p, q) / 2.0
    return center, radius

def circle_from_three_points(p, q, r):
    """Return circle defined by three non-collinear points."""
    d = 2 * (p[0]*(q[1]-r[1]) + q[0]*(r[1]-p[1]) + r[0]*(p[1]-q[1]))
    if d == 0:
        return p, float('inf')
    ux = ((p[0]**2 + p[1]**2)*(q[1]-r[1]) + (q[0]**2 + q[1]**2)*(r[1]-p[1]) + (r[0]**2 + r[1]**2)*(p[1]-q[1])) / d
    uy = ((p[0]**2 + p[1]**2)*(r[0]-q[0]) + (q[0]**2 + q[1]**2)*(p[0]-r[0]) + (r[0]**2 + r[1]**2)*(q[0]-p[0])) / d
    center = (ux, uy)
    radius = dist(center, p)
    return center, radius

def minimum_enclosing_circle(points):
    """
    Computes the minimum enclosing circle for a set of 2D points using a randomized algorithm.
    Returns the center and radius of the circle.
    """
    shuffled = points.copy()
    random.shuffle(shuffled)
    
    center = (0, 0)
    radius = 0
    for i, p in enumerate(shuffled):
        if not is_in_circle(p, center, radius):
            center = p
            radius = 0
            for j in range(i):
                q = shuffled[j]
                if not is_in_circle(q, center, radius):
                    center, radius = circle_from_two_points(p, q)
                    for k in range(j):
                        r_point = shuffled[k]
                        if not is_in_circle(r_point, center, radius):
                            center, radius = circle_from_three_points(p, q, r_point)
    return center, radius

# ----------------- Main Plotting Function -----------------

def plot_elements_from_plsda(pls_loadings, sites_df=None, outline_method="convex",
                             element_properties_file="data/elemental-property-list.xlsx"):
    """
    Projects chemical elements into the PLS‑DA loading space using the loadings computed from PLS‑DA.
    Elements are colored by group. Additionally, if sites_df is provided, outlines will be drawn around 
    the elements for each site. Depending on the input variable `outline_method` the outline will 
    either be a convex hull (outline_method="convex") or a minimum enclosing circle (outline_method="circle"). 
    The output plot file name changes accordingly.
    
    Parameters:
      - pls_loadings: DataFrame with columns "Feature", "Component_1_Loading", "Component_2_Loading".
      - sites_df: Optional DataFrame with site information. Expected columns: "Filename", "Formula", "2c", "6h (2)", "RE".
      - element_properties_file: Path to the element property Excel file.
      - outline_method: Either "convex" or "circle". Determines which outline method to use.
    
    Returns:
      - coordinates: A NumPy array (n_elements x 2) with the projected coordinates.
      - merged_df: A DataFrame containing element properties with added PLS‑DA coordinates.
    """
    # Load element property data.
    df_props = pd.read_excel(element_properties_file)
    df_props.dropna(axis=1, how='all', inplace=True)
    
    # Identify property columns (all except "Symbol").
    prop_columns = [col for col in df_props.columns if col != "Symbol"]
    
    # Standardize column names.
    pls_loadings["Feature_std"] = pls_loadings["Feature"].str.strip().str.lower()
    prop_columns_std = {col: col.strip().lower() for col in prop_columns}
    
    # Identify common features.
    common_features = []
    for orig_col, std_col in prop_columns_std.items():
        if std_col in set(pls_loadings["Feature_std"]):
            common_features.append(orig_col)
    
    if not common_features:
        raise ValueError("No common features found between PLS‑DA loadings and element property file columns!")
    
    print("Common features between property file and PLS‑DA:", common_features)
    
    # Extract data for common features and scale.
    X_elements = df_props[common_features].values
    scaler = StandardScaler()
    X_elements_scaled = scaler.fit_transform(X_elements)
    minmax_scaler = MinMaxScaler()
    X_elements_normalized = minmax_scaler.fit_transform(X_elements_scaled)
    
    # Build the loadings matrix.
    loadings_list = []
    for feature in common_features:
        key = feature.strip().lower()
        loading_row = pls_loadings.loc[pls_loadings["Feature_std"] == key, 
                                       ["Component_1_Loading", "Component_2_Loading"]]
        if loading_row.empty:
            loadings_list.append([0, 0])
        else:
            loadings_list.append(loading_row.iloc[0].values)
    loadings_matrix = np.array(loadings_list)
    
    # Project each element into PLS‑DA space.
    coordinates = X_elements_normalized.dot(loadings_matrix)
    
    # Merge coordinates into the element DataFrame.
    merged_df = df_props.copy()
    merged_df["PLSDA_Component_1"] = coordinates[:, 0]
    merged_df["PLSDA_Component_2"] = coordinates[:, 1]
    
    # Map each element to its group.
    elements_by_group = {
        "alkali_metals": ["Li", "Na", "K", "Rb", "Cs", "Fr"],
        "alkaline_earth_metals": ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],
        "transition_metals": [
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"
        ],
        "lanthanides": ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", 
                         "Ho", "Er", "Tm", "Yb", "Lu"],
        "actinides": ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", 
                      "Es", "Fm", "Md", "No", "Lr"],
        "metalloids": ["B", "Si", "Ge", "As", "Sb", "Te", "Po"],
        "non_metals": ["H", "C", "N", "O", "P", "S", "Se"],
        "halogens": ["F", "Cl", "Br", "I", "At", "Ts"],
        "noble_gases": ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"],
        "post_transition_metals": ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv"]
    }
    group_colors = {
        "alkali_metals": "blue",
        "alkaline_earth_metals": "turquoise",
        "transition_metals": "palegreen",
        "lanthanides": "yellow",
        "actinides": "goldenrod",
        "metalloids": "orange",
        "non_metals": "orangered",
        "halogens": "red",
        "noble_gases": "skyblue",
        "post_transition_metals": "darkgreen",
        "Other": "grey"
    }
    
    element_to_group = {}
    for group, elements in elements_by_group.items():
        for element in elements:
            element_to_group[element] = group
    
    groups = []
    for symbol in merged_df["Symbol"]:
        grp = element_to_group.get(symbol, "Other")
        groups.append(grp)
    merged_df["Group"] = groups
    
    # Create the base scatter plot.
    plt.figure(figsize=(8, 6), dpi=500)
    plt.style.use("ggplot")
    
    for group in merged_df["Group"].unique():
        subset = merged_df[merged_df["Group"] == group]
        color = group_colors.get(group, group_colors["Other"])
        plt.scatter(subset["PLSDA_Component_1"], subset["PLSDA_Component_2"], 
                    color=color, s=250, alpha=0.8, label=group)
        for _, row in subset.iterrows():
            plt.text(row["PLSDA_Component_1"], row["PLSDA_Component_2"], 
                     str(row["Symbol"]), fontsize=12,
                     horizontalalignment='center', verticalalignment='center')
    
    # ----- If sites_df is provided, add outlines around site points -----
    if sites_df is not None:
        # Collect unique site elements using SiteElement.
        unique_2c = set()
        unique_6h2 = set()
        unique_RE = set()
        for _, row in sites_df.iterrows():
            site = SiteElement(row)
            if site.site_2c:
                unique_2c.add(site.site_2c)
            if site.site_6h2:
                unique_6h2.add(site.site_6h2)
            if site.site_RE:
                unique_RE.add(site.site_RE)
        
        # Define colors for site outlines.
        site_outline_colors = {"2c": "#0348a1", "6h (2)": "#ffb01c", "RE": "#c3121e"}
        
        # Convex hull outline function.
        def plot_convex_outline(subset, label, color):
            points = subset[["PLSDA_Component_1", "PLSDA_Component_2"]].values
            n_points = len(points)
            if n_points < 2:
                return
            elif n_points == 2:
                plt.plot(points[:, 0], points[:, 1], linestyle="--", color=color, linewidth=2, label=label)
            else:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points_closed = np.vstack([hull_points, hull_points[0]])
                plt.plot(hull_points_closed[:, 0], hull_points_closed[:, 1],
                         linestyle="--", color=color, linewidth=2, label=label)
                plt.fill(hull_points_closed[:, 0], hull_points_closed[:, 1], color=color, alpha=0.2)
        
        # Updated circle outline function with offsets to avoid overlap.
        def plot_circle_outline(subset, label, color):
            points = subset[["PLSDA_Component_1", "PLSDA_Component_2"]].values
            pts_list = [tuple(point) for point in points]
            if len(pts_list) < 1:
                return
            center, radius = minimum_enclosing_circle(pts_list)
            # Define a small offset for each site type (as a fraction of the radius)
            offset_dict = {
                "2c Outline": (-0.2, 0),    # Shift left
                "6h (2) Outline": (0.2, 0),   # Shift right
                "RE Outline": (0, 0.2)        # Shift up
            }
            offset = offset_dict.get(label, (0, 0))
            # Apply the offset: new center = original center + (offset * radius)
            new_center = (center[0] + offset[0] * radius, center[1] + offset[1] * radius)
            angle = np.linspace(0, 2*np.pi, 100)
            circle_x = new_center[0] + radius * np.cos(angle)
            circle_y = new_center[1] + radius * np.sin(angle)
            plt.plot(circle_x, circle_y, linestyle="--", color=color, linewidth=2, label=label)
            plt.fill(circle_x, circle_y, color=color, alpha=0.2)
        
        # Choose outline function based on outline_method.
        if outline_method == "convex":
            plot_outline = plot_convex_outline
        elif outline_method == "circle":
            plot_outline = plot_circle_outline
        else:
            raise ValueError("outline_method must be either 'convex' or 'circle'.")
        
        subset_2c = merged_df[merged_df["Symbol"].isin(unique_2c)]
        plot_outline(subset_2c, "2c Outline", site_outline_colors["2c"])
        
        subset_6h2 = merged_df[merged_df["Symbol"].isin(unique_6h2)]
        plot_outline(subset_6h2, "6h (2) Outline", site_outline_colors["6h (2)"])
        
        subset_RE = merged_df[merged_df["Symbol"].isin(unique_RE)]
        plot_outline(subset_RE, "RE Outline", site_outline_colors["RE"])
    
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    
    output_filename = f"elements_plot_{outline_method}.png"
    plt.savefig(output_filename, dpi=500)
    plt.show()
    
    # Save coordinates to an Excel file.
    coordinates_df = merged_df.loc[:, ["Symbol", "PLSDA_Component_1", "PLSDA_Component_2"]].copy()
    coordinates_df.columns = ["Symbol", "x", "y"]
    coordinates_df.to_excel("outputs/coordinates.xlsx", index=False)
    print("Coordinates saved to outputs/coordinates.xlsx")
    
    return coordinates, merged_df