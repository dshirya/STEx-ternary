import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial import ConvexHull
from utils.class_object import SiteElement

# Define your element groups and colors:
elements_by_group = {
    "alkali_metals": ["Li", "Na", "K", "Rb", "Cs", "Fr"],
    "alkaline_earth_metals": ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],
    "transition_metals": [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"
    ],
    "lanthanides": [
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", 
        "Ho", "Er", "Tm", "Yb", "Lu"
    ],
    "actinides": [
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", 
        "Es", "Fm", "Md", "No", "Lr"
    ],
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

def plot_elements_from_plsda_loadings(pls_loadings, 
                                      sites_df=None,
                                        element_properties_file="data/elemental-property-list.xlsx",
                                        ):
    """
    Projects chemical elements into the PLS‑DA loading space using the loadings computed from PLS‑DA.
    The element properties are scaled (using StandardScaler) and normalized (MinMaxScaler) in the same way as 
    the PLS‑DA features, and then each element is projected into the 2D latent space via the dot product 
    with the loadings.
    
    In addition, elements are colored according to their group using the provided dictionaries.
    If a DataFrame containing site information is provided (with columns "Filename", "Formula", "2c", "6h (2)", "RE"),
    the chemical element symbols for each site are extracted (using the SiteElement class) and an outline is drawn:
      - For sites with three or more points, a convex hull is computed, outlined with a dashed line, and filled with color (alpha=0.2).
      - For sites with exactly two points, a dashed line is drawn connecting them.
    
    Parameters:
      - pls_loadings: DataFrame with columns "Feature", "Component_1_Loading", "Component_2_Loading".
                      The "Feature" names should correspond to property names (columns) in the element property file.
      - element_properties_file: Path to the Excel file containing element properties.
          It must include a "Symbol" column for element labels; all other columns are considered as properties.
      - sites_df: Optional DataFrame with site information. Expected columns: "Filename", "Formula", "2c", "6h (2)", "RE".
    
    Returns:
      - coordinates: A NumPy array (n_elements x 2) with the projected coordinates.
      - merged_df: A DataFrame containing the original element properties along with the computed PLS‑DA coordinates.
    """
    # Load the element property data.
    df_props = pd.read_excel(element_properties_file)
    df_props.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaN values.
    
    # Identify property columns (all columns except the label "Symbol")
    prop_columns = [col for col in df_props.columns if col != "Symbol"]
    
    # Standardize column names in both datasets to make matching robust.
    pls_loadings["Feature_std"] = pls_loadings["Feature"].str.strip().str.lower()
    prop_columns_std = {col: col.strip().lower() for col in prop_columns}
    
    # Identify common features by comparing standardized names.
    common_features = []
    for orig_col, std_col in prop_columns_std.items():
        if std_col in set(pls_loadings["Feature_std"]):
            common_features.append(orig_col)
    
    if not common_features:
        raise ValueError("No common features found between PLS‑DA loadings and element property file columns!")
    
    print("Common features between property file and PLS‑DA:", common_features)
    
    # Extract data for the common features.
    X_elements = df_props[common_features].values  # shape: (n_elements, n_common_features)
    
    # Scale the element properties using StandardScaler and then apply MinMaxScaler.
    scaler = StandardScaler()
    X_elements_scaled = scaler.fit_transform(X_elements)
    minmax_scaler = MinMaxScaler()
    X_elements_normalized = minmax_scaler.fit_transform(X_elements_scaled)
    
    # Build the loadings matrix for the common features (matching order).
    loadings_list = []
    for feature in common_features:
        key = feature.strip().lower()
        loading_row = pls_loadings.loc[pls_loadings["Feature_std"] == key, 
                                       ["Component_1_Loading", "Component_2_Loading"]]
        if loading_row.empty:
            loadings_list.append([0, 0])
        else:
            loadings_list.append(loading_row.iloc[0].values)
    loadings_matrix = np.array(loadings_list)  # shape: (n_common_features, 2)
    
    # Project each element onto the PLS‑DA loading space (using dot product).
    coordinates = X_elements_normalized.dot(loadings_matrix)
    
    # Merge computed coordinates into the element DataFrame.
    merged_df = df_props.copy()
    merged_df["PLSDA_Component_1"] = coordinates[:, 0]
    merged_df["PLSDA_Component_2"] = coordinates[:, 1]
    
    # Map each element to its group.
    element_to_group = {}
    for group, elements in elements_by_group.items():
        for element in elements:
            element_to_group[element] = group
            
    groups = []
    for symbol in merged_df["Symbol"]:
        grp = element_to_group.get(symbol, "Other")
        groups.append(grp)
    merged_df["Group"] = groups
    
    # Create the scatter plot.
    plt.figure(figsize=(8, 6), dpi=500)
    plt.style.use("ggplot")
    
    # Plot by group: iterate over unique groups from the merged DataFrame.
    for group in merged_df["Group"].unique():
        subset = merged_df[merged_df["Group"] == group]
        color = group_colors.get(group, group_colors["Other"])
        plt.scatter(subset["PLSDA_Component_1"], subset["PLSDA_Component_2"], 
                    color=color, s=250, alpha=0.8, label=group)
        for _, row in subset.iterrows():
            plt.text(row["PLSDA_Component_1"], row["PLSDA_Component_2"], 
                     str(row["Symbol"]), fontsize=12,
                     horizontalalignment='center', verticalalignment='center')
    
    # -------- New: Draw dashed outline (filled with alpha=0.2) for sites if sites_df is provided --------
    if sites_df is not None:
        # Import the SiteElement class.
        # It is assumed that this class is stored in utils/class.py.
        

        # Initialize sets to collect unique primary elements for each site.
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
        
        # Define colors for outlines.
        site_outline_colors = {"2c": "#0348a1", "6h (2)": "#ffb01c", "RE": "#c3121e"}
        
        def plot_outline(subset, label, color):
            """
            Given a subset of points, plot an outline:
              - If there are exactly 2 points, draw a dashed line connecting them.
              - If there are 3 or more points, compute the convex hull,
                plot a dashed outline, and fill it with a transparent color.
            """
            points = subset[["PLSDA_Component_1", "PLSDA_Component_2"]].values
            n_points = len(points)
            if n_points < 2:
                return
            elif n_points == 2:
                # Draw dashed line connecting the two points.
                plt.plot(points[:, 0], points[:, 1], linestyle="--", 
                         color=color, linewidth=2, label=label)
            else:
                hull = ConvexHull(points)
                # Retrieve hull vertices and close the polygon by appending the first vertex.
                hull_points = points[hull.vertices]
                hull_points_closed = np.vstack([hull_points, hull_points[0]])
                # Plot dashed outline.
                plt.plot(hull_points_closed[:, 0], hull_points_closed[:, 1],
                         linestyle="--", color=color, linewidth=2, label=label)
                # Fill the polygon with the same color at alpha=0.2.
                plt.fill(hull_points_closed[:, 0], hull_points_closed[:, 1], 
                         color=color, alpha=0.2)
        
        # For site "2c":
        subset_2c = merged_df[merged_df["Symbol"].isin(unique_2c)]
        plot_outline(subset_2c, "2c Outline", site_outline_colors["2c"])
        
        # For site "6h (2)":
        subset_6h2 = merged_df[merged_df["Symbol"].isin(unique_6h2)]
        plot_outline(subset_6h2, "6h (2) Outline", site_outline_colors["6h (2)"])
        
        # For site "RE":
        subset_RE = merged_df[merged_df["Symbol"].isin(unique_RE)]
        plot_outline(subset_RE, "RE Outline", site_outline_colors["RE"])
    # -------------------------------------------------------------------------------------------
    
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig("elements_plot.png", dpi=500)
    plt.show()
    
    # Create output Excel file "coordinates.xlsx" with columns: Symbol, x, y
    coordinates_df = merged_df.loc[:, ["Symbol", "PLSDA_Component_1", "PLSDA_Component_2"]].copy()
    coordinates_df.columns = ["Symbol", "x", "y"]
    coordinates_df.to_excel("outputs/coordinates.xlsx", index=False)
    print("Coordinates saved to coordinates.xlsx")
    
    return coordinates, merged_df