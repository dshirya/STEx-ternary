import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define your element groups and colors (unchanged):
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

def plot_elements_from_plsda_weights(pls_weights, element_properties_file="data/elemental-property-list.xlsx"):
    """
    Projects chemical elements into the PLS‑DA space using the weight matrix (instead of loadings)
    computed from the PLS‑DA model. The element properties are scaled (using StandardScaler) and then
    normalized (using MinMaxScaler), exactly as for the PLS‑DA features. Each element is then projected 
    into the 2D latent space via the dot product with the weight matrix.
    
    In addition, elements are colored according to their group using the given dictionaries.
    
    Parameters:
      - pls_weights: DataFrame with columns "Feature", "Component_1_Weight", "Component_2_Weight".
                     The "Feature" names should correspond to property names (columns) in the element 
                     property file.
      - element_properties_file: Path to the Excel file containing element properties.
          It must include a "Symbol" column (element labels); all other columns are taken as properties.
    
    Returns:
      - coordinates: A NumPy array (n_elements x 2) with the projected coordinates.
      - merged_df: A DataFrame containing the original element properties along with the computed PLS‑DA coordinates.
    """
    # Load the element property data.
    df_props = pd.read_excel(element_properties_file)
    df_props.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaN values.
    
    # Identify property columns (all columns except the label "Symbol")
    prop_columns = [col for col in df_props.columns if col != "Symbol"]
    
    # Standardize column names for robust matching.
    pls_weights["Feature_std"] = pls_weights["Feature"].str.strip().str.lower()
    prop_columns_std = {col: col.strip().lower() for col in prop_columns}
    
    # Identify common features by comparing standardized names.
    common_features = []
    for orig_col, std_col in prop_columns_std.items():
        if std_col in set(pls_weights["Feature_std"]):
            common_features.append(orig_col)
    
    if not common_features:
        raise ValueError("No common features found between PLS‑DA weights and element property file columns!")
    
    print("Common features between property file and PLS‑DA:", common_features)
    
    # Extract data for the common features.
    X_elements = df_props[common_features].values  # shape: (n_elements, n_common_features)
    
    # Scale the element properties using StandardScaler and then normalize using MinMaxScaler.
    scaler = StandardScaler()
    X_elements_scaled = scaler.fit_transform(X_elements)
    minmax_scaler = MinMaxScaler()
    X_elements_normalized = minmax_scaler.fit_transform(X_elements_scaled)
    
    # Build the weight matrix for the common features (ensuring the order matches).
    weights_list = []
    for feature in common_features:
        key = feature.strip().lower()
        weight_row = pls_weights.loc[pls_weights["Feature_std"] == key, 
                                     ["Component_1_Weight", "Component_2_Weight"]]
        if weight_row.empty:
            weights_list.append([0, 0])
        else:
            weights_list.append(weight_row.iloc[0].values)
    weights_matrix = np.array(weights_list)  # shape: (n_common_features, 2)
    
    # Project each element using the weight matrix (dot product).
    coordinates = X_elements_normalized.dot(weights_matrix)
    
    # Compute explained variances for the projected data (for axis labels).
    total_variance = np.sum(np.var(X_elements_normalized, axis=0))
    comp_variances = np.var(coordinates, axis=0)
    explained_ratio = comp_variances / total_variance * 100
    xlabel = f"PLS component 1 ({explained_ratio[0]:.1f}%)"
    ylabel = f"PLS component 2 ({explained_ratio[1]:.1f}%)"
    
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
    
    # Create the plot.
    plt.figure(figsize=(8, 6), dpi=500)
    plt.style.use("ggplot")
    
    # Plot each group separately.
    for group in merged_df["Group"].unique():
        subset = merged_df[merged_df["Group"] == group]
        color = group_colors.get(group, group_colors["Other"])
        plt.scatter(subset["PLSDA_Component_1"], subset["PLSDA_Component_2"],
                    color=color, s=250, alpha=0.8, label=group)
        for _, row in subset.iterrows():
            plt.text(row["PLSDA_Component_1"], row["PLSDA_Component_2"],
                     str(row["Symbol"]), fontsize=13,
                     horizontalalignment='center', verticalalignment='center')
    
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    
    plt.savefig("elements_plot1.png", dpi=500)
    plt.show()
    
    return coordinates, merged_df