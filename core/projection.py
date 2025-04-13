import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.class_object import SiteElement

def prepare_coordinates(pls_loadings, element_properties_file):
    df_props = pd.read_excel(element_properties_file)
    df_props.dropna(axis=1, how='all', inplace=True)

    prop_columns = [col for col in df_props.columns if col != "Symbol"]
    pls_loadings["Feature_std"] = pls_loadings["Feature"].str.strip().str.lower()
    prop_columns_std = {col: col.strip().lower() for col in prop_columns}

    common_features = [orig_col for orig_col, std_col in prop_columns_std.items()
                       if std_col in set(pls_loadings["Feature_std"])]

    if not common_features:
        raise ValueError("No common features found between PLS‑DA loadings and element property file columns!")

    print("Common features between property file and PLS‑DA:", common_features)

    X = df_props[common_features].values
    X_scaled = StandardScaler().fit_transform(X)
    X_normalized = MinMaxScaler().fit_transform(X_scaled)

    loadings = []
    for feature in common_features:
        key = feature.strip().lower()
        row = pls_loadings.loc[pls_loadings["Feature_std"] == key,
                               ["Component_1_Loading", "Component_2_Loading"]]
        loadings.append(row.iloc[0].values if not row.empty else [0, 0])
    loadings_matrix = np.array(loadings)

    coordinates = X_normalized.dot(loadings_matrix)

    df_props["PLSDA_Component_1"] = coordinates[:, 0]
    df_props["PLSDA_Component_2"] = coordinates[:, 1]

    return df_props, coordinates

def plot_elements(df, group_colors, output_path="plots/elements_plot.svg"):
    plt.figure(figsize=(8, 6), dpi=500)
    plt.style.use("ggplot")

    for group in df["Group"].unique():
        subset = df[df["Group"] == group]
        color = group_colors.get(group, "grey")
        plt.scatter(subset["PLSDA_Component_1"], subset["PLSDA_Component_2"],
                    color=color, s=600, alpha=0.6, label=group)
        for _, row in subset.iterrows():
            plt.text(row["PLSDA_Component_1"], row["PLSDA_Component_2"],
                     str(row["Symbol"]), fontsize=18,
                     horizontalalignment='center', verticalalignment='center')

    plt.xlabel("PC1", fontsize=18)
    plt.ylabel("PC2", fontsize=18)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.show()

def plot_elements_from_plsda_loadings(pls_loadings,
                                      sites_df=None,
                                      element_properties_file="data/elemental-property-list.xlsx"):
    df_props, coordinates = prepare_coordinates(pls_loadings, element_properties_file)

    # Grouping logic (optional)
    group_colors = {"R": "#c3121e", 
                    "M": "#0348a1", 
                    "X": "#ffb01c", 
                    "Other": "grey"}
    if sites_df is not None:
        unique_M, unique_X, unique_R = set(), set(), set()
        for _, row in sites_df.iterrows():
            site = SiteElement(row)
            if site.site_M: 
                unique_M.add(site.site_M)
            if site.site_X: 
                unique_X.add(site.site_X)
            if site.site_R: 
                unique_R.add(site.site_R)

        def get_group(symbol):
            if symbol in unique_M: 
                return "M"
            elif symbol in unique_X: 
                return "X"
            elif symbol in unique_R: 
                return "R"
            return "Other"

        df_props["Group"] = df_props["Symbol"].apply(get_group)
    else:
        df_props["Group"] = "Other"

    # Plot
    plot_elements(df_props, group_colors)

    # Save coordinates
    coordinates_df = df_props[["Symbol", "PLSDA_Component_1", "PLSDA_Component_2"]].copy()
    coordinates_df.columns = ["Symbol", "x", "y"]
    coordinates_df.to_excel("outputs/coordinates.xlsx", index=False)
    print("Coordinates saved to outputs/coordinates.xlsx")

    return coordinates_df