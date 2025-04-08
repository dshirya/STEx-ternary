# plsda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

def run_pls_da(df, output_loadings_excel="PLS_DA_Full_Loadings.xlsx"):
    # -----------------------------
    # Prepare the data for PLS‑DA
    # -----------------------------
    # Drop non-numerical columns: "Filename", "Formula", "Site", "Site_Label"
    X = df.drop(columns=["Filename", "Formula", "Site", "Site_Label"])
    y = df["Site_Label"]

    # External scaling: standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Normalize each sample (row) to unit norm (L2 normalization)
    minmax_scaler = MinMaxScaler()
    X_normalized = minmax_scaler.fit_transform(X_scaled)
    
    # One-hot encode the class labels for PLS‑DA
    y_dummies = pd.get_dummies(y)
    Y = y_dummies.values  # shape: (n_samples, n_classes)
    
    # -------------------------------------
    # Fit the PLS‑DA model with 2 components
    # -------------------------------------
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_normalized, Y)
    
    # Get the X scores (latent space projections)
    X_scores = pls.x_scores_
    
    # ---------------------------------------
    # Compute explained variance (for labels)
    # ---------------------------------------
    total_variance = np.sum(np.var(X_normalized, axis=0))
    explained_variances = np.var(X_scores, axis=0)
    explained_ratio = explained_variances / total_variance * 100
    
    # -----------------
    # Generate scatter plot
    # -----------------
    colors = [
        "#c3121e",  # Sangre
        "#0348a1",  # Neptune
        "#ffb01c",  # Pumpkin
        "#027608",  # Clover
        "#1dace6",  # Cerulean
        "#9c5300",  # Cocoa
        "#9966cc",  # Amethyst
    ]
    
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 6), dpi=500)
    
    unique_classes = y.unique()
    color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}
    
    for cls in unique_classes:
        idx = (y == cls)
        plt.scatter(X_scores[idx, 0], X_scores[idx, 1],
                    color=color_map[cls], label=cls, s=250, alpha=0.5, edgecolors="none")
    
    plt.xlabel(f"LV1 ({explained_ratio[0]:.1f}%)", fontsize=16)
    plt.ylabel(f"LV2 ({explained_ratio[1]:.1f}%)", fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig("PLS_DA_Scatter_Plot.png", dpi=500)
    plt.show()
    
    # --------------------------------------
    # Compute and output full loadings per component
    # --------------------------------------
    # Use the x_loadings_ from the model; each column corresponds to a component.
    x_loadings = pls.x_loadings_
    loadings_df = pd.DataFrame(
        x_loadings,
        columns=[f"Component_{i+1}_Loading" for i in range(pls.n_components)]
    )
    loadings_df.insert(0, "Feature", X.columns.tolist())
    
    # Save to Excel
    loadings_df.to_excel(output_loadings_excel, index=False)
    print(f"Full loadings saved to {output_loadings_excel}.")
    
    # Return the loadings DataFrame for use in subsequent PCA analysis
    return loadings_df