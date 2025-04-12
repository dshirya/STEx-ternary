import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

def plot_confidence_ellipse(x, y, ax, n_std=2.0, edgecolor='black', facecolor='none', alpha=1.0, **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must have the same size")
    
    # Compute the covariance and its eigenvalues/eigenvectors
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eig(cov)
    
    # Determine the angle in degrees
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    
    # Width and height of the ellipse scaled by the standard deviation factor
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    # Create and add the ellipse patch with the specified alpha for transparency
    ellipse = Ellipse((mean_x, mean_y), width=width, height=height,
                      angle=angle, edgecolor=edgecolor,
                      facecolor=facecolor, lw=2, alpha=alpha, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

def run_pls_da(df, output_loadings_excel="outputs/PLS_DA_loadings.xlsx"):
    # -----------------------------
    # Prepare the data for PLS‑DA
    # -----------------------------
    # Drop non-numerical columns: "Filename", "Formula", "Site", "Site_Label"
    X = df.drop(columns=["Filename", "Formula", "Site", "Site_Label"])
    y = df["Site_Label"]

    # External scaling: standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Normalize each sample (row) to unit norm (L2 normalization) using a MinMaxScaler
    # (alternatively, you might want to use a row-wise normalization method)
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
    # Generate scatter plot with ellipsoids for class space and centroids (no legend for ellipses)
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
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    
    unique_classes = y.unique()
    color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}
    
    # Loop over each class to plot the points and the corresponding confidence ellipse
    for cls in unique_classes:
        idx = (y == cls)
        points = X_scores[idx, :]  # Points for the current class
        
        # Scatter the points for the class
        ax.scatter(points[:, 0], points[:, 1],
                   color=color_map[cls], label=cls, s=250, alpha=0.5, edgecolors="none")
        
        # Add the ellipsoidal space
        # You can adjust n_std for a wider/narrower ellipse if desired.
        plot_confidence_ellipse(points[:, 0], points[:, 1], ax, n_std=2.0,
                        edgecolor=color_map[cls],
                        facecolor=color_map[cls],  # Use the same color as the edge or another preferred color
                        alpha=0.2)
    
    ax.set_xlabel(f"LV1 ({explained_ratio[0]:.1f}%)", fontsize=18)
    ax.set_ylabel(f"LV2 ({explained_ratio[1]:.1f}%)", fontsize=18)
    
    # Adding and styling the legend
    legend = ax.legend(fontsize=18)
    for text in legend.get_texts():
        text.set_fontstyle('italic')
    
    plt.tick_params(axis='both', labelsize=18)

    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig("plots/PLS_DA_Scatter_Plot.png", dpi=500)
    plt.show()
    
    # --------------------------------------
    # Compute and output full loadings per component
    # --------------------------------------
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