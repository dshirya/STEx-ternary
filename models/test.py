import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
import random, math

# ----- Helper functions for minimum enclosing circle -----

def dist(p, q):
    """Compute Euclidean distance between points p and q."""
    return math.hypot(p[0] - q[0], p[1] - q[1])

def is_in_circle(p, center, radius):
    """Check if point p is inside the circle given by center and radius (with tolerance)."""
    return dist(p, center) <= radius + 1e-8

def circle_from_two_points(p, q):
    """Return the circle defined by two points p and q."""
    center = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
    radius = dist(p, q) / 2.0
    return center, radius

def circle_from_three_points(p, q, r):
    """Return the circle defined by three non-collinear points p, q, and r."""
    d = 2 * (p[0]*(q[1]-r[1]) + q[0]*(r[1]-p[1]) + r[0]*(p[1]-q[1]))
    if d == 0:
        # Collinear points; this should not normally occur.
        return p, float('inf')
    ux = ((p[0]**2 + p[1]**2)*(q[1]-r[1]) + (q[0]**2 + q[1]**2)*(r[1]-p[1]) + (r[0]**2 + r[1]**2)*(p[1]-q[1])) / d
    uy = ((p[0]**2 + p[1]**2)*(r[0]-q[0]) + (q[0]**2 + q[1]**2)*(p[0]-r[0]) + (r[0]**2 + r[1]**2)*(q[0]-p[0])) / d
    center = (ux, uy)
    radius = dist(center, p)
    return center, radius

def minimum_enclosing_circle(points):
    """
    Computes the minimum enclosing circle for a set of 2D points using a simple randomized algorithm.
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

# ----- Updated PLS-DA plotting function using minimum enclosing circles -----

def pls_da_circle(df, output_loadings_excel="PLS_DA_Full_Loadings.xlsx"):
    # -----------------------------
    # Prepare the data for PLS‑DA
    # -----------------------------
    # Drop non-numerical columns: "Filename", "Formula", "Site", "Site_Label"
    X = df.drop(columns=["Filename", "Formula", "Site", "Site_Label"])
    y = df["Site_Label"]

    # External scaling: standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Normalize each sample (row) using MinMaxScaler (for range scaling)
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
    # Plotting setup
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
    
    # For each class, compute the minimum enclosing circle and plot
    for cls in unique_classes:
        idx = (y == cls)
        points = X_scores[idx, :]  # Points for this class
        pts_list = [tuple(point) for point in points]
        center, radius = minimum_enclosing_circle(pts_list)
        
        # Plot the scatter points for the class
        plt.scatter(points[:, 0], points[:, 1],
                    color=color_map[cls], label=cls, s=250, alpha=0.5, edgecolors="none")
                    
        # Optionally, mark the center of the minimum circle
        plt.scatter(center[0], center[1], marker='X', color='black', 
                    s=150, edgecolors='white', linewidth=2, label='_nolegend_')
        
        # Generate circle coordinates for the minimum enclosing circle
        angle = np.linspace(0, 2 * np.pi, 100)
        circle_x = center[0] + radius * np.cos(angle)
        circle_y = center[1] + radius * np.sin(angle)
        plt.plot(circle_x, circle_y, color=color_map[cls], lw=2, linestyle='--')
    
    plt.xlabel(f"LV1 ({explained_ratio[0]:.1f}%)", fontsize=16)
    plt.ylabel(f"LV2 ({explained_ratio[1]:.1f}%)", fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig("PLS_DA_Scatter_Plot_circle.png", dpi=500)
    plt.show()