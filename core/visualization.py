import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from utils.class_object import SiteElement

# ---------------- Appearance Variables ----------------
COORD_MULTIPLIER = 5

# Circle settings
BASE_CIRCLE_RADIUS = 0.6
BASE_CIRCLE_LINEWIDTH = 2
PATCH_CIRCLE_RADIUS = 0.59

# Text settings
TEXT_SIZE = 36

# Figure size factors (to compute dimensions based on the coordinate range)
FIGURE_WIDTH_FACTOR = 0.8
FIGURE_HEIGHT_FACTOR = 0.8

# Convex hull outline settings
HULL_LINETYPE = "--"
HULL_LINEWIDTH = 2
HULL_FILL_ALPHA = 0.2

# Site colors for patch overlays and convex hull outlines
SITE_OUTLINE_COLORS = {"M": "#0348a1", "X": "#ffb01c", "R": "#c3121e"}

# Compound marker settings
COMPOUND_MARKER_SIZE = 20
COMPOUND_LINE_WIDTH = 1
COMPOUND_MARKER_COLOR = "black"  # default compound marker color

# --------------------------------------------------------

def parse_formula_elements(formula):
    """
    Parses a chemical formula string to extract a list of (element, count) pairs.

    For example:
        "La2NiO4" -> [("La", 2.0), ("Ni", 1.0), ("O", 4.0)]

    If no numeric count is given, defaults to 1.0.
    """
    s = str(formula).strip()
    if not s:
        return []
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
    matches = re.findall(pattern, s)
    results = []
    for element, count_str in matches:
        count = float(count_str) if count_str != "" else 1.0
        results.append((element, count))
    return results

def visualize_elements(coord_df, sites_df=None, compounds_markers=True):
    """
    Visualizes chemical element coordinates with site-based patches, convex hull outlines,
    and compound weighted coordinate markers with connecting lines.
    
    Parameters:
      coordinate_file : str
          Path to the Excel file with coordinates.
      sites_df : DataFrame or None
          DataFrame with site information. Expected columns: "Filename", "Formula", "M", "X", "R", "Notes".
      compounds_markers : bool
          When True, compute and plot compound markers and connecting lines.
          
    The function saves the plot with a dynamically generated file name and displays it.
    """
    # ----- Load coordinates and apply scaling -----
    coord_df = coord_df.copy()
    coord_df["x"] = coord_df["x"] * COORD_MULTIPLIER
    coord_df["y"] = coord_df["y"] * COORD_MULTIPLIER

    # Determine plot dimensions based on the coordinate range.
    x_min, x_max = coord_df["x"].min(), coord_df["x"].max()
    y_min, y_max = coord_df["y"].min(), coord_df["y"].max()
    fig_width = (x_max - x_min) * FIGURE_WIDTH_FACTOR
    fig_height = (y_max - y_min) * FIGURE_HEIGHT_FACTOR

    # Create figure with white background (no grid, no axes).
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    # ----- Plot base circles and element text -----
    element_coords = {}  # maps element symbol to (x, y) coordinates.
    for idx, row in coord_df.iterrows():
        x, y, element = row["x"], row["y"], row["Symbol"]
        element_coords[element] = (x, y)
        base_circle = plt.Circle((x, y), radius=BASE_CIRCLE_RADIUS,
                                 edgecolor="black", facecolor="none",
                                 linewidth=BASE_CIRCLE_LINEWIDTH, alpha=1.0, zorder=3)
        ax.add_patch(base_circle)
        ax.text(x, y, element, fontsize=TEXT_SIZE, ha="center", va="center", zorder=4)

    # ----- Process site information to extract unique element sets -----
    unique_sites = {"M": set(), "X": set(), "R": set()}
    if sites_df is not None:
        for _, row in sites_df.iterrows():
            site = SiteElement(row)
            if site.site_M:
                unique_sites["M"].add(site.site_M)
            if site.site_X:
                unique_sites["X"].add(site.site_X)
            if site.site_R:
                unique_sites["R"].add(site.site_R)

    # ----- Draw colored patches on top of base circles -----
    for element, (x, y) in element_coords.items():
        patch_color = None
        for site in ["M", "X", "R"]:
            if element in unique_sites[site]:
                patch_color = SITE_OUTLINE_COLORS[site]
                break
        if patch_color is not None:
            patch_circle = plt.Circle((x, y), radius=PATCH_CIRCLE_RADIUS,
                                      edgecolor=None, facecolor=patch_color,
                                      alpha=0.5, zorder=3.5)
            ax.add_patch(patch_circle)

    # ----- Define helper function to plot convex hull outlines -----
    def plot_site_outline(element_set, color, label):
        points = []
        for elem in element_set:
            if elem in element_coords:
                points.append(element_coords[elem])
        points = np.array(points)
        n_points = len(points)
        if n_points < 2:
            return
        elif n_points == 2:
            ax.plot(points[:, 0], points[:, 1], linestyle=HULL_LINETYPE,
                    color=color, linewidth=HULL_LINEWIDTH, label=label)
        else:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points_closed = np.vstack([hull_points, hull_points[0]])
            ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1],
                    linestyle=HULL_LINETYPE, color=color, linewidth=HULL_LINEWIDTH, label=label)
            ax.fill(hull_points_closed[:, 0], hull_points_closed[:, 1],
                    color=color, alpha=HULL_FILL_ALPHA)

    # ----- Draw convex hull outlines for each site group -----
    if sites_df is not None:
        for site, color in SITE_OUTLINE_COLORS.items():
            plot_site_outline(unique_sites[site], color, f"{site} Outline")

    # ----- Compute weighted compound markers from formula column -----
    candidate_found = False  # flag to track if there is any candidate entry
    if sites_df is not None and compounds_markers:
        for idx, row in sites_df.iterrows():
            formula = row["Formula"]
            items = parse_formula_elements(formula)
            if not items:
                continue
            total_count = sum(count for _, count in items)
            weighted_sum = np.array([0.0, 0.0])
            compound_elements = []  # list of elements with coordinates in the compound
            for el, count in items:
                if el in element_coords:
                    compound_elements.append(el)
                    weighted_sum += count * np.array(element_coords[el])
            if total_count > 0 and compound_elements:
                weighted_coord = weighted_sum / total_count
                note = str(row.get("Notes", "")).strip().lower()
                # Check if this compound row is marked as candidate.
                if note == "candidate":
                    candidate_found = True
                    marker_color = "#027608"
                    alpha = 1
                    size = COMPOUND_MARKER_SIZE * 1.5
                    line_width = COMPOUND_LINE_WIDTH * 3  # Double the line width for candidate.
                    # Instead of plotting a filled marker, create a Circle patch:
                    candidate_marker = plt.Circle((weighted_coord[0], weighted_coord[1]),
                                                radius=0.25,  # size/2 gives an appropriate radius
                                                edgecolor=marker_color,
                                                facecolor="none",  # no filling
                                                linestyle="--",   # dashed outline
                                                linewidth=line_width,
                                                alpha=alpha,
                                                zorder=5)
                    ax.add_patch(candidate_marker)
                    linestyle="--"
                else:
                    marker_color = COMPOUND_MARKER_COLOR
                    alpha = 0.3
                    size = COMPOUND_MARKER_SIZE
                    line_width = COMPOUND_LINE_WIDTH
                    # For standard markers, use the filled marker version:
                    ax.plot(weighted_coord[0], weighted_coord[1],
                            marker="o", markersize=size,
                            markerfacecolor=marker_color,
                            zorder=5, alpha=alpha,
                            markeredgecolor="none")
                    linestyle="-"

                # Draw connecting lines from the compound marker to each element of the compound.
                for el in compound_elements:
                    el_coord = element_coords[el]
                    ax.plot([weighted_coord[0], el_coord[0]], [weighted_coord[1], el_coord[1]],
                            color=marker_color, linestyle=linestyle,
                            linewidth=line_width, zorder=4, alpha=alpha)

    plt.tight_layout()

    # ----- Build output file name based on conditions -----
    output_filename = "plots/elements_visualization"
    if compounds_markers:
        output_filename += "_compound"
    if sites_df is not None and candidate_found:
        output_filename += "_candidate"
        
    output_filepath = output_filename + ".svg"
    plt.savefig(output_filepath, dpi=500, bbox_inches="tight")
    plt.show()