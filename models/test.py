import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from utils.class_object import SiteElement

# ---------------- Appearance Variables ----------------
# Coordinate scaling
COORD_MULTIPLIER = 5

# Circle settings
BASE_CIRCLE_RADIUS = 0.5
BASE_CIRCLE_LINEWIDTH = 2
PATCH_CIRCLE_RADIUS = 0.49

# Text settings
TEXT_SIZE = 20

# Figure size factors (to compute dimensions based on the coordinate range)
FIGURE_WIDTH_FACTOR = 0.8
FIGURE_HEIGHT_FACTOR = 0.8

# Convex hull outline settings
HULL_LINETYPE = "--"
HULL_LINEWIDTH = 2
HULL_FILL_ALPHA = 0.2

# Site colors for patch overlays and convex hull outlines
SITE_OUTLINE_COLORS = {"2c": "#0348a1", "6h (2)": "#ffb01c", "RE": "#c3121e"}

# Compound marker settings
COMPOUND_MARKER_SIZE = 10
COMPOUND_LINE_WIDTH = 1
COMPOUND_MARKER_COLOR = "black"
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

def visualize_elements(sites_df=None, coordinate_file="outputs/coordinates.xlsx"):
    """
    Visualizes chemical element coordinates with site-based patches, convex hull outlines,
    and compound weighted coordinate markers with connecting lines.

    The coordinates are read from an Excel file with columns: "Symbol", "x", "y".
      - All coordinate values are multiplied by COORD_MULTIPLIER.
      - Figure dimensions are computed as:
            fig_width = (x_max - x_min) * FIGURE_WIDTH_FACTOR
            fig_height = (y_max - y_min) * FIGURE_HEIGHT_FACTOR

    Each element is plotted as a base circle (edge only) with centered text.
    If sites_df is provided (with columns "Filename", "Formula", "2c", "6h (2)", "RE"),
    it is processed via the SiteElement class to extract unique primary element symbols for each site.
    For elements belonging to a site, a filled patch (overlay circle) is drawn using a site-specific color.
    Convex hull outlines (dashed and filled with transparency) are drawn around the groups of elements per site.
    
    Additionally, the compound formula (from the "Formula" column in sites_df) is parsed and its
    weighted coordinate is computed using the element coordinates. A black circle marker is plotted at
    the weighted coordinate, and black lines are drawn connecting this marker to each element 
    that forms the compound.

    Parameters:
      coordinate_file : str
          Path to the Excel file with coordinates.
      sites_df : DataFrame or None
          DataFrame with site information. Expected columns: "Filename", "Formula", "2c", "6h (2)", "RE".

    The function saves the plot as "elements_visualization.png" and displays it.
    """
    # ----- Load coordinates and apply scaling -----
    coord_df = pd.read_excel(coordinate_file)
    # Multiply coordinates by the defined multiplier.
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
    element_coords = {}  # dictionary mapping element symbol to its (x,y) coordinates.
    for idx, row in coord_df.iterrows():
        x, y, element = row["x"], row["y"], row["Symbol"]
        element_coords[element] = (x, y)
        base_circle = plt.Circle((x, y), radius=BASE_CIRCLE_RADIUS,
                                 edgecolor="black", facecolor="none",
                                 linewidth=BASE_CIRCLE_LINEWIDTH, alpha=1.0, zorder=3)
        ax.add_patch(base_circle)
        ax.text(x, y, element, fontsize=TEXT_SIZE, ha="center", va="center", zorder=4)

    # ----- Process site information to extract unique element sets -----
    unique_sites = {"2c": set(), "6h (2)": set(), "RE": set()}
    if sites_df is not None:
        for _, row in sites_df.iterrows():
            site = SiteElement(row)
            if site.site_2c:
                unique_sites["2c"].add(site.site_2c)
            if site.site_6h2:
                unique_sites["6h (2)"].add(site.site_6h2)
            if site.site_RE:
                unique_sites["RE"].add(site.site_RE)

    # ----- Draw colored patches on top of base circles -----
    for element, (x, y) in element_coords.items():
        patch_color = None
        for site in ["2c", "6h (2)", "RE"]:
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
    if sites_df is not None:
        # For each compound (row) in the sites DataFrame, parse the formula and compute a weighted coordinate.
        for idx, row in sites_df.iterrows():
            formula = row["Formula"]
            items = parse_formula_elements(formula)
            if not items:
                continue
            total_count = sum(count for _, count in items)
            weighted_sum = np.array([0.0, 0.0])
            compound_elements = []  # store elements in the compound that have coordinates
            for el, count in items:
                if el in element_coords:
                    compound_elements.append(el)
                    weighted_sum += count * np.array(element_coords[el])
            if total_count > 0 and compound_elements:
                weighted_coord = weighted_sum / total_count
                # Plot the compound marker (black circle marker).
                ax.plot(weighted_coord[0], weighted_coord[1], marker="o",
                        markersize=COMPOUND_MARKER_SIZE, markerfacecolor=COMPOUND_MARKER_COLOR,
                        zorder=5, alpha= 0.5, markeredgecolor = "none")
                # Draw connecting lines from the compound marker to each element used in the compound.
                for el in compound_elements:
                    el_coord = element_coords[el]
                    ax.plot([weighted_coord[0], el_coord[0]], [weighted_coord[1], el_coord[1]],
                            color=COMPOUND_MARKER_COLOR, linestyle="-", linewidth=COMPOUND_LINE_WIDTH, zorder=4, alpha = 0.3)

    plt.tight_layout()
    plt.savefig("elements_visualization.png", dpi=500, bbox_inches="tight")
    plt.show()
