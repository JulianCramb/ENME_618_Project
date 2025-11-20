import laspy
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

def load_lidar(path):
    """
    Load a LAZ / COPC LAZ using laspy with full attribute safety.
    Returns: xyz, intensity, classification, scan_angle, header
    """
    las = laspy.read(path)

    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    dims = las.point_format.dimension_names

    intensity = las.intensity.astype(np.float32) if "intensity" in dims else None
    classification = las.classification.astype(np.uint8) if "classification" in dims else None

    if "scan_angle" in dims:
        scan_angle = las.scan_angle.astype(np.float32)
    elif "scan_angle_rank" in dims:
        scan_angle = las.scan_angle_rank.astype(np.float32)
    else:
        scan_angle = None

    return xyz, intensity, classification, scan_angle, las.header


def visualize(xyz, classification):
    """
    Visualize the point cloud in PyVista colored by classification.
    Selected LAS classes with labels:
    1: Unclassified
    2: Ground
    3: Low Vegetation
    4: Medium Vegetation
    5: High Vegetation
    7: Low Noise
    9: Water
    17: Bridge Deck
    18: High Noise
    """
    if classification is None:
        print("No classification field provided.")
        return

    # Define classes and their labels
    class_labels = {
        1: "Unclassified",
        2: "Ground",
        3: "Low Vegetation",
        4: "Medium Vegetation",
        5: "High Vegetation",
        7: "Low Noise",
        9: "Water",
        17: "Bridge Deck",
        18: "High Noise"
    }

    # Define RGB colors for each class
    colors = {
        1: [0.6, 0.6, 0.6],    # gray
        2: [0.8, 0.7, 0.5],    # light brown
        3: [0.5, 1.0, 0.5],    # light green
        4: [0.0, 0.8, 0.0],    # medium green
        5: [0.0, 0.5, 0.0],    # dark green
        7: [1.0, 0.0, 1.0],    # pink
        9: [0.0, 0.0, 1.0],    # blue
        17: [1.0, 0.5, 0.0],   # orange
        18: [1.0, 0.0, 0.0],   # red
    }

    # Mask points in selected classes
    mask = np.isin(classification, list(class_labels.keys()))
    xyz_vis = xyz[mask]
    class_vis = classification[mask]

    cloud = pv.PolyData(xyz_vis)
    cloud["classification"] = class_vis

    # Map class IDs to RGB colors
    point_colors = np.array([colors[c] for c in class_vis])
    
    plotter = pv.Plotter()
    plotter.add_points(
        cloud,
        scalars=point_colors,
        rgb=True,
        render_points_as_spheres=False,
        point_size=2
    )

    # Add legend using a list of tuples
    legend_entries = [(label, colors[c]) for c, label in class_labels.items()]
    plotter.add_legend(legend_entries)

    plotter.show()

NOISE_CLASS = 18  # LAS standard for high-noise/ghost points

# ------------------ Noise Functions ------------------

def add_range_noise(xyz, sigma_z=0.03):
    noisy_xyz = xyz.copy()
    noisy_xyz[:, 2] += np.random.normal(0, sigma_z, size=xyz.shape[0])
    return noisy_xyz

def add_horizontal_noise(xyz, sigma_xy=0.05):
    noisy_xyz = xyz.copy()
    noisy_xyz[:, :2] += np.random.normal(0, sigma_xy, size=(xyz.shape[0], 2))
    return noisy_xyz

def add_scan_angle_noise(xyz, scan_angle, base_sigma=0.02):
    noisy_xyz = xyz.copy()
    if scan_angle is not None:
        sigma = base_sigma * np.abs(np.tan(scan_angle))
        noisy_xyz[:, 2] += np.random.normal(0, sigma, size=xyz.shape[0])
    return noisy_xyz

def add_surface_noise(xyz, classification,
                      ground_sigma=0.02,
                      veg_sigma=0.10,
                      building_sigma=0.05):
    noisy_xyz = xyz.copy()
    noise = np.zeros(xyz.shape[0])

    g_mask = classification == 2
    noise[g_mask] = np.random.normal(0, ground_sigma, g_mask.sum())

    v_mask = np.isin(classification, [3,4,5])
    noise[v_mask] = np.random.normal(0, veg_sigma, v_mask.sum())

    b_mask = classification == 6
    noise[b_mask] = np.random.normal(0, building_sigma, b_mask.sum())

    other = ~(g_mask | v_mask | b_mask)
    noise[other] = np.random.normal(0, ground_sigma, other.sum())

    noisy_xyz[:, 2] += noise
    return noisy_xyz

def add_intensity_noise(intensity, sigma_add=5.0, sigma_mult=0.05):
    noisy_intensity = intensity.copy()
    mult = np.random.normal(0, sigma_mult, size=intensity.shape)
    add = np.random.normal(0, sigma_add, size=intensity.shape)
    noisy_intensity = intensity * (1 + mult) + add
    return noisy_intensity

def add_outliers(xyz, classification=None, num_outliers=10000, bounds=None):
    if bounds is None:
        bounds = {
            "x": (xyz[:,0].min(), xyz[:,0].max()),
            "y": (xyz[:,1].min(), xyz[:,1].max()),
            "z": (xyz[:,2].min(), xyz[:,2].max()),
        }
    outliers = np.column_stack([
        np.random.uniform(bounds["x"][0], bounds["x"][1], num_outliers),
        np.random.uniform(bounds["y"][0], bounds["y"][1], num_outliers),
        np.random.uniform(bounds["z"][0], bounds["z"][1], num_outliers),
    ])
    outlier_classes = np.full(num_outliers, NOISE_CLASS, dtype=int)

    xyz_combined = np.vstack([xyz, outliers])
    if classification is not None:
        classification_combined = np.hstack([classification, outlier_classes])
        return xyz_combined, classification_combined
    else:
        return xyz_combined, outlier_classes

# ------------------ MASTER SIMULATION ------------------

def simulate_noise(
    xyz,
    intensity=None,
    classification=None,
    scan_angle=None,
    apply={
        "range": True,
        "horizontal": True,
        "scan_angle": True,
        "surface": True,
        "outliers": True,
        "intensity": True,
    },
    num_outliers=10000
):
    """
    Apply noise to positions and intensity, but **do not change original classifications**.
    Only added outliers are labeled 18.
    """
    noisy_xyz = xyz.copy()
    noisy_class = classification.copy() if classification is not None else np.zeros(xyz.shape[0], dtype=int)
    noisy_intensity = intensity.copy() if intensity is not None else None

    # ------------------ Apply positional noise ------------------
    if apply.get("range", False):
        noisy_xyz = add_range_noise(noisy_xyz)
    if apply.get("horizontal", False):
        noisy_xyz = add_horizontal_noise(noisy_xyz)
    if apply.get("scan_angle", False) and (scan_angle is not None):
        noisy_xyz = add_scan_angle_noise(noisy_xyz, scan_angle)
    if apply.get("surface", False) and (classification is not None):
        noisy_xyz = add_surface_noise(noisy_xyz, classification)
    if apply.get("intensity", False) and (intensity is not None):
        noisy_intensity = add_intensity_noise(noisy_intensity)

    # ------------------ Add outliers labeled as 18 ------------------
    if apply.get("outliers", False):
        noisy_xyz, noisy_class = add_outliers(noisy_xyz, noisy_class, num_outliers=num_outliers)

    return noisy_xyz, noisy_class, noisy_intensity