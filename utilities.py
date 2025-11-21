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
    
    #classification = las.classification.astype(np.uint8) if "classification" in dims else None
    classification = np.array(las.classification, dtype=np.uint8) if "classification" in dims else None

    if "scan_angle" in dims:
        scan_angle = las.scan_angle.astype(np.float32)
    elif "scan_angle_rank" in dims:
        scan_angle = las.scan_angle_rank.astype(np.float32)
    else:
        scan_angle = None

    return xyz, intensity, classification, scan_angle, las.header

def print_lidar(xyz, intensity, classification, output_path, header):
    las = laspy.LasData(header)

    las.x = xyz[:,0]
    las.y = xyz[:,1]
    las.z = xyz[:,2]
    las.intensity = intensity
    las.classification = classification

    # 4. Write the data to a file
    las.write(output_path)
    print(f"Successfully saved LAS file to: {output_path}")


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
        6: "Building",
        7: "Low Noise",
        9: "Water",
        17: "Bridge Deck",
        18: "High Noise",
        28: "Urban"
    }

    # Define RGB colors for each class
    colors = {
        1: [0.6, 0.6, 0.6],    # gray
        2: [0.8, 0.7, 0.5],    # light brown
        3: [0.5, 1.0, 0.5],    # light green
        4: [0.0, 0.8, 0.0],    # medium green
        5: [0.0, 0.5, 0.0],    # dark green
        6: [0.4, 0.0, 0.6],    # dark purple
        7: [1.0, 0.0, 1.0],    # pink
        9: [0.0, 0.0, 1.0],    # blue
        17: [1.0, 0.5, 0.0],   # orange
        18: [1.0, 0.0, 0.0],   # red
        28: [0.8, 0.6, 1.0],   # light purple
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
def random_indices(n_points, n_sample):
    n_sample = min(n_sample, n_points)
    return np.random.choice(n_points, n_sample, replace=False)

def select_random_indices(n_total, n_select, used_indices):
    available = np.setdiff1d(np.arange(n_total), used_indices, assume_unique=True)
    chosen = np.random.choice(available, size=n_select, replace=False)
    return chosen

def add_range_noise(xyz, idx, sigma_z=0.03):
    xyz = xyz.copy()
    xyz[idx, 2] += np.random.normal(0, sigma_z, size=len(idx))
    return xyz


def add_horizontal_noise(xyz, idx, sigma_xy=0.05):
    xyz = xyz.copy()
    xyz[idx, :2] += np.random.normal(0, sigma_xy, size=(len(idx), 2))
    return xyz


def add_scan_angle_noise(xyz, scan_angle, idx, base_sigma=0.02):
    xyz = xyz.copy()
    if scan_angle is not None:
        sigma = base_sigma * np.abs(np.tan(scan_angle[idx]))
        xyz[idx, 2] += np.random.normal(0, sigma)
    return xyz


def add_surface_noise(xyz, classification, idx,
                      ground_sigma=0.02,
                      veg_sigma=0.10,
                      building_sigma=0.05):
    xyz = xyz.copy()
    noise = np.zeros(len(idx))

    selected_class = classification[idx]

    g = selected_class == 2
    v = np.isin(selected_class, [3,4,5])
    b = selected_class == 6
    other = ~(g | v | b)

    noise[g] = np.random.normal(0, ground_sigma, g.sum())
    noise[v] = np.random.normal(0, veg_sigma, v.sum())
    noise[b] = np.random.normal(0, building_sigma, b.sum())
    noise[other] = np.random.normal(0, ground_sigma, other.sum())

    xyz[idx, 2] += noise
    return xyz

def add_intensity_noise(intensity, idx, sigma_add=5.0, sigma_mult=0.05):
    intensity = intensity.copy()
    mult = np.random.normal(0, sigma_mult, size=len(idx))
    add = np.random.normal(0, sigma_add, size=len(idx))
    intensity[idx] = intensity[idx] * (1 + mult) + add
    return intensity

def add_outliers(
    xyz,
    classification=None,
    intensity=None,
    num_outliers=10000,
    sigma=2.0
):
    """
    Create outliers by copying random existing points and aggressively perturbing them.
    These new points are labelled as noise (class 18).
    """

    N = xyz.shape[0]
    idx = np.random.choice(N, num_outliers, replace=False)

    # Copy selected points
    outliers_xyz = xyz[idx].copy()

    # Apply large random displacement
    outliers_xyz += np.random.normal(0, sigma, size=outliers_xyz.shape)

    outlier_classes = np.full(num_outliers, NOISE_CLASS, dtype=int)

    if classification is not None and intensity is not None:
        outlier_intensity = intensity[idx].copy()

        xyz_combined = np.vstack([xyz, outliers_xyz])
        class_combined = np.hstack([classification, outlier_classes])
        intensity_combined = np.hstack([intensity, outlier_intensity])

        return xyz_combined, class_combined, intensity_combined

    elif classification is not None:
        xyz_combined = np.vstack([xyz, outliers_xyz])
        class_combined = np.hstack([classification, outlier_classes])
        return xyz_combined, class_combined

    return np.vstack([xyz, outliers_xyz]), outlier_classes

# ------------------ MASTER SIMULATION ------------------

def simulate_noise(
    xyz,
    intensity=None,
    classification=None,
    scan_angle=None,
    num_per_type=5000,
    num_outliers = 500,
    apply={
        "range": True,
        "horizontal": True,
        "scan_angle": True,
        "surface": True,
        "outliers": True,
        "intensity": True,
    },
    ):
        noisy_xyz = xyz.copy()
        noisy_class = classification.copy() if classification is not None else np.zeros(len(xyz), dtype=int)
        noisy_intensity = intensity.copy() if intensity is not None else None

        n_points = xyz.shape[0]
        all_noisy_indices = set()

        # --- RANGE NOISE ---
        if apply.get("range", False):
            idx = random_indices(n_points, num_per_type)
            noisy_xyz = add_range_noise(noisy_xyz, idx)
            all_noisy_indices.update(idx)

        # --- HORIZONTAL NOISE ---
        if apply.get("horizontal", False):
            idx = random_indices(n_points, num_per_type)
            noisy_xyz = add_horizontal_noise(noisy_xyz, idx)
            all_noisy_indices.update(idx)

        # --- SCAN ANGLE ---
        if apply.get("scan_angle", False) and scan_angle is not None:
            idx = random_indices(n_points, num_per_type)
            noisy_xyz = add_scan_angle_noise(noisy_xyz, scan_angle, idx)
            all_noisy_indices.update(idx)

        # --- SURFACE NOISE ---
        if apply.get("surface", False) and classification is not None:
            idx = random_indices(n_points, num_per_type)
            noisy_xyz = add_surface_noise(noisy_xyz, classification, idx)
            all_noisy_indices.update(idx)

        # --- INTENSITY NOISE ---
        if apply.get("intensity", False) and noisy_intensity is not None:
            idx = random_indices(n_points, num_per_type)
            noisy_intensity = add_intensity_noise(noisy_intensity, idx)
            all_noisy_indices.update(idx)

        # --- RELABEL ALL AFFECTED POINTS ---
        if classification is not None:
            noisy_class[list(all_noisy_indices)] = NOISE_CLASS

        # --- ADD OUTLIERS ---
        if apply.get("outliers", False):
            noisy_xyz, noisy_class, noisy_intensity = add_outliers(
                noisy_xyz,
                noisy_class,
                noisy_intensity,
                num_outliers=num_outliers,
                sigma=2.0  # increase for stronger noise
            )

        return noisy_xyz, noisy_class, noisy_intensity