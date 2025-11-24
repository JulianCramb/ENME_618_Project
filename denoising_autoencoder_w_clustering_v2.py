import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from utilities import load_lidar, visualize, simulate_noise, visualize_anomalies, visualize_clusters 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
import pdb
import random

# model params
INPUT_DIM = 4
LATENT_DIM = 4     # no dimensionality reduction as small dataset already
EPOCHS = 200        
BATCH_SIZE_TILES = 4 # tiles per train batch
LEARNING_RATE = 1e-4
NOISE_FACTOR = 0.2 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_PERCENT = 0.1
N_CLUSTERS = 400
SEED = 42

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

# function to get points in each cluster
def get_point_indices_by_cluster_ids(cluster_labels, target_ids):
    mask = np.isin(cluster_labels, target_ids)
    return np.where(mask)[0]

# function to map to clusters
def map_clusters_to_indices(cluster_labels, unique_cluster_ids):
    tile_indices_map = []
    for cluster_id in unique_cluster_ids:
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) > 0:
            tile_indices_map.append(indices)
    return tile_indices_map

# initial denoising autoencoder (not deep enough)
class DenoisingAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DenoisingAE, self).__init__()
        
        # Encoder: 4 -> 3 -> 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: 2 -> 3 -> 4
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# new AE with more hidden dims
class AdvancedDenoisingAE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=4):
        super(AdvancedDenoisingAE, self).__init__()
        
        # Encoder: 4 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.1), 
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: 4 -> 8 -> 16 -> 4
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


# cluster dataset loader
class TileDataset(Dataset):
    def __init__(self, X_clean_all_norm, tile_indices_map, noise_factor):
        
        self.X_clean_norm = X_clean_all_norm
        self.tile_indices_map = tile_indices_map
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.tile_indices_map)

    def _add_gaussian_noise(self, data):
        """Adds Gaussian noise to the data for the DAE input."""
        X_noisy = data + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return np.clip(X_noisy, 0., 1.)
        
    def __getitem__(self, idx):
        point_indices = self.tile_indices_map[idx]
        
        X_clean_tile_norm = self.X_clean_norm[point_indices]
        
        X_noisy_norm = self._add_gaussian_noise(X_clean_tile_norm)

        X_noisy_tensor = torch.tensor(X_noisy_norm, dtype=torch.float32)
        X_clean_target_tensor = torch.tensor(X_clean_tile_norm, dtype=torch.float32)
        
        return X_noisy_tensor, X_clean_target_tensor

def tile_collate_fn(batch):
    noisy_tensors = [item[0] for item in batch]
    clean_tensors = [item[1] for item in batch]
    
    batch_noisy = torch.cat(noisy_tensors, dim=0)
    batch_clean = torch.cat(clean_tensors, dim=0)
    
    return batch_noisy, batch_clean

# main function
if __name__ == '__main__':

    model = AdvancedDenoisingAE(INPUT_DIM, LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss(reduction='mean') # L2 loss for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    out_folder = "run_ae_forest"
    os.makedirs(out_folder, exist_ok=True)

    #pc1_clean = 'Data/300-5050_2015.las'
    pc1_clean = 'Data/segmented_points_forest.las'
    xyz, intensity, classification, scan_angle, header = load_lidar(pc1_clean)
    
    # (1) spatial clustering according to xyz
    
    X_full_coords = xyz 

    print(f"Clustering {len(X_full_coords)} points into {N_CLUSTERS} spatial clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init='auto', verbose=0)
    cluster_labels = kmeans.fit_predict(X_full_coords)
    unique_cluster_ids = np.unique(cluster_labels)
    print(f"Successfully generated {len(unique_cluster_ids)} unique clusters.")

    train_ids, temp_ids, _, _ = train_test_split(unique_cluster_ids, unique_cluster_ids, test_size=0.4, random_state=SEED)
    val_ids, test_ids, _, _ = train_test_split(temp_ids, temp_ids, test_size=0.5, random_state=SEED)
    
    train_indices = get_point_indices_by_cluster_ids(cluster_labels, train_ids)
    val_indices = get_point_indices_by_cluster_ids(cluster_labels, val_ids)
    test_indices = get_point_indices_by_cluster_ids(cluster_labels, test_ids)
    
    # (2) extracting datasets for train (60%), validation (20%) and test (20%)
    
    def slice_data(indices):
        return (
            xyz[indices],
            intensity[indices],
            classification[indices]
        )

    xyz_train_clean_raw, int_train_raw, class_train_raw = slice_data(train_indices)
    xyz_val_clean_raw, int_val_raw, class_val_raw = slice_data(val_indices)
    xyz_test_clean_raw, int_test_raw, class_test_raw = slice_data(test_indices)
    
    print("Number of training points: " + str(xyz_train_clean_raw.shape[0]))
    print("Number of validation points: " + str(xyz_val_clean_raw.shape[0]))
    print("Number of testing points: " + str(xyz_test_clean_raw.shape[0]))

    # (3) Normalization
    
    cp_train = np.hstack((xyz_train_clean_raw, int_train_raw.reshape(-1, 1)))
    valid_ind_train = (class_train_raw != 7) & (class_train_raw != 18)
    cp_train = cp_train[valid_ind_train]
    
    min_values = np.min(cp_train, axis=0)
    max_values = np.max(cp_train, axis=0)
    
    data_range = max_values - min_values
    zero_range_mask = data_range == 0
    denominator = np.where(zero_range_mask, 1e-8, data_range)
    X_clean_train_norm = (cp_train - min_values) / denominator
    X_clean_train_norm[:, zero_range_mask] = 0.0

    # (4) prepare batches for training
    original_train_indices = np.where(valid_ind_train)[0]
    train_cluster_labels = cluster_labels[train_indices][valid_ind_train]
    unique_train_cluster_ids = np.unique(train_cluster_labels)
    filtered_coords = xyz_train_clean_raw[valid_ind_train]
    filtered_intensity = int_train_raw[valid_ind_train]
    
    tile_indices_map_dict = {id_: [] for id_ in unique_train_cluster_ids}
    
    for i, cluster_id in enumerate(train_cluster_labels):
        tile_indices_map_dict[cluster_id].append(i)
        
    tile_indices_map = [np.array(v) for v in tile_indices_map_dict.values() if v]

    train_dataset = TileDataset(
        X_clean_train_norm,
        tile_indices_map,
        NOISE_FACTOR
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE_TILES, 
        shuffle=True,
        collate_fn=tile_collate_fn
    )

    print(f"Total training clusters (tiles): {len(train_dataset)}")
    
    # (5) training loop
    
    print("\n--- Training Denoising Autoencoder ---")
    total_points_trained = X_clean_train_norm.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for noisy_data, clean_target in train_loader:
            noisy_data = noisy_data.to(DEVICE)
            clean_target = clean_target.to(DEVICE)
            
            optimizer.zero_grad()
            reconstructed_data = model(noisy_data)
            loss = criterion(reconstructed_data, clean_target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * noisy_data.size(0)
        
        epoch_loss = running_loss / total_points_trained
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}") 

    # (6) anomaly reconstruction error threshold using validation dataset
    
    cp_val = np.hstack((xyz_val_clean_raw, int_val_raw.reshape(-1, 1)))
    valid_ind_val = (class_val_raw != 7) & (class_val_raw != 18)
    cp_val = cp_val[valid_ind_val]
    
    data_range = max_values - min_values
    denominator = np.where(data_range == 0, 1e-8, data_range)
    X_val_norm = (cp_val - min_values) / denominator
    X_val_norm[:, zero_range_mask] = 0.0 

    X_target_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32).to(DEVICE)

    model.eval() 
    with torch.no_grad():
        val_reconstructed = model(X_target_val_tensor)
        val_error_per_point = torch.mean((X_target_val_tensor - val_reconstructed)**2, dim=1)

    val_error_np = val_error_per_point.cpu().numpy()

    THRESHOLD_PERCENTILE = 99
    anomaly_threshold = np.percentile(val_error_np, THRESHOLD_PERCENTILE)

    print(f"\n--- Anomaly Threshold Setting ---")
    print(f"Mean Validation Error: {np.mean(val_error_np):.6f}")
    print(f"Anomaly Threshold ({THRESHOLD_PERCENTILE}th percentile): {anomaly_threshold:.6f}")

    # (7) Anomaly detection on test set

    # add outliers to test dataset
    xyz_noisy, classification_noisy, intensity_noisy = simulate_noise(
        xyz_test_clean_raw,
        intensity=int_test_raw,
        classification=class_test_raw,
        scan_angle=None,
        num_per_type=0,
        num_outliers = 50000,
        apply={
            "range": True,
            "horizontal": True,
            "scan_angle": True,
            "surface": True,
            "outliers": True,
            "intensity": True,
        }
    )

    # combine test features
    cp_data_test = np.hstack((xyz_noisy, intensity_noisy.reshape(-1, 1) ))

    # normalize
    data_range = max_values - min_values
    denominator = np.where(data_range == 0, 1e-8, data_range)
    X_test_data_norm = (cp_data_test - min_values) / denominator
    X_test_data_norm[:, zero_range_mask] = 0.0 
    
    X_test_tensor = torch.tensor(X_test_data_norm, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        test_reconstructed = model(X_test_tensor)
        test_error_per_point = torch.mean((X_test_tensor - test_reconstructed)**2, dim=1)

    test_error_np = test_error_per_point.cpu().numpy()

    # flag anomalies
    is_anomaly = test_error_np > anomaly_threshold

    # (8) accuracy assessment

    # EXPECTED: true Anomaly labels are 7 (low noise) OR 18 (high noise/outlier)
    anomaly_classification = np.zeros_like(is_anomaly, dtype=int)
    ind = (classification_noisy == 7) | (classification_noisy == 18) # <--- **FIXED:** Changed & to |
    anomaly_classification[ind] = 1

    # PREDICTED
    anomaly_predicted = is_anomaly.astype(int)

    # plotting anomalies
    visualize_anomalies(xyz_noisy, anomaly_predicted, out_folder + "/predicted_anomaly_plot_best_model.png")

    # confusion matrix
    TP = np.sum((anomaly_predicted == 1) & (anomaly_classification == 1))
    TN = np.sum((anomaly_predicted == 0) & (anomaly_classification == 0))
    FP = np.sum((anomaly_predicted == 1) & (anomaly_classification == 0))
    FN = np.sum((anomaly_predicted == 0) & (anomaly_classification == 1))
    
    # accuracy metrics
    total = len(anomaly_predicted)
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 50)
    print("Anomaly Detection Performance Metrics:")
    print("=" * 50)
    print(f"Total Samples: {total}")
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print("-" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("=" * 50)