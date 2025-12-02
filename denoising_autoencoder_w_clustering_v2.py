import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from utilities import load_lidar, visualize, simulate_noise, visualize_anomalies, visualize_clusters 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pdb
import random
from io import StringIO

# model params
INPUT_DIM = 3
LATENT_DIM = 3     # no dimensionality reduction as small dataset already
EPOCHS = 200       
BATCH_SIZE_TILES = 4 # tiles per train batch
LEARNING_RATE = 1e-4
NOISE_FACTOR = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLUSTERS = 50
SEED = 42
CLUSTER_TYPE = "GMM"

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
            #nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed
    
# variational AE Test
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=4, latent_dim=4):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder: 4 -> 16 -> 8 -> (mu, log_var) for latent_dim
        self.encoder_fc1 = nn.Linear(input_dim, 16)
        self.encoder_fc2 = nn.Linear(16, 8)
        
        # Output layers for mu and log_var
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_log_var = nn.Linear(8, latent_dim)
        
        # Decoder: latent_dim -> 8 -> 16 -> 4
        self.decoder_fc1 = nn.Linear(latent_dim, 8)
        self.decoder_fc2 = nn.Linear(8, 16)
        self.decoder_fc3 = nn.Linear(16, input_dim)
        
        self.dropout = nn.Dropout(p=0.1)

    def encode(self, x):
        h1 = F.relu(self.encoder_fc1(x))
        h1 = self.dropout(h1) # Apply dropout
        h2 = F.relu(self.encoder_fc2(h1))
        
        mu = self.fc_mu(h2)
        log_var = self.fc_log_var(h2)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) 
        eps = torch.randn_like(std)   
        return mu + eps * std         

    def decode(self, z):
        h3 = F.relu(self.decoder_fc1(z))
        h4 = F.relu(self.decoder_fc2(h3))
        return self.decoder_fc3(h4) #torch.sigmoid(self.decoder_fc3(h4))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var


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
        return X_noisy #np.clip(X_noisy, 0., 1.)
        
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

    #model = AdvancedDenoisingAE(INPUT_DIM, LATENT_DIM).to(DEVICE)
    model = VariationalAutoencoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss(reduction='mean') # L2 loss for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #  vae loss
    def vae_loss(reconstruction, x, mu, log_var):

        # MSE Loss
        recon_loss = criterion(reconstruction, x)
        
        # KL Divergence Loss
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        kl_div /= x.size(0)

        # Total VAE loss
        return recon_loss + kl_div

    out_folder = "run_ae_full_forest"
    os.makedirs(out_folder, exist_ok=True)

    #pc1_clean = 'Data/segmented_points_urban.las'
    #pc1_clean = 'Data/segmented_points_forest.las'
    pc1_clean = 'Data/segmented_points_full_forest.las'
    xyz, intensity, classification, scan_angle, header = load_lidar(pc1_clean)
    
    # (1) spatial clustering according to xyz
    
    X_full_coords = xyz 

    print(f"Clustering {len(X_full_coords)} points into {N_CLUSTERS} spatial clusters...")

    if CLUSTER_TYPE == "KMeans":
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init='auto', verbose=0)
        cluster_labels = kmeans.fit_predict(X_full_coords)

    elif CLUSTER_TYPE == "GMM":
        scaler = StandardScaler()
        X_full_coords_scaled = scaler.fit_transform(X_full_coords)

        gmm = GaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type='full',
            random_state=SEED
        )
        gmm.fit(X_full_coords_scaled)
        cluster_labels = gmm.predict(X_full_coords_scaled)
    
    unique_cluster_ids = np.unique(cluster_labels)
    print(f"Successfully generated {len(unique_cluster_ids)} unique clusters.")

    # visualize the input clusters used for batches
    visualize_clusters(xyz, cluster_labels, out_folder + "/input_clusters.png")

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
    
    #cp_train = np.hstack((xyz_train_clean_raw, int_train_raw.reshape(-1, 1)))
    cp_train = xyz_train_clean_raw
    valid_ind_train = (class_train_raw != 7) & (class_train_raw != 18)
    cp_train = cp_train[valid_ind_train]
    
    #min_values = np.min(cp_train, axis=0)
    #max_values = np.max(cp_train, axis=0)
    
    #data_range = max_values - min_values
    #zero_range_mask = data_range == 0
    #denominator = np.where(zero_range_mask, 1e-8, data_range)
    #X_clean_train_norm = (cp_train - min_values) / denominator
    #X_clean_train_norm[:, zero_range_mask] = 0.0

    # standardized scaling
    mu_values = np.mean(cp_train, axis=0)
    sigma_values = np.std(cp_train, axis=0)
    zero_sigma_mask = sigma_values == 0
    denominator = np.where(zero_sigma_mask, 1.0, sigma_values)
    X_clean_train_norm = (cp_train - mu_values) / denominator
    X_clean_train_norm[:, zero_sigma_mask] = 0.0

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
    
    print("\n--- Training Cluster Informed Variational Autoencoder ---")
    total_points_trained = X_clean_train_norm.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for noisy_data, clean_target in train_loader:
            noisy_data = noisy_data.to(DEVICE)
            clean_target = clean_target.to(DEVICE)
            
            optimizer.zero_grad()
            #reconstructed_data = model(noisy_data)
            reconstructed_data, mu, log_var = model(noisy_data)

            #loss = criterion(reconstructed_data, clean_target)
            loss = vae_loss(reconstructed_data, clean_target, mu, log_var)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * noisy_data.size(0)
        
        epoch_loss = running_loss / total_points_trained
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}") 

    # training GMM on train dataset
    print(" Training GMM...")
    #gmm_train = gmm
    #gmm_train = GaussianMixture(n_components=N_CLUSTERS, covariance_type='full', random_state=SEED)
    #gmm_train.fit(X_clean_train_norm)

    # (6) anomaly reconstruction error threshold using validation dataset
    
    #cp_val = np.hstack((xyz_val_clean_raw, int_val_raw.reshape(-1, 1)))
    cp_val = xyz_val_clean_raw
    valid_ind_val = (class_val_raw != 7) & (class_val_raw != 18)
    cp_val = cp_val[valid_ind_val]
    
    #data_range = max_values - min_values
    #denominator = np.where(data_range == 0, 1e-8, data_range)
    #X_val_norm = (cp_val - min_values) / denominator
    #X_val_norm[:, zero_range_mask] = 0.0 

    X_val_norm = (cp_val - mu_values) / denominator
    X_val_norm[:, zero_sigma_mask] = 0.0

    X_target_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32).to(DEVICE)

    model.eval() 
    with torch.no_grad():
        #val_reconstructed = model(X_target_val_tensor)
        #val_error_per_point = torch.mean((X_target_val_tensor - val_reconstructed)**2, dim=1)

        val_reconstructed, val_mu, val_log_var = model(X_target_val_tensor)
        
        # Reconstruction error per point
        val_recon_error_per_point = torch.mean((X_target_val_tensor - val_reconstructed)**2, dim=1)
        
        val_kl_div_per_point = -0.5 * torch.sum(1 + val_log_var - val_mu.pow(2) - val_log_var.exp(), dim=1)
        
        # Total VAE anomaly score per point
        val_error_per_point = val_recon_error_per_point + val_kl_div_per_point

    val_error_np = val_error_per_point.cpu().numpy()

    THRESHOLD_PERCENTILE = 95
    anomaly_threshold = np.percentile(val_error_np, THRESHOLD_PERCENTILE)

    print(f"\n--- Anomaly Threshold Setting ---")
    print(f"Mean Validation Error: {np.mean(val_error_np):.6f}")
    print(f"VAE Anomaly Threshold ({THRESHOLD_PERCENTILE}th percentile): {anomaly_threshold:.6f}")

    # determining gmm threshold
    #val_log_likelihoods = gmm_train.score_samples(X_val_norm)
    val_log_likelihoods = gmm.score_samples(X_clean_train_norm)

    THRESHOLD_PERCENTILE = 5 
    log_likelihood_threshold = np.percentile(val_log_likelihoods, THRESHOLD_PERCENTILE)

    print(f"GMM Log-Likelihood Threshold ({THRESHOLD_PERCENTILE}th percentile): {log_likelihood_threshold:.4f}")


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

    # ensure anomalies are within the point cloud limits
    mask = (
        (xyz_noisy[:, 0] >= np.min(xyz[:,0])) & (xyz_noisy[:, 0] <= np.max(xyz[:,0])) &
        (xyz_noisy[:, 1] >= np.min(xyz[:,1])) & (xyz_noisy[:, 1] <= np.max(xyz[:,1]))
    )

    xyz_noisy = xyz_noisy[mask]
    classification_noisy = classification_noisy[mask]
    intensity_noisy = intensity_noisy[mask]

    # combine test features
    #cp_data_test = np.hstack((xyz_noisy, intensity_noisy.reshape(-1, 1) ))
    cp_data_test = xyz_noisy

    # normalize
    #data_range = max_values - min_values
    #denominator = np.where(data_range == 0, 1e-8, data_range)
    #X_test_data_norm = (cp_data_test - min_values) / denominator
    #X_test_data_norm[:, zero_range_mask] = 0.0 

    X_test_data_norm = (cp_data_test - mu_values) / denominator
    X_test_data_norm[:, zero_sigma_mask] = 0.0
    
    X_test_tensor = torch.tensor(X_test_data_norm, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        #test_reconstructed = model(X_test_tensor)
        #test_error_per_point = torch.mean((X_test_tensor - test_reconstructed)**2, dim=1)

        test_reconstructed, test_mu, test_log_var = model(X_test_tensor)
    
        # Reconstruction error per point for test set
        test_recon_error_per_point = torch.mean((X_test_tensor - test_reconstructed)**2, dim=1)
        
        # KL Divergence per point for test set
        test_kl_div_per_point = -0.5 * torch.sum(1 + test_log_var - test_mu.pow(2) - test_log_var.exp(), dim=1)
        
        # Total VAE anomaly score per point for test set
        test_error_per_point = test_recon_error_per_point + test_kl_div_per_point

    test_error_np = test_error_per_point.cpu().numpy()

    # flag anomalies
    is_anomaly = test_error_np > anomaly_threshold

    # gmm anomaly test
    #test_log_likelihoods = gmm_train.score_samples(X_test_data_norm)
    test_log_likelihoods = gmm.score_samples(X_test_data_norm)
    gmm_is_anomaly = test_log_likelihoods < log_likelihood_threshold

    # (8) accuracy assessment

    # EXPECTED: true Anomaly labels are 7 (low noise) OR 18 (high noise/outlier)
    anomaly_classification = np.zeros_like(is_anomaly, dtype=int)
    ind = (classification_noisy == 7) | (classification_noisy == 18)
    anomaly_classification[ind] = 1

    # PREDICTED
    anomaly_predicted = is_anomaly.astype(int)
    gmm_anomaly_predicted = gmm_is_anomaly.astype(int)

    # plotting anomalies

    xyz_all = np.concatenate((xyz_noisy, xyz_train_clean_raw, xyz_val_clean_raw), axis=0)
    expected_all = np.concatenate((anomaly_classification, np.zeros(xyz_train_clean_raw.shape[0]), np.zeros(xyz_val_clean_raw.shape[0])), axis=0)
    predicted_all = np.concatenate((anomaly_predicted, np.zeros(xyz_train_clean_raw.shape[0]), np.zeros(xyz_val_clean_raw.shape[0])), axis=0)
    gmm_predicted_all = np.concatenate((gmm_anomaly_predicted, np.zeros(xyz_train_clean_raw.shape[0]), np.zeros(xyz_val_clean_raw.shape[0])), axis=0)
    visualize_anomalies(xyz_all, predicted_all, out_folder + "/predicted_anomaly_plot.png")
    visualize_anomalies(xyz_all, expected_all, out_folder + "/expected_anomaly_plot.png")
    visualize_anomalies(xyz_all, gmm_predicted_all, out_folder + "/gmm_predicted_anomaly_plot.png")

output_buffer = StringIO()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~VAE ACCURACY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# confusion matrix
TP_vae = np.sum((anomaly_predicted == 1) & (anomaly_classification == 1))
TN_vae = np.sum((anomaly_predicted == 0) & (anomaly_classification == 0))
FP_vae = np.sum((anomaly_predicted == 1) & (anomaly_classification == 0))
FN_vae = np.sum((anomaly_predicted == 0) & (anomaly_classification == 1))

# accuracy metrics
total_vae = len(anomaly_predicted)
accuracy_vae = (TP_vae + TN_vae) / total_vae if total_vae > 0 else 0
precision_vae = TP_vae / (TP_vae + FP_vae) if (TP_vae + FP_vae) > 0 else 0
recall_vae = TP_vae / (TP_vae + FN_vae) if (TP_vae + FN_vae) > 0 else 0
f1_score_vae = 2 * (precision_vae * recall_vae) / (precision_vae + recall_vae) if (precision_vae + recall_vae) > 0 else 0

output_buffer.write("\n" + "=" * 50 + "\n")
output_buffer.write("VAE Anomaly Detection Performance Metrics:\n")
output_buffer.write("=" * 50 + "\n")
output_buffer.write(f"Total Samples: {total_vae}\n")
output_buffer.write(f"True Positives (TP): {TP_vae}\n")
output_buffer.write(f"True Negatives (TN): {TN_vae}\n")
output_buffer.write(f"False Positives (FP): {FP_vae}\n")
output_buffer.write(f"False Negatives (FN): {FN_vae}\n")
output_buffer.write("-" * 50 + "\n")
output_buffer.write(f"Accuracy:  {accuracy_vae:.4f}\n")
output_buffer.write(f"Precision: {precision_vae:.4f}\n")
output_buffer.write(f"Recall:    {recall_vae:.4f}\n")
output_buffer.write(f"F1-Score:  {f1_score_vae:.4f}\n")
output_buffer.write("=" * 50 + "\n")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GMM ACCURACY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# confusion matrix
TP_gmm = np.sum((gmm_anomaly_predicted == 1) & (anomaly_classification == 1))
TN_gmm = np.sum((gmm_anomaly_predicted == 0) & (anomaly_classification == 0))
FP_gmm = np.sum((gmm_anomaly_predicted == 1) & (anomaly_classification == 0))
FN_gmm = np.sum((gmm_anomaly_predicted == 0) & (anomaly_classification == 1))

# accuracy metrics
total_gmm = len(gmm_anomaly_predicted)
accuracy_gmm = (TP_gmm + TN_gmm) / total_gmm if total_gmm > 0 else 0
precision_gmm = TP_gmm / (TP_gmm + FP_gmm) if (TP_gmm + FP_gmm) > 0 else 0
recall_gmm = TP_gmm / (TP_gmm + FN_gmm) if (TP_gmm + FN_gmm) > 0 else 0
f1_score_gmm = 2 * (precision_gmm * recall_gmm) / (precision_gmm + recall_gmm) if (precision_gmm + recall_gmm) > 0 else 0

output_buffer.write("\n" + "=" * 50 + "\n")
output_buffer.write("GMM Anomaly Detection Performance Metrics:\n")
output_buffer.write("=" * 50 + "\n")
output_buffer.write(f"Total Samples: {total_gmm}\n")
output_buffer.write(f"True Positives (TP): {TP_gmm}\n")
output_buffer.write(f"True Negatives (TN): {TN_gmm}\n")
output_buffer.write(f"False Positives (FP): {FP_gmm}\n")
output_buffer.write(f"False Negatives (FN): {FN_gmm}\n")
output_buffer.write("-" * 50 + "\n")
output_buffer.write(f"Accuracy:  {accuracy_gmm:.4f}\n")
output_buffer.write(f"Precision: {precision_gmm:.4f}\n")
output_buffer.write(f"Recall:    {recall_gmm:.4f}\n")
output_buffer.write(f"F1-Score:  {f1_score_gmm:.4f}\n")
output_buffer.write("=" * 50 + "\n")

full_output = output_buffer.getvalue()
print(full_output)

output_filename = out_folder + "/accuracy_metrics.txt"
try:
    with open(output_filename, 'w') as f:
        f.write(full_output)
    print(f"\nSuccessfully wrote metrics to file: {os.path.abspath(output_filename)}")
except IOError:
    print(f"\nError: Could not write metrics to file {output_filename}")