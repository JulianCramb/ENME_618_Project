import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utilities import load_lidar, visualize, simulate_noise, visualize_anomalies, visualize_clusters
import pdb
import os

# model params
X_DIM = 4 
Z_DIM = 4 
K_RANGE = [48]
K_DEFAULT = 48 
LAMBDA_1 = 1.0 
LAMBDA_2 = 0.005
EPOCHS = 10 
BATCH_SIZE = 640
LEARNING_RATE = 1e-4

EPSILON = 1e-12
COV_EPS = 1e-6
ANOMALY_PERCENTILE = 95

# DLGMM

class EncoderNet(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(EncoderNet, self).__init__()
        # Encoder: x_dim -> 10 -> z_dim
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 10),
            nn.Tanh(),
            nn.Linear(10, z_dim) 
        )

    def forward(self, x):
        z_c = self.encoder(x)
        return z_c

class EstimationNet(nn.Module):
    def __init__(self, z_dim, k): 
        super(EstimationNet, self).__init__()
        # MLP: z_dim -> 10 -> k (number of Gaussian components)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, k)
        )

    def forward(self, z):
        p = self.mlp(z)
        gamma = torch.softmax(p, dim=1)
        return gamma
    

def calculate_gmm_parameters(z, gamma):

    N = z.size(0)
    D = z.size(1)
    K = gamma.size(1)

    # phi
    phi = torch.sum(gamma, dim=0) / N
    phi = phi / (torch.sum(phi) + EPSILON)

    # mu
    gamma_unsqueeze = gamma.unsqueeze(-1)
    mu = torch.sum(z.unsqueeze(1) * gamma_unsqueeze, dim=0)
    mu = mu / (torch.sum(gamma_unsqueeze, dim=0) + EPSILON)

    # sigma
    z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
    z_mu_outer = z_mu.unsqueeze(-1) @ z_mu.unsqueeze(-2)
    gamma_unsqueeze_outer = gamma.unsqueeze(-1).unsqueeze(-1)

    Sigma = torch.sum(z_mu_outer * gamma_unsqueeze_outer, dim=0)
    Sigma = Sigma / (torch.sum(gamma_unsqueeze, dim=0).unsqueeze(-1) + EPSILON)

    # regularization
    I = torch.eye(D, device=z.device).unsqueeze(0).repeat(K, 1, 1)
    Sigma = Sigma + I * COV_EPS

    return phi, mu, Sigma

def calculate_sample_energy(z, phi, mu, Sigma):
    N, D = z.size()
    K = phi.size(0)

    log_phi = torch.log(phi + EPSILON)

    # Reshape z to N x 1 x D and mu to 1 x K x D
    z_reshaped = z.unsqueeze(1)
    mu_reshaped = mu.unsqueeze(0)

    # Difference: z_mu shape N x K x D
    z_mu = z_reshaped - mu_reshaped

    # Inverse and log determinant of covariance matrices
    Sigma_inv = torch.inverse(Sigma)
    log_det_Sigma = torch.logdet(Sigma)

    # Calculate the Mahalanobis distance squared (d_k^2)
    d_squared = (z_mu.unsqueeze(-2) @ Sigma_inv.unsqueeze(0) @ z_mu.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    # Log-likelihood term (log N(z | mu_k, Sigma_k))
    log_N = -0.5 * D * np.log(2 * np.pi) - 0.5 * log_det_Sigma.unsqueeze(0) - 0.5 * d_squared

    # Log-sum-exp trick for E(z) = -log(sum_k (phi_k * N(z | mu_k, Sigma_k)))
    log_prob = log_phi.unsqueeze(0) + log_N
    
    # Calculate energy E(z_i) = -log(P(z_i))
    sample_energy = -torch.logsumexp(log_prob, dim=1)

    return sample_energy, Sigma

def dl_gmm_loss(z, phi, mu, Sigma):
    """
    The simplified DLGMM objective function.
    J = lambda_1 * L_energy + lambda_2 * L_penalty
    """
    # 1. GMM Energy Loss (L_energy)
    sample_energy, _ = calculate_sample_energy(z, phi, mu, Sigma) 
    energy_loss = torch.mean(sample_energy)

    # 2. Penalty Term (L_penalty) - Regularization to prevent singularities
    penalty_loss = torch.sum(1.0 / (torch.diagonal(Sigma, dim1=-2, dim2=-1) + EPSILON))

    # Combined Loss
    total_loss = LAMBDA_1 * energy_loss + LAMBDA_2 * penalty_loss
    
    reconstruction_loss = torch.tensor(0.0, device=z.device) 
    
    return total_loss, reconstruction_loss, energy_loss, penalty_loss


class DLGMM(nn.Module):
    """
    The complete Deep Latent Gaussian Mixture Model (Encoder + GMM).
    """
    def __init__(self, x_dim, z_dim, k):
        super(DLGMM, self).__init__()
        self.k = k
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.z_dim_concat = z_dim 

        self.encoder_net = EncoderNet(x_dim, z_dim)
        self.estimation_net = EstimationNet(self.z_dim_concat, k)

    def forward(self, x):
        z_c = self.encoder_net(x)
        z = z_c 
        x_hat = z_c 
        gamma = self.estimation_net(z)

        return z, x_hat, gamma

# --- 4. DATASET AND TRAINING SETUP ---

class PointCloudDataset(Dataset):
    """
    Custom Dataset for Point Cloud Data (XYZI).
    """
    def __init__(self, data_numpy_array):
        self.data = torch.from_numpy(data_numpy_array).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_dlgmm(model, data_loader, optimizer, full_data_tensor):
    """
    Training loop for the DLGMM model. This version calculates and prints 
    the average loss per epoch. It returns the trained model and final GMM parameters
    calculated over the *full* training tensor.
    """
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_total_loss = 0.0
        num_batches = 0
        
        for x in data_loader:
            x = x.float() 
            optimizer.zero_grad()

            z, _, gamma = model(x) 
            # Calculate GMM parameters on the batch for loss computation
            phi, mu, Sigma = calculate_gmm_parameters(z, gamma)

            total_loss, _, _, _ = dl_gmm_loss(z, phi, mu, Sigma)

            total_loss.backward()
            optimizer.step()
            
            # Accumulate loss for epoch average
            epoch_total_loss += total_loss.item()
            num_batches += 1
            
        avg_epoch_loss = epoch_total_loss / num_batches
        
        # Print the epoch's average loss
        print(f"Epoch: {epoch}, Average Loss: {avg_epoch_loss:.6f}")
            
    # After final training epoch, calculate final GMM parameters over the *entire* dataset
    model.eval()
    with torch.no_grad():
        # Get latent vectors (z) and soft assignments (gamma) for all data
        z_full, _, gamma_full = model(full_data_tensor)
        # Calculate GMM params using all z_full and gamma_full
        phi_final, mu_final, Sigma_final = calculate_gmm_parameters(z_full, gamma_full)
    
    return model, phi_final, mu_final, Sigma_final


def calculate_bic(N, K, Z_DIM, log_likelihood):
    """
    Calculates the Bayesian Information Criterion (BIC).
    BIC = k * ln(N) - 2 * ln(L_hat)
    """
    # Number of parameters in the GMM component (assuming fixed encoder structure)
    num_gmm_parameters = K * Z_DIM + K * (Z_DIM * (Z_DIM + 1) // 2) + K - 1
    
    bic = num_gmm_parameters * np.log(N) - 2 * log_likelihood.item()
    return bic

def predict_anomaly_scores(model, data_tensor, phi, mu, Sigma):
    """
    Computes the anomaly scores (Sample Energy) for the given data tensor
    using the final, trained GMM parameters.
    """
    model.eval()
    with torch.no_grad():
        # 1. Get latent features Z_c
        z, _, _ = model(data_tensor)
        # 2. Calculate the anomaly score (Sample Energy)
        scores, _ = calculate_sample_energy(z, phi, mu, Sigma)
        return scores.numpy()
    
def get_hard_cluster_assignments(model, data_tensor):
    """
    Computes the hard cluster assignment (index of component with max probability)
    for each sample.
    """
    model.eval()
    with torch.no_grad():
        # 1. Get latent features Z and soft assignments gamma
        z, _, gamma = model(data_tensor)
        # 2. Hard assignment is the index of the maximum probability in gamma
        # Output is 0-indexed (0 to K-1)
        cluster_indices = torch.argmax(gamma, dim=1)
        return cluster_indices.numpy()

# --- 5. MAIN EXECUTION ---

if __name__ == '__main__':
    # make output folder
    out_folder = "run1_urban"
    os.makedirs(out_folder, exist_ok=True)

    # 1. Data Setup for Point Cloud (XYZI)
    pc1 = 'Data/segmented_points_urban_w_noise.las'

    # load point cloud
    xyz, intensity, classification, scan_angle, header = load_lidar(pc1)

    # concatenate xyz, intensity
    cp_data = np.hstack((xyz, intensity.reshape(-1, 1) ))

    # --- Data Standardization ---
    # Center the data
    data_mean = np.mean(cp_data, axis=0)
    cp_data-= data_mean
    # Scale the data (Standardize)
    data_std = np.std(cp_data, axis=0)
    # Prevent division by zero if a feature is constant
    data_std[data_std == 0] = 1.0 
    cp_data /= data_std
    print("Data successfully standardized (zero mean, unit variance).")
    
    point_cloud_dataset = PointCloudDataset(cp_data)
    data_loader = DataLoader(
        point_cloud_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    best_k = K_DEFAULT
    min_bic = float('inf')
    
    # Convert entire dataset to a single tensor for final BIC calculation (log-likelihood)
    full_data_tensor = point_cloud_dataset.data 
    N_samples = len(full_data_tensor)

    # --- K-Value Search using BIC ---
    print("-" * 50)
    print(f"Starting Hyperparameter Search for optimal K: {K_RANGE}")
    print("-" * 50)

    bic_scores = []

    for current_k in K_RANGE:
        # 2. Model Initialization (for current K)
        model = DLGMM(x_dim=X_DIM, z_dim=Z_DIM, k=current_k) 
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 3. Training Loop and Final GMM Parameter Calculation
        model_k, phi_final, mu_final, Sigma_final = train_dlgmm(
            model, data_loader, optimizer, full_data_tensor
        )
        
        # 4. Calculate BIC on the full dataset
        with torch.no_grad():
            z_full, _, _ = model_k(full_data_tensor)
            sample_energy_full, _ = calculate_sample_energy(z_full, phi_final, mu_final, Sigma_final)
            total_log_likelihood = -torch.sum(sample_energy_full)
            current_bic = calculate_bic(N_samples, current_k, Z_DIM, total_log_likelihood)

            bic_scores.append(current_bic)
            
            # Update best K and save the best model's components
            if current_bic < min_bic:
                min_bic = current_bic
                best_k = current_k
                best_model_state = model_k.state_dict()
                best_phi = phi_final
                best_mu = mu_final
                best_Sigma = Sigma_final

            print(f"K = {current_k} | BIC Score: {current_bic:.4f}")

    print("\n" + "=" * 50)
    print(f"Optimal K found: {best_k}")
    print("=" * 50)

    # save bic results for plotting
    bic_results =  np.column_stack((K_RANGE, bic_scores))
    np.savetxt(out_folder + "/bic_scores.csv", bic_results, delimiter=",")

    # save best model
    torch.save(best_model_state, out_folder + "/model.pth")

    # --- Anomaly Labeling Process ---
    
    # 1. Load the Best Model and Parameters
    final_model = DLGMM(x_dim=X_DIM, z_dim=Z_DIM, k=best_k)
    final_model.load_state_dict(best_model_state)
    
    # 2. Calculate Threshold based on Training Data Scores
    # We use the full training data set to find the threshold.
    train_scores = predict_anomaly_scores(
        final_model, full_data_tensor, best_phi, best_mu, best_Sigma
    )

    # Determine the threshold: e.g., the 95th percentile of the training scores
    anomaly_threshold = np.percentile(train_scores, ANOMALY_PERCENTILE)

    print(f"\n--- Anomaly Labeling Summary ---")
    print(f"Threshold set at {ANOMALY_PERCENTILE}th percentile: {anomaly_threshold:.4f}")
    
    # PERFORM CLUSTERING
    # Calculate scores and binary classification
    test_scores = predict_anomaly_scores(
        final_model, full_data_tensor, best_phi, best_mu, best_Sigma
    )
    is_anomaly = test_scores > anomaly_threshold
    
    # Calculate hard cluster assignments (0-indexed) for all test points
    test_cluster_indices = get_hard_cluster_assignments(final_model, full_data_tensor)
    
    # Initialize final labels array
    final_labels = np.zeros_like(test_scores, dtype=int)
    
    # 4. Report Results
    print("\nClustering with Best Model:")
    print("-------------------------------------")
    
    for i in range(len(test_scores)):
        score = test_scores[i]
        is_a = is_anomaly[i]
        cluster_idx = test_cluster_indices[i]

        # Determine the final label
        if is_a:
            # -1 for Outlier/Anomaly
            final_label_int = -1
        else:
            # Cluster index is 0-indexed, so add 1 for user-friendly 1-indexed cluster labels
            final_label_int = cluster_idx + 1
        
        # Store the label
        final_labels[i] = final_label_int

    # accuracy assessment
    # EXPECTED
    anomaly_classification = np.zeros_like(test_scores, dtype=int)
    # low noise (7) and high noise (18)
    ind = (classification == 7) & (classification == 18)
    anomaly_classification[ind] = 1

    # PREDICTED
    anomaly_predicted = np.zeros_like(test_scores, dtype=int)
    ind = final_labels == -1
    anomaly_predicted[ind] = 1

    # plotting anomalies
    visualize_anomalies(xyz, anomaly_predicted, out_folder + "/predicted_anomaly_plot_best_model.png")

    # plotting clusters
    visualize_clusters(xyz, final_labels, out_folder + "/predicted_clusters_plot_best_model.png")

    # confusion matrix
    TP = np.sum((anomaly_predicted == 1) & (anomaly_classification == 1))
    TN = np.sum((anomaly_predicted == 0) & (anomaly_classification == 0))
    FP = np.sum((anomaly_predicted == 1) & (anomaly_classification == 0))
    FN = np.sum((anomaly_predicted == 0) & (anomaly_classification == 1))
    
    # 4. Calculate Metrics
    # Total samples
    total = len(test_scores)
    
    # Accuracy: (TP + TN) / Total
    accuracy = (TP + TN) / total if total > 0 else 0
    
    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall (Sensitivity): TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
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
