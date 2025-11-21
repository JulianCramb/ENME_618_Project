import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utilities import load_lidar, visualize, simulate_noise
import pdb

# --- 1. CONFIGURATION CONSTANTS ---
# X_DIM: 4 (X, Y, Z, Intensity)
X_DIM = 4           # Input feature dimension
Z_DIM = 4           # Dimension of the latent space (Z_c)
# K is now defined as a range to test in the main execution block
K_RANGE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] # Range of Gaussian components to test
K_DEFAULT = 40      # Default K, used for module initialization if not training loop
LAMBDA_1 = 1.0      # Weight for the GMM Energy term
LAMBDA_2 = 0.005    # Weight for the penalty term
EPOCHS = 20         # Reduced epochs for faster iterative selection
BATCH_SIZE = 640    # Batch size
LEARNING_RATE = 1e-4

# Stability constants
EPSILON = 1e-12     # General small constant for stability
COV_EPS = 1e-6      # Small constant added to covariance diagonal for stability
ANOMALY_PERCENTILE = 95 # Threshold percentile for anomaly score (e.g., top 5% are anomalies)

# --- 2. DLGMM Components ---

class EncoderNet(nn.Module):
    """
    The Encoder Network (replaces CompressionNet)
    Learns a latent representation Z_c directly from the input x.
    """
    def __init__(self, x_dim, z_dim):
        super(EncoderNet, self).__init__()
        # Encoder: x_dim -> 10 -> z_dim
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 10),
            nn.Tanh(),
            nn.Linear(10, z_dim) # Latent representation z_c
        )

    def forward(self, x):
        z_c = self.encoder(x)
        return z_c

class EstimationNet(nn.Module):
    """
    The Estimation Network (MLP).
    Predicts the soft-cluster assignment (gamma) based on the latent vector z_c.
    """
    def __init__(self, z_dim, k): # Now uses z_dim directly
        super(EstimationNet, self).__init__()
        # MLP: z_dim -> 10 -> k (number of Gaussian components)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, k) # Output raw scores p
        )

    def forward(self, z):
        # Output p is the raw score before softmax
        p = self.mlp(z)
        # Apply softmax to get soft assignments gamma
        gamma = torch.softmax(p, dim=1)
        return gamma

# --- 3. GMM Helper Functions ---

def calculate_gmm_parameters(z, gamma):
    """
    Computes the GMM parameters (phi, mu, Sigma) from the combined feature vector z
    and the soft assignments gamma, for the current batch/dataset.
    """
    N = z.size(0) # Batch size or total samples
    D = z.size(1) # Feature dimension (Z_DIM)
    K = gamma.size(1) # Number of components

    # 1. Calculate mixture weights (phi)
    phi = torch.sum(gamma, dim=0) / N
    phi = phi / (torch.sum(phi) + EPSILON)

    # 2. Calculate means (mu)
    gamma_unsqueeze = gamma.unsqueeze(-1)
    mu = torch.sum(z.unsqueeze(1) * gamma_unsqueeze, dim=0)
    mu = mu / (torch.sum(gamma_unsqueeze, dim=0) + EPSILON)

    # 3. Calculate covariance matrices (Sigma)
    z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
    z_mu_outer = z_mu.unsqueeze(-1) @ z_mu.unsqueeze(-2)
    gamma_unsqueeze_outer = gamma.unsqueeze(-1).unsqueeze(-1)

    Sigma = torch.sum(z_mu_outer * gamma_unsqueeze_outer, dim=0)
    Sigma = Sigma / (torch.sum(gamma_unsqueeze, dim=0).unsqueeze(-1) + EPSILON)

    # Add regularization for stability (COV_EPS * Identity matrix)
    I = torch.eye(D, device=z.device).unsqueeze(0).repeat(K, 1, 1)
    Sigma = Sigma + I * COV_EPS

    return phi, mu, Sigma

def calculate_sample_energy(z, phi, mu, Sigma):
    """
    Calculates the energy (negative log-likelihood) of each sample z
    under the estimated GMM components. This is the anomaly score.
    """
    # N: batch size, D: feature dim, K: components
    N, D = z.size()
    K = phi.size(0)

    # Calculate log(pi_k) (log-mixture weights)
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

# --- 5. MAIN EXECUTION ---

if __name__ == '__main__':
    # 1. Data Setup for Point Cloud (XYZI)
    pc1 = 'data_mabel_test/pc_072E15NE25NE_20171009.copc.laz' # farm land near medicine hat

    # load point cloud
    xyz, intensity, classification, scan_angle, header = load_lidar(pc1)

    # concatenate xyz, intensity
    cp_data = np.hstack((xyz, intensity.reshape(-1, 1) ))

    # grab subset of data (10%)
    np.random.seed(32)
    N = cp_data.shape[0]
    sample_size = int(0.1 * N)
    indices = np.random.choice(N, sample_size, replace=False)
    cp_data_subset = cp_data[indices]

    # --- Data Standardization Added Here ---
    # Center the data
    data_mean = np.mean(cp_data_subset, axis=0)
    cp_data_subset -= data_mean
    # Scale the data (Standardize)
    data_std = np.std(cp_data_subset, axis=0)
    # Prevent division by zero if a feature is constant
    data_std[data_std == 0] = 1.0 
    cp_data_subset /= data_std
    print("Data successfully standardized (zero mean, unit variance).")
    
    point_cloud_dataset = PointCloudDataset(cp_data_subset)
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
    
    # 3. Apply Classification to a New Test Set
    # Generate a small new test set for demonstration
    # This test set will contain known normal and known anomalous points
    test_normal = np.random.normal(loc=[10, 10, 5, 0.5], scale=1.0, size=(10, 4))
    test_anomaly = np.random.uniform(low=[100, 100, 100, 5.0], high=[200, 200, 200, 10.0], size=(5, 4))
    
    test_data_np = np.concatenate([test_normal, test_anomaly], axis=0)

    # Apply the same standardization (using the training data's mean and std)
    test_data_np = (test_data_np - data_mean) / data_std

    test_data_tensor = torch.from_numpy(test_data_np).float()
    
    test_scores = predict_anomaly_scores(
        final_model, test_data_tensor, best_phi, best_mu, best_Sigma
    )

    # Classify points
    is_anomaly = test_scores > anomaly_threshold
    
    # 4. Report Results
    print("\nTest Data Classification:")
    for i, (score, is_a) in enumerate(zip(test_scores, is_anomaly)):
        label = "ANOMALY" if is_a else "NORMAL"
        # The first 10 points are simulated normal, the last 5 are simulated anomalies
        true_label = "Anomaly (True)" if i >= 10 else "Normal (True)"
        
        print(f"Sample {i+1:02d} | Score: {score:.4f} | Prediction: {label} ({true_label})")