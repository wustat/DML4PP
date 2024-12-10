import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from dml_utils.dgp import GD
from dml_utils.dml import DML, Estimate, Bootstrap
from dml_utils import dgp, dml
from joblib import Parallel, delayed
import time
import sys


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')

def run_dml_pipeline(beta, n=5000, p=20, Sigma=None, K_main=5, K_inner=5, B=500, binary_A=True, device='cuda',
                    beta_init=1.0, lr=1e-3, epochs=600, random_state=42, linear=False,
                    early_stopping_patience=50, early_stopping_delta=1e-4, 
                    early_stopping_verbose=False, use_linear_models=False):
    """
    Run the complete DML pipeline: Data Generation, Nuisance Parameter Estimation, and Causal Estimation.

    Parameters:
    - beta (float): Coefficient for A in the outcome model.
    - n (int): Number of samples.
    - p (int): Number of covariates/features.
    - Sigma (torch.Tensor or None): Covariance matrix of X.
    - K_main (int): Number of main folds for cross-fitting.
    - K_inner (int): Number of inner folds for Wp cross-fitting.
    - binary_A (bool): If True, treat A as binary.
    - device (str): Device to use ('cuda' or 'cpu').
    - beta_init (float): Initial guess for beta0.
    - lr (float): Learning rate for beta estimation.
    - epochs (int): Number of epochs for beta estimation.
    - random_state (int or None): Seed for reproducibility.
    - early_stopping_patience (int): Patience for Early Stopping.
    - early_stopping_delta (float): Minimum improvement for Early Stopping.
    - early_stopping_verbose (bool): Verbosity for Early Stopping.
    - use_linear_models (bool): If True, use linear models for nuisance parameters.

    Returns:
    - beta_estimated (float): Estimated causal parameter beta0.
    """
    # Generate Data
    print(f"Generating synthetic data with seed {random_state}...")
    X, Y, A = GD(beta=beta, n=n, p=p, Sigma=Sigma, binary_A=binary_A, random_state=random_state, linear=linear)

    # Estimate nuisance parameters with nested K-fold cross-fitting
    print(f"\nEstimating nuisance parameters with nested K-fold cross-fitting for {random_state}...")
    dml_results = DML(Y, A, X, K=K_main, binary_A=binary_A, device=device, epoches=epochs,
                     early_stopping_patience=early_stopping_patience, 
                     early_stopping_delta=early_stopping_delta, 
                     early_stopping_verbose=early_stopping_verbose)

    # Estimate Causal Parameter
    print(f"\nEstimating causal parameter beta0 for {random_state}...")
    beta_estimated = Estimate(Y, A, dml_results, beta_init=beta_init, 
                              lr=lr, epochs=epochs, device=device)
    print(f"\nTrue beta: {beta}, Estimated beta: {beta_estimated} for {random_state}")
    
    print(f"\nRunning bootstrap for {random_state}...")
    bootstrap_res = Bootstrap(Y, A, dml_results, B=B)
    
    res = {'true_beta': beta, 'estimated_beta': beta_estimated, 'bootstrap_res': bootstrap_res, 'beta_init': beta_init, 'lr': lr, 'epochs': epochs, 'B': B, 'K_main': K_main, 'n': n, 'binary_A': binary_A, 'random_state': random_state}
    with open(f'results_{random_state}.pkl', 'wb') as f:
        pickle.dump(res, f)
    print(f"Results saved to results_{random_state}.pkl")
    return None


def run_simulation_partial(sim_arg):
    return run_dml_pipeline(**sim_arg)

def main(start_id, num_simulations):
    now = time.time()
    print(f"Running DML simulation at time {now}")
    #num_simulations = 100
    num_gpus = torch.cuda.device_count()
    pool_size = num_gpus
    print(f"Number of GPUs: {num_gpus}")
    simulation_args = []
    for sim_id in range(start_id, start_id+num_simulations):
        gpu_id = sim_id % num_gpus
        np.random.seed(sim_id)
        beta = np.random.uniform(-2, 2)
        simulation_args.append({'beta': beta, 'random_state': sim_id, 'device': f'cuda:{gpu_id}'})
    
    #results = Parallel(n_jobs=pool_size)(delayed(run_simulation_partial)(sim_arg) for sim_arg in simulation_args)
    for i in simulation_args:
        run_simulation_partial(i)
    
    # Save results to pickle
    # with open('results.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    # print("Results saved to results.pkl")
    print(f"Time taken: {time.time() - now} seconds")
    
if __name__ == '__main__':
    # input: start_id, num_simulations
    start = int(sys.argv[1])
    num = int(sys.argv[2])
    main(start, num)    
    
    