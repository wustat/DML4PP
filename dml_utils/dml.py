import torch
import torch.nn as nn
import torch.optim as optim
from dml_utils.nuisance import NuisanceModelY, NuisanceModelA, NuisanceModelAtemp, NuisanceModelWp, NuisanceModeltnk, NuisanceModelAY0, train_nuisance_model, logit, EarlyStopping
import numpy as np
from scipy.optimize import root_scalar
import wandb

def Split(K, n_samples, random_state=None, device='cpu'):
    """
    Randomly split the n_samples into K folds.

    Parameters:
    - K (int): Number of folds.
    - n_samples (int): Total number of samples.
    - random_state (int, optional): Seed for reproducibility. Defaults to None.
    - device (str, optional): Device to perform computations on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
    - I1 (torch.Tensor): Tensor of shape (n_samples,) containing fold indices from 1 to K.
    """
    # Initialize a random number generator for reproducibility if a seed is provided
    if random_state is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(random_state)
    else:
        generator = None  # Use the default generator

    # Initialize a tensor to hold fold assignments, defaulting to 0
    I1 = torch.zeros(n_samples, dtype=torch.long, device=device)

    # Generate a random permutation of indices from 0 to n_samples-1
    Newids = torch.randperm(n_samples, generator=generator, device=device)

    # Determine the size of each fold
    m = n_samples // K

    # Assign fold indices
    for i in range(K):
        start = i * m
        # Ensure that the last fold includes any remaining samples due to integer division
        end = (i + 1) * m if i != K - 1 else n_samples
        fold_ids = Newids[start:end]
        I1[fold_ids] = i + 1  # Folds are 1-indexed

    return I1

def DML(Y, A, X, K=5, epoches = 200, binary_A=True, device='cuda',
        early_stopping_patience=10, early_stopping_delta=1e-4, 
        early_stopping_verbose=False):
    """
    Estimate nuisance parameters using Double Machine Learning with cross-fitting and multiple models for Wp and E[A|X].

    Parameters:
    - Y (torch.Tensor): Outcome variable, shape (n_samples,)
    - A (torch.Tensor): Treatment variable, shape (n_samples,)
    - X (torch.Tensor): Covariates, shape (n_samples, p)
    - K (int): Number of folds for cross-fitting
    - binary_A (bool): If True, treat A as binary
    - device (str): Device to use ('cuda' or 'cpu')
    - early_stopping_patience (int): Patience for Early Stopping
    - early_stopping_delta (float): Minimum change in the monitored quantity
    - early_stopping_verbose (bool): Verbosity for Early Stopping
    - use_linear_models (bool): If True, use linear models for nuisance parameters

    Returns:
    - dml_results (dict): Dictionary containing estimated nuisance parameters and trained models
        - 'mXp': torch.Tensor, E[A | Y=0, X] predictions
        - 'rXp': torch.Tensor, Wp predictions (logit(E[Y | A, X]))
        - 'ap': torch.Tensor, E[A | X] predictions
        - 'model_AY0': Trained model for E[A | Y=0, X]
    """
    n_samples, p = X.shape
    # check A dim is 1 or others
    if len(A.shape) == 1:
        A_dim = 1
    else:
        A_dim = A.shape[1]
    mXp = torch.zeros(n_samples, device=device)
    rXp = torch.zeros(n_samples, device=device)
    ap = torch.zeros(n_samples, device=device)
    

    # Split data into K folds
    fold_indices = Split(K, n_samples, random_state=42).cpu().numpy()

    # For each main fold k (validation fold)
    for k in range(1, K + 1):
        #print(f"\nMain Fold {k}/{K}:")
        
        # Validation indices for fold k
        val_idx_k = np.where(fold_indices == k)[0]
        # Training indices (excluding fold k)
        train_idx_k = np.where(fold_indices != k)[0]

        X_val_k = X[val_idx_k].to(device)
        A_val_k = A[val_idx_k].to(device)
        Y_val_k = Y[val_idx_k].to(device)

        X_train_k = X[train_idx_k].to(device)
        A_train_k = A[train_idx_k].to(device)
        Y_train_k = Y[train_idx_k].to(device)

        # Initialize lists to store predictions from models
        Wp_preds = torch.zeros(len(train_idx_k), device=device)
        ap_preds = torch.zeros(len(train_idx_k), device=device)
        aBar = []

        fold_inner = Split(K, len(train_idx_k), random_state=42).cpu().numpy()
        # For each model i in 1 to K (excluding k) INNER LOOP
        for i in range(1, K + 1):

            #print(f"Looping through inner fold {i}/outer fold {K}...")

            # Training indices for model i (exclude folds k and i)
            train_idx_i = np.where(fold_inner != i)[0]
            
            X_train_i = X_train_k[train_idx_i].to(device)
            A_train_i = A_train_k[train_idx_i].to(device)
            Y_train_i = Y_train_k[train_idx_i].to(device)
            
            val_idx_i = np.where(fold_inner == i)[0]
            X_val_i = X_train_k[val_idx_i].to(device)
            A_val_i = A_train_k[val_idx_i].to(device)
            Y_val_i = Y_train_k[val_idx_i].to(device)

            # Initialize models for Wp and E[A|X]
           
            model_Wp = NuisanceModelY(input_dim=p + A_dim).to(device)
            model_Atemp = NuisanceModelAtemp(input_dim=p, binary_A=binary_A).to(device)

            optimizer_Wp = optim.Adam(model_Wp.parameters(), lr=1e-3)
            criterion_Wp = nn.BCEWithLogitsLoss()

            optimizer_A = optim.Adam(model_Atemp.parameters(), lr=1e-3)
            criterion_A = nn.MSELoss() if not binary_A else nn.BCELoss()

            # Train model_Wp on training data excluding folds k and i
            train_nuisance_model(model_Wp, optimizer_Wp, criterion_Wp,
                                 A_train_i, Y_train_i, X_train_i,
                                 epochs=epoches, batch_size=64, binary_A=False,
                                 early_stopping_patience=early_stopping_patience,
                                 early_stopping_delta=early_stopping_delta,
                                 early_stopping_verbose=early_stopping_verbose)

            # Train model_A on training data excluding folds k and i
            train_nuisance_model(model_Atemp, optimizer_A, criterion_A,
                                 A_train_i, A_train_i, X_train_i,
                                 epochs=epoches, batch_size=64, binary_A=binary_A,
                                 early_stopping_patience=early_stopping_patience,
                                 early_stopping_delta=early_stopping_delta,
                                 early_stopping_verbose=early_stopping_verbose)

            # Predict Wp and ap on validation data (fold k)
            model_Wp.eval()
            model_Atemp.eval()
            with torch.no_grad():
                outputs_Wp_i = model_Wp(A_val_i, X_val_i)
                Wp_preds[val_idx_i] = outputs_Wp_i

                outputs_aptemp_i = model_Atemp(X_val_i)
                ap_preds[val_idx_i] = outputs_aptemp_i
                
                outputs_aBar = model_Atemp(X_val_k)
                aBar.append(outputs_aBar)

        # Average predictions over i != k
        Wp_preds = torch.clamp(Wp_preds, 1e-7, 1 - 1e-7)
        Wp_logits = logit(Wp_preds)
        aBar_mean = torch.stack(aBar, dim=0).mean(dim=0)
        Ares2 = A_train_k - ap_preds
        betaNi = (Ares2 * Wp_logits).sum()/ (Ares2**2).sum()
        model_logits_X = NuisanceModeltnk(input_dim=p).to(device)
        optimizer_logits_X = optim.Adam(model_logits_X.parameters(), lr=1e-3)
        criterion_logits_X = nn.MSELoss()
        train_nuisance_model(model_logits_X, optimizer_logits_X, criterion_logits_X,
                                    A_train_k, Wp_logits, X_train_k,
                                    epochs=epoches, batch_size=64, binary_A=False,  
                                    early_stopping_patience=early_stopping_patience,
                                    early_stopping_delta=early_stopping_delta,
                                    early_stopping_verbose=early_stopping_verbose)
        
        model_logits_X.eval()
        tnk = model_logits_X(X_val_k)
        rnk = tnk - betaNi * aBar_mean  # this requires the number of total samples to be integer multiple of K^2
        
        rXp[val_idx_k] = rnk

        # Train model_AY0: E[A | Y=0, X] on training data excluding fold k
        #print("  Training model_AY0: E[A | Y=0, X]")
        
        model_AY0 = NuisanceModelAY0(input_dim=p, binary_A=binary_A).to(device)

        optimizer_AY0 = optim.Adam(model_AY0.parameters(), lr=1e-3)
        criterion_AY0 = nn.MSELoss() if not binary_A else nn.BCELoss()

        # Filter training data where Y=0
        mask_Y0 = Y_train_k == 0
        if mask_Y0.sum() > 0:
            A_train_AY0 = A_train_k[mask_Y0]
            X_train_AY0 = X_train_k[mask_Y0]
            train_nuisance_model(model_AY0, optimizer_AY0, criterion_AY0,
                                 A_train_AY0, A_train_AY0, X_train_AY0,
                                 epochs=epoches, batch_size=64, binary_A=binary_A,
                                 early_stopping_patience=early_stopping_patience,
                                 early_stopping_delta=early_stopping_delta,
                                 early_stopping_verbose=early_stopping_verbose)
        else:
            pass
            #print("    No samples with Y=0 in training set for model_AY0.")

        # Predict mXp on validation set
        #print("  Predicting mXp on validation set...")
        model_AY0.eval()
        with torch.no_grad():
           
            A_pred_Y0 = model_AY0(X_val_k)
            mXp[val_idx_k] = A_pred_Y0

    # Collect results
    dml_results = {
        'mXp': mXp.detach().cpu(),
        'rXp': rXp.detach().cpu(),
    }

    return dml_results

def Estimate(Y, A, dml, beta_init=1.0, lr=1e-3, epochs=200, device='cuda', real_data=False):
    """
    Estimate the causal parameter beta0 by solving the estimating equation.
    
    Parameters:
    - Y (torch.Tensor): Outcome variable, shape (n_samples,)
    - A (torch.Tensor): Treatment variable, shape (n_samples,)
    - dml (dict): Dictionary containing 'mXp' and 'rXp'
        - 'mXp' (torch.Tensor): E[A | Y=0, X] predictions
        - 'rXp' (torch.Tensor): E[W_p | X] predictions
    - beta_init (float): Initial guess for beta0
    - lr (float): Learning rate for optimizer
    - epochs (int): Number of optimization epochs
    - device (str): Device to use ('cuda' or 'cpu')
    
    Returns:
    - beta_estimated (float): Estimated causal parameter beta0
    """
    # Initialize beta as a tensor with requires_grad=True
    beta = torch.tensor(beta_init, device=device, dtype=torch.float32, requires_grad=True)
    
    optimizer = optim.Adam([beta], lr=lr)
    
    Y = Y.to(device)
    A = A.to(device)
    mXp = dml['mXp'].to(device)
    rXp = dml['rXp'].to(device)
    
    # only keep index in rXp with value in [-2,2], do the same for A and Y and mXp
    if real_data:
        mask = (rXp <= 4)
        root_range = 20
    else:
        mask = (rXp <= 2)
        root_range = 10
    A = A[mask].cpu().numpy()
    Y = Y[mask].cpu().numpy()
    mXp = mXp[mask].cpu().numpy()
    rXp = rXp[mask].cpu().numpy()
    
    # # Define EarlyStopping for beta estimation
    # early_stopping = EarlyStopping(patience=20, verbose=True, delta=1e-6, path='best_beta.pt')
    
    # for epoch in range(epochs):
    #     optimizer.zero_grad()
        
    #     resA = A - mXp
    #     C = torch.mean(resA * (1 - Y) * torch.exp(rXp))
        
    #     g_beta = torch.mean(Y * torch.exp(-beta * A) * resA) - C
        
    #     loss = g_beta ** 2  # Minimize the squared estimating equation
        
    #     loss.backward()
    #     optimizer.step()
        
    #     # Compute validation loss (using training loss as a proxy)
    #     val_loss = loss.item()
        
    #     # Early Stopping
    #     # early_stopping(val_loss, model=None)  # No model to save
    #     if epoch % 50 == 0:
    #         pass
            #print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Beta: {beta.item():.6f}')
        
        # if early_stopping.early_stop:
        #     print(f'Early stopping at epoch {epoch+1}')
        #     break
    C = np.mean((A - mXp) * (1 - Y) * np.exp(rXp))
    def g(beta):
        return np.mean(Y * np.exp(-beta * A) * (A - mXp)) - C
    try:
        sol = root_scalar(g, bracket=[-root_range, root_range], method='brentq')
        beta0 = sol.root
    except ValueError:
        # If root not found, assign beta0 = 0
        beta0 = None
    
    # Optionally, load the best beta if saved
    # Here, since no model is saved, return the current beta
    return beta0

def Bootstrap(Y, A, dml, B=1000, real_data=False):
    """
    Perform bootstrapping to estimate the causal parameter beta0.
    
    Parameters:
    - Y (torch.Tensor): Outcome variable, shape (n_samples,)
    - A (torch.Tensor): Treatment variable, shape (n_samples,)
    - dml (dict): Dictionary containing:
        - 'mXp' (torch.Tensor): E[A | Y=0, X] predictions, shape (n_samples,)
        - 'rXp' (torch.Tensor): Wp = logit(E[Y | A, X]) predictions, shape (n_samples,)
    - B (int): Number of bootstrap iterations
    - device (str): Device to use ('cuda' or 'cpu')
    
    Returns:
    - dict: Contains quantiles, mean, and standard deviation of the estimated Betas
        - 'quantiles': numpy array with 2.5% and 97.5% percentiles
        - 'mean': Mean of Betas (excluding None)
        - 'sd': Standard deviation of Betas (excluding None)
    """
    # Move tensors to the specified device and detach from computation graph
    Y = Y.cpu()
    A = A.cpu()
    mXp = dml['mXp'].cpu()
    rXp = dml['rXp'].cpu()
    
    if real_data:
        mask = (rXp <= 4)
        root_range = 20
    else:
        mask = (rXp <= 2)
        root_range = 10
    A = A[mask]
    Y = Y[mask]
    mXp = mXp[mask]
    rXp = rXp[mask]
    # Compute resA once
    resA = A - mXp  # Shape: (n_samples,)
    
    # Convert tensors to CPU numpy arrays for root finding
    Y_np = Y.cpu().numpy()
    A_np = A.cpu().numpy()
    resA_np = resA.cpu().numpy()
    rXp_np = rXp.cpu().numpy()
    
    # Initialize list to store Betas
    Betas = []
    
    for b in range(B):
        # Generate e ~ N(1,1)
        e = np.random.normal(loc=1.0, scale=1.0, size=Y_np.shape[0])
        
        # Compute C = sum(e * resA * (1 - Y) * exp(rXp)
        C = np.mean(e * resA_np * (1 - Y_np) * np.exp(rXp_np))
        
        # Define g(beta) = sum(e * Y * exp(-beta * A) * resA) - C
        def g(beta):
            return np.mean(e * Y_np * np.exp(-beta * A_np) * resA_np) - C
        
        # Find root using brentq within [0, 5]
        try:
            sol = root_scalar(g, bracket=[-root_range, root_range], method='brentq')
            beta0 = sol.root
        except ValueError:
            # If root not found, assign beta0 = 0
            beta0 = None
        
        Betas.append(beta0)
        
        # Optional: Print progress every 100 iterations
        if (b + 1) % 100 == 0:
            pass
            #print(f'Completed {b + 1}/{B} bootstrap iterations')
    
    # Convert Betas to numpy array
    Betas = np.array(Betas)
    
    # Exclude Betas equal to 0 (failed root-finding)
    Betas_non_none = Betas[Betas != None]
    
    # Compute quantiles, mean, and standard deviation
    quantiles = np.quantile(Betas_non_none, [0.025, 0.975])
    mean_beta = np.mean(Betas_non_none)
    sd_beta = np.std(Betas_non_none)
    
    return {
        'quantiles': quantiles,
        'mean': mean_beta,
        'sd': sd_beta
    }
