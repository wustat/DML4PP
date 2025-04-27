import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, random_split

def DML(Y, F, Z, K_outer=5, K_inner=3, epochs=100, batch_size=64, 
    early_stopping_patience=10, early_stopping_delta=0.0,
    early_stopping_verbose=False, val_ratio=0.2, fit_verbose=False):
    """
    Perform Double/Debiased Machine Learning (DML) to estimate provider effects.
    
    Parameters:
        Y (np.ndarray): Outcome variable (1D array).
        F (np.ndarray): Provider ID (1D array).
        Z (np.ndarray): Confounders/covariates (2D array).
        K_outer (int): Number of outer cross-fitting folds.
        K_inner (int): Number of inner cross-fitting folds.
        epochs (int): Number of training epochs for the neural networks.
        batch_size (int): Batch size for training the neural networks.
        early_stopping_patience (int): Patience for early stopping (epochs).
        early_stopping_delta (float): Minimum delta to consider an improvement in validation loss.
        early_stopping_verbose (bool): Whether to print early stopping messages.
        val_ratio (float): Ratio of the training data to use for validation.
        fit_verbose (bool): Whether to print training progress.

    Returns:
        torch.Tensor: Estimated treatment effect (gamma).
    """
    n_providers = len(np.unique(F))
    providers = torch.unique(torch.as_tensor(F))
    folds_outer = StratifiedKFold(n_splits=K_outer, shuffle=True)
    folds_inner = StratifiedKFold(n_splits=K_inner, shuffle=True)
    g = torch.zeros(len(Y)) # nonlinear predictor of the outcome model
    d = torch.zeros((len(Y), n_providers)) # propensity scores
    for fold_outer, (idx_train_outer, idx_val_outer) in enumerate(
        folds_outer.split(Y, F)):
        Y_train_outer, Y_val_outer = Y[idx_train_outer], Y[idx_val_outer]
        F_train_outer, F_val_outer = F[idx_train_outer], F[idx_val_outer]
        Z_train_outer, Z_val_outer = Z[idx_train_outer], Z[idx_val_outer]
        logit_train_outer = torch.zeros(len(Y_train_outer))
        prob_train_outer = torch.zeros((len(Y_train_outer), n_providers))
        prob_val_outer = torch.zeros((len(Y_val_outer), n_providers))
        for fold_inner, (idx_train_inner, idx_val_inner) in enumerate(
            folds_inner.split(Y_train_outer, F_train_outer)):
            Y_train_inner, Y_val_inner =
                Y_train_outer[idx_train_inner], Y_train_outer[idx_val_inner]
            F_train_inner, F_val_inner =
                F_train_outer[idx_train_inner], F_train_outer[idx_val_inner]
            Z_train_inner, Z_val_inner =
                Z_train_outer[idx_train_inner], Z_train_outer[idx_val_inner]
            # initialization of ModelY and ModelF
            modelY = ModelY(n_providers, Z_train_inner.shape[1])
            modelF = ModelF(Z_train_inner.shape[1], n_providers)
            optim_modelY = optim.Adam(modelY.parameters(), lr=1e-3, amsgrad=True)
            optim_modelF = optim.Adam(modelF.parameters(), lr=1e-3, amsgrad=True)
            # fitting
            fit(modelY, optim_modelY, torch.as_tensor(Y_train_inner).float(),
                torch.as_tensor(F_train_inner).float(),
                torch.as_tensor(Z_train_inner).float(),
                epochs, batch_size, early_stopping_patience, early_stopping_delta,
                early_stopping_verbose, val_ratio, fit_verbose)
            fit(modelF, optim_modelF, torch.as_tensor(Y_train_inner).float(),
                torch.as_tensor(F_train_inner).float(),
                torch.as_tensor(Z_train_inner).float(),
                epochs, batch_size, early_stopping_patience, early_stopping_delta,
                early_stopping_verbose, val_ratio, fit_verbose)
            modelY.eval()
            modelF.eval()
            with torch.no_grad():
                logit_train_outer[idx_val_inner] = modelY.logit(
                    torch.as_tensor(F_val_inner),
                    torch.as_tensor(Z_val_inner)
                ).cpu().flatten() # logit M as in (3.4) of Liu et al. (2021)
                prob_val_inner = torch.softmax(
                    modelF(torch.as_tensor(Z_val_inner)), dim=1
                ).cpu()
                prob_train_outer[idx_val_inner, :] = prob_val_inner
                prob_val_outer += prob_val_inner
        prob_val_outer /= K_inner
        # beta as in (3.4) of Liu et al. (2021)
        gamma_outer = torch.linalg.lstsq(
            torch.nn.functional.one_hot(torch.as_tensor(F_train_outer).long(), 
            num_classes=n_providers) - prob_train_outer,
            logit_train_outer).solution.flatten()
        # initialization of Modelt and ModelF (for Y=0 only)
        modelt = Modelt(Z_train_outer.shape[1])
        modelF_Y0 = ModelF(Z_train_outer.shape[1], n_providers)
        optim_modelt = optim.Adam(modelt.parameters(), lr=1e-3, amsgrad=True)
        optim_modelF_Y0 = optim.Adam(modelF_Y0.parameters(), lr=1e-3, amsgrad=True)
        fit(modelt, optim_modelt, logit_train_outer,
            torch.as_tensor(F_train_outer).float(),
            torch.as_tensor(Z_train_outer).float(),
            epochs, batch_size, early_stopping_patience, early_stopping_delta,
            early_stopping_verbose, val_ratio, fit_verbose)
        mask_Y0 = (Y_train_outer == 0)
        fit(modelF_Y0, optim_modelF_Y0,
            torch.as_tensor(Y_train_outer[mask_Y0]).float(),
            torch.as_tensor(F_train_outer[mask_Y0]).float(),
            torch.as_tensor(Z_train_outer[mask_Y0, :]).float(),
            epochs, batch_size, early_stopping_patience, early_stopping_delta,
            early_stopping_verbose, val_ratio, fit_verbose)
        modelt.eval()
        modelF_Y0.eval()
        g[idx_val_outer] = modelt(torch.as_tensor(Z_val_outer)).cpu().flatten() -
            prob_val_outer @ gamma_outer
        d[idx_val_outer, :] =
            torch.softmax(modelF_Y0(torch.as_tensor(Z_val_outer)), dim=1).cpu()
    # solve cross-fitted Neymann orthogonal estimating equations as in Liu et al. (2021) eq. (3.3)
    # psi(x) = expit(-r_0(x))
    sigmoid_g = torch.sigmoid(g)
    F_onehot = torch.nn.functional.one_hot(torch.as_tensor(F).long(),
        num_classes=n_providers)
    d_comp = F_onehot - d
    gamma = -torch.log(torch.clamp(torch.linalg.solve(
      F_onehot.t() @ ((1 - sigmoid_g)[:, None] * torch.as_tensor(Y)[:, None] * d_comp),
      (sigmoid_g[Y == 0, None] * d_comp[Y == 0, :]).sum(dim=0)),
      min=1e-7))
    return gamma

def init_weights_biases(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

class ModelY(nn.Module):
    """
    Model for E[Y] = expit{gamma[F] + r(Z)}
    where gamma is a learnable vector of provider effects (n_providers,)
    """
    def __init__(self, n_providers, Z_dim):
        super(ModelY, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(n_providers))
        self.network = nn.Sequential(
            nn.Linear(Z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1, bias=False)
        )
        self.apply(init_weights_biases)

    def forward(self, F, Z):
        """
        F: (batch_size,) - integer-encoded provider IDs
        Z: (batch_size, Z_dim) - covariates/features
        Returns: (batch_size,) - predicted E[Y]
        """
        lin = self.gamma[F]                        # (batch_size,)
        nonlin = self.network(Z).squeeze(-1)       # (batch_size,)
        par_lin = lin + nonlin
        return torch.sigmoid(par_lin)              # (batch_size,)
    
    def logit(self, F, Z): # a method
        """
        F: (batch_size,) - integer-encoded provider IDs
        Z: (batch_size, Z_dim) - covariates/features
        Returns: (batch_size,) - logits of predicted E[Y]
        """
        lin = self.gamma[F]
        nonlin = self.network(Z).squeeze(-1)
        par_lin = lin + nonlin
        return par_lin
    
class ModelF(nn.Module):
    """
    Model for E[F | Z] or E[F | Z, Y=0], where F is a categorical variable (provider ID).
    """
    def __init__(self, Z_dim, n_providers):
        super(ModelF, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(Z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_providers)
        )
        self.apply(init_weights_biases)
    
    def forward(self, Z):
        """
        Z: (batch_size, Z_dim)
        Returns: (batch_size, n_providers) - logits
        Use nn.CrossEntropyLoss for training, which expects logits 
        (not probabilities) and target as integer class indices.
        # Inference (to get probabilities):
        # probs = torch.softmax(model(Z), dim=1)
        """
        return self.network(Z)  # Return logits directly
      
class Modelt(nn.Module):
    """
    Model for E[logit(M_0) | Z] as in Liu et al. 2021 (right above Eq. (3.5)).
    Predicts a continuous outcome given input features Z.
    """
    def __init__(self, Z_dim):
        super(Modelt, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(Z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output is a scalar (logit)
        )
        self.apply(init_weights_biases)
    
    def forward(self, Z):
        """
        Z: (batch_size, Z_dim)
        Returns: (batch_size,) - predicted logit(M_0)
        """
        return self.network(Z).squeeze(-1)  # Ensures output is (batch_size,)

class EarlyStopping:
    """
    Early stopping halts training if the validation loss does not improve after a specified patience period.
    See https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
    """
    def __init__(self, patience=10, delta=0.0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait after the last improvement in validation loss before stopping.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message each time the validation loss improves.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score == float('inf'): # Handle first epoch
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping patience: {self.counter}/{self.patience} epochs without improvement.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

def fit(model, optimizer, R, F, Z, criterion=None, epochs=100,
    batch_size=64, early_stopping_patience=10, early_stopping_delta=0.0,
    early_stopping_verbose=False, val_ratio=0.2, fit_verbose=False):
    """
    Train a nuisance parameter model with early stopping.

    Parameters:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        R (torch.Tensor): Response variable tensor.
        F (torch.Tensor): Provider indicator tensor.
        Z (torch.Tensor): Covariate tensor.
        criterion: Loss function. If None, selects based on model type.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training.
        early_stopping_patience (int): Number of epochs with no validation loss improvement before stopping.
        early_stopping_delta (float): Minimum change in validation loss to qualify as improvement.
        early_stopping_verbose (bool): If True, prints a message for each validation loss improvement.
        val_ratio (float): Fraction of data to use for validation (default: 0.2 for 80/20 split).
        fit_verbose (bool): If True, prints training and validation losses every 10 epochs.
    """
    # Automatically select loss function if not provided
    if criterion is None:
        if isinstance(model, ModelY):
            # Binary outcome
            criterion = nn.BCELoss()  # or nn.BCEWithLogitsLoss() if ModelY outputs logits
        elif isinstance(model, ModelF):
            # Categorical outcome (assume F is class indices)
            criterion = nn.CrossEntropyLoss()
        elif isinstance(model, Modelt):
            # Continuous outcome
            criterion = nn.MSELoss()
        else:
            raise ValueError("Unknown model class.")
    # Compose dataset based on model type
    if isinstance(model, ModelY):
        dataset = TensorDataset(R, F, Z)
    elif isinstance(model, ModelF):
        dataset = TensorDataset(F, Z)
    elif isinstance(model, Modelt):
        dataset = TensorDataset(R, Z)
    # Split the dataset into training and validation sets
    n_val = int(val_ratio * len(dataset))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        verbose=early_stopping_verbose,
        delta=early_stopping_delta
    )

    for epoch in range(epochs):
        model.train()  # Ensure model is in training mode
        train_loss = 0.0
        train_samples = 0
        # Training phase
        for batch in train_loader:
            optimizer.zero_grad()
            if isinstance(model, ModelY):
                batch_R, batch_F, batch_Z = batch
                pred = model(batch_F, batch_Z)
                loss = criterion(pred, batch_R.float())
            elif isinstance(model, ModelF):
                batch_F, batch_Z = batch
                pred = model(batch_Z)
                loss = criterion(pred, batch_F.long())
            elif isinstance(model, Modelt):
                batch_R, batch_Z = batch
                pred = model(batch_Z)
                loss = criterion(pred, batch_R.float())
            batch_size = batch_Z.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size
            loss.backward()
            optimizer.step()
        train_loss /= train_samples
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(model, ModelY):
                    batch_R, batch_F, batch_Z = batch
                    pred = model(batch_F, batch_Z)
                    loss = criterion(pred, batch_R.float())
                elif isinstance(model, ModelF):
                    batch_F, batch_Z = batch
                    pred = model(batch_Z)
                    loss = criterion(pred, batch_F.long())
                elif isinstance(model, Modelt):
                    batch_R, batch_Z = batch
                    pred = model(batch_Z)
                    loss = criterion(pred, batch_R.float())
                batch_size = batch_Z.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
        val_loss /= val_samples
        # Early stopping
        early_stopping(val_loss, model)
        if fit_verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"training loss: {train_loss:.4f}; "
                f"validation loss: {val_loss:.4f}")
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            # Load the best model state
            model.load_state_dict(early_stopping.best_model_state)
            break
    return model
