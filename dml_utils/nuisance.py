import torch.nn.functional as F
from sklearn.model_selection import KFold

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, random_split

def DML(Y, F, Z, K_outer=5, K_inner=4, epochs=100, batch_size=64, 
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
            Y_train_inner, Y_val_inner = \
                Y_train_outer[idx_train_inner], Y_train_outer[idx_val_inner]
            F_train_inner, F_val_inner = \
                F_train_outer[idx_train_inner], F_train_outer[idx_val_inner]
            Z_train_inner, Z_val_inner = \
                Z_train_outer[idx_train_inner], Z_train_outer[idx_val_inner]
            # initialization of ModelY and ModelF
            modelY = ModelY(n_providers, Z_train_inner.shape[1])
            modelF = ModelF(Z_train_inner.shape[1], n_providers)
            optim_modelY = optim.Adam(modelY.parameters(), lr=1e-3, amsgrad=True)
            optim_modelF = optim.Adam(modelF.parameters(), lr=1e-3, amsgrad=True)
            # fitting
            # fit(modelY, optim_modelY, torch.as_tensor(Y_train_inner).float(),
            #     torch.as_tensor(F_train_inner).float(),
            #     torch.as_tensor(Z_train_inner).float(), None,
            #     epochs, batch_size, early_stopping_patience, early_stopping_delta,
            #     early_stopping_verbose, val_ratio, fit_verbose)
            
            fit(modelY, optim_modelY, torch.as_tensor(Y_train_inner[np.where(F_train_inner != providers[-1].cpu().numpy())[0]]).float(),
                torch.as_tensor(F_train_inner[np.where(F_train_inner != providers[-1].cpu().numpy())[0]]).float(),
                torch.as_tensor(Z_train_inner[np.where(F_train_inner != providers[-1].cpu().numpy())[0]]).float(), None,
                epochs, batch_size, early_stopping_patience, early_stopping_delta,
                early_stopping_verbose, val_ratio, fit_verbose)
            fit(modelF, optim_modelF, torch.as_tensor(Y_train_inner).float(),
                torch.as_tensor(F_train_inner).float(),
                torch.as_tensor(Z_train_inner).float(), None, 
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
        prob_val_outer_noref = prob_val_outer[:,:-1]
        prob_train_outer_noref = prob_train_outer[:,:-1]
        F_train_outer_noref = F_train_outer[np.where(F_train_outer != providers[-1].cpu().numpy())[0]]
        logit_train_outer_noref = logit_train_outer[np.where(F_train_outer != providers[-1].cpu().numpy())[0]]
        # this is for adding ref group
        gamma_outer = torch.linalg.lstsq(
            torch.nn.functional.one_hot(torch.as_tensor(F_train_outer_noref).long(), 
            num_classes=n_providers-1) - prob_train_outer_noref[np.where(F_train_outer != providers[-1].cpu().numpy())[0]],
            logit_train_outer_noref).solution.flatten()
        # gamma_outer = torch.linalg.lstsq(
        #     torch.nn.functional.one_hot(torch.as_tensor(F_train_outer).long(), 
        #     num_classes=n_providers) - prob_train_outer,
        #     logit_train_outer).solution.flatten()
        # initialization of Modelt and ModelF (for Y=0 only)
        modelt = Modelt(Z_train_outer.shape[1])
        modelF_Y0 = ModelF(Z_train_outer.shape[1], n_providers)
        optim_modelt = optim.Adam(modelt.parameters(), lr=1e-3, amsgrad=True)
        optim_modelF_Y0 = optim.Adam(modelF_Y0.parameters(), lr=1e-3, amsgrad=True)
        fit(modelt, optim_modelt, logit_train_outer,
            torch.as_tensor(F_train_outer).float(),
            torch.as_tensor(Z_train_outer).float(), None,
            epochs, batch_size, early_stopping_patience, early_stopping_delta,
            early_stopping_verbose, val_ratio, fit_verbose)
        mask_Y0 = (Y_train_outer == 0)
        fit(modelF_Y0, optim_modelF_Y0,
            torch.as_tensor(Y_train_outer[mask_Y0]).float(),
            torch.as_tensor(F_train_outer[mask_Y0]).float(),
            torch.as_tensor(Z_train_outer[mask_Y0, :]).float(), None, 
            epochs, batch_size, early_stopping_patience, early_stopping_delta,
            early_stopping_verbose, val_ratio, fit_verbose)
        modelt.eval()
        modelF_Y0.eval()
        g[idx_val_outer] = modelt(torch.as_tensor(Z_val_outer)).cpu().flatten() - \
            prob_val_outer_noref @ gamma_outer  # was prob_val_outer @ gamma_outer
        d[idx_val_outer, :] = \
            torch.softmax(modelF_Y0(torch.as_tensor(Z_val_outer)), dim=1).cpu()
    # solve cross-fitted Neymann orthogonal estimating equations as in Liu et al. (2021) eq. (3.3)
    # psi(x) = expit(-r_0(x))
    sigmoid_g = torch.sigmoid(g)
    # F_onehot = torch.nn.functional.one_hot(torch.as_tensor(F).long(),
    #     num_classes=n_providers)
    F_noref = F[np.where(F != providers[-1].cpu().numpy())[0]]
    F_onehot = torch.nn.functional.one_hot(torch.as_tensor(F_noref).long(), num_classes=n_providers-1)
    d_comp = F_onehot - d[np.where(F != providers[-1].cpu().numpy())[0]][:, :-1] # used to be F_onehot - d, change to d[:, :-1] for no ref
    sigmoid_g = sigmoid_g[np.where(F != providers[-1].cpu().numpy())[0]]
    Y = Y[np.where(F != providers[-1].cpu().numpy())[0]]
    gamma = -torch.log(torch.clamp(torch.linalg.solve(
      F_onehot.float().t() @ ((1 - sigmoid_g)[:, None] * torch.as_tensor(Y)[:, None] * d_comp ) ,
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
            nn.Linear(Z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1, bias=False)
        )
        self.apply(init_weights_biases)

    def forward(self, F, Z):
        """
        F: (batch_size,) - integer-encoded provider IDs
        Z: (batch_size, Z_dim) - covariates/features
        Returns: (batch_size,) - predicted E[Y]
        """
        # check if F is float
        if torch.is_floating_point(F):
            lin = self.gamma[F.int()]  
        else:
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

# class ModelY(nn.Module):
#     """
#     Model for E[Y] = expit{gamma[F] + r(Z)}
#     where gamma is a learnable vector of provider effects (n_providers,)
#     """
#     def __init__(self, n_providers, Z_dim):
#         super(ModelY, self).__init__()
#         #self.gamma = nn.Parameter(torch.zeros(n_providers))
#         self.n_providers = n_providers
#         self.network = nn.Sequential(
#             nn.Linear(Z_dim+n_providers, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1, bias=False)
#         )
#         self.apply(init_weights_biases)

#     def forward(self, F, Z):
#         """
#         F: (batch_size,) - integer-encoded provider IDs
#         Z: (batch_size, Z_dim) - covariates/features
#         Returns: (batch_size,) - predicted E[Y]
#         """
#         onehot_F = torch.nn.functional.one_hot(F.long(), num_classes=self.n_providers).float()
#         par_lin = self.network(torch.cat((Z, onehot_F), dim=1)).squeeze(-1)       # (batch_size,)
#         return torch.sigmoid(par_lin)              # (batch_size,)

#     def logit(self, F, Z): # a method
#         """
#         F: (batch_size,) - integer-encoded provider IDs
#         Z: (batch_size, Z_dim) - covariates/features
#         Returns: (batch_size,) - logits of predicted E[Y]
#         """
#         onehot_F = torch.nn.functional.one_hot(F.long(), num_classes=self.n_providers).float()
#         par_lin = self.network(torch.cat((Z, onehot_F), dim=1)).squeeze(-1)
#         return par_lin
    
class ModelF(nn.Module):
    """
    Model for E[F | Z] or E[F | Z, Y=0], where F is a categorical variable (provider ID).
    """
    def __init__(self, Z_dim, n_providers):
        super(ModelF, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(Z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_providers)
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
            nn.Linear(Z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output is a scalar (logit)
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
            #print("Early stopping triggered.")
            # Load the best model state
            model.load_state_dict(early_stopping.best_model_state)
            break
    return model

def logit(x):
    """
    Compute the logit function with numerical stability.
    
    Parameters:
    - x: torch.Tensor, probabilities in (0, 1)
    
    Returns:
    - logit(x): torch.Tensor, log-odds
    """
    x = torch.clamp(x, 1e-7, 1 - 1e-7)
    return torch.log(x / (1 - x))

class NuisanceModelY(nn.Module):
    """
    Model to estimate E[Y | A, X].
    """
    def __init__(self, input_dim):
        super(NuisanceModelY, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, 1)  # Output: prob(Y)
        )
    
    def forward(self, A, X):
        # Concatenate A and X
        if len(A.shape) == 1:
            inputs = torch.cat([A.unsqueeze(1), X], dim=1)
        else:
            inputs = torch.cat([A, X], dim=1)
        return torch.sigmoid(self.network(inputs).squeeze())

class NuisanceModelY_tr(nn.Module):
    """
    Transformer-based model to estimate E[Y | A, X].
    
    Here, each scalar from the concatenated [A, X] is treated as a token.
    """
    def __init__(self, input_dim, d_model=16, nhead=2, num_layers=2, dim_feedforward=128):
        """
        Args:
            input_dim (int): Total number of features (i.e. 1 for A + dimension of X).
            d_model (int): Embedding dimension for each token.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network in the encoder.
        """
        super(NuisanceModelY_tr, self).__init__()
        # Each scalar feature becomes a token with dimension 1.
        # We embed each token to dimension d_model.
        self.token_embedding = nn.Linear(1, d_model)
        # Create learnable positional embeddings for each token.
        self.positional_embedding = nn.Parameter(torch.randn(input_dim, d_model))
        
        # Define a Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer to produce a single output.
        self.output_linear = nn.Linear(d_model, 1)
    
    def forward(self, A, X):
        # Concatenate A and X along the feature dimension.
        # If A is a 1D tensor (batch,), unsqueeze to (batch, 1).
        if len(A.shape) == 1:
            inputs = torch.cat([A.unsqueeze(1), X], dim=1)
        else:
            inputs = torch.cat([A, X], dim=1)  # shape: (batch_size, total_features)
        
        # Treat each scalar as a token by adding a last singleton dimension.
        # New shape: (batch_size, total_features, 1)
        tokens = inputs.unsqueeze(-1)
        # Embed each token: (batch_size, total_features, d_model)
        embedded_tokens = self.token_embedding(tokens)
        # Add positional embeddings (broadcasted along the batch dimension).
        # positional_embedding shape: (total_features, d_model)
        embedded_tokens = embedded_tokens + self.positional_embedding.unsqueeze(0)
        
        # Transformer expects input shape: (sequence_length, batch_size, d_model)
        transformer_input = embedded_tokens.transpose(0, 1)
        transformer_output = self.transformer(transformer_input)
        # Pool over the sequence dimension (e.g., mean pooling).
        pooled = transformer_output.mean(dim=0)  # shape: (batch_size, d_model)
        
        # Final linear layer to produce the output, then apply sigmoid.
        out = self.output_linear(pooled).squeeze(-1)
        return torch.sigmoid(out)

    
class NuisanceModelAtemp(nn.Module):
    """
    Model to estimate E[A | X].
    """
    def __init__(self, input_dim, binary_A=False):
        super(NuisanceModelAtemp, self).__init__()
        self.binary_A = binary_A
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, 1)
            # Output: A prediction
        )
    
    def forward(self, X):
        outputs = self.network(X).squeeze()
        if self.binary_A:
            # If A is binary, output probability
            outputs = torch.sigmoid(outputs)
        return outputs


class NuisanceModelAtemp_tr(nn.Module):
    """
    Transformer-based model to estimate E[A | X] without using a CLS token.
    Each scalar in X is treated as a token. After embedding and passing the tokens
    through a Transformer encoder, we aggregate via mean pooling to produce a scalar output.
    """
    def __init__(self, input_dim, d_model=16, nhead=2, num_layers=2, dim_feedforward=128, binary_A=False):
        """
        Args:
            input_dim (int): Number of features in X.
            d_model (int): Embedding dimension for each token.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network in the encoder.
            binary_A (bool): If True, apply sigmoid to the output.
        """
        super(NuisanceModelAtemp_tr, self).__init__()
        self.binary_A = binary_A
        
        # Each scalar feature in X is treated as a token, embed from dimension 1 to d_model.
        self.token_embedding = nn.Linear(1, d_model)
        
        # Learnable positional embeddings for each token (one per feature in X).
        self.positional_embedding = nn.Parameter(torch.randn(input_dim, d_model))
        
        # Define the Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer to produce a scalar output.
        self.output_linear = nn.Linear(d_model, 1)
    
    def forward(self, X):
        """
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Output predictions (batch_size,) where each element is the estimated A.
        """
        # Convert each scalar feature to a token: (batch_size, input_dim, 1)
        tokens = X.unsqueeze(-1)
        
        # Embed tokens: each scalar becomes a d_model-dimensional vector.
        embedded_tokens = self.token_embedding(tokens)
        
        # Add positional embeddings to maintain order information.
        embedded_tokens = embedded_tokens + self.positional_embedding.unsqueeze(0)
        
        # Transformer expects input shape: (sequence_length, batch_size, d_model)
        transformer_input = embedded_tokens.transpose(0, 1)
        transformer_output = self.transformer(transformer_input)
        
        # Mean pooling across the sequence (token) dimension.
        pooled = transformer_output.mean(dim=0)  # (batch_size, d_model)
        
        # Produce final scalar output.
        out = self.output_linear(pooled).squeeze(-1)  # (batch_size)
        
        # For binary A, apply sigmoid.
        if self.binary_A:
            out = torch.sigmoid(out)
        return out

class NuisanceModelAtempMulti_mlp(nn.Module):
    """
    Model to estimate E[A | X], which is a vector of length l, l is given
    """
    def __init__(self, input_dim, l, binary_A=False):
        super(NuisanceModelAtempMulti_mlp, self).__init__()
        self.binary_A = binary_A
        self.network = nn.Sequential(
            nn.Linear(input_dim, 4*l),
            nn.ReLU(),
            nn.Linear(4*l, 4*l),
            nn.ReLU(),
            nn.Linear(4*l, l)
            # Output: A prediction
        )
    
    def forward(self, X):
        # softmax to standardize the output
        outputs = F.softmax(self.network(X), dim=1)
        return outputs

class NuisanceModelAMulti_mlp(nn.Module):
    """
    Model to estimate E[A | X], which is a vector of length l, l is given
    """
    def __init__(self, input_dim, l, binary_A=False):
        super(NuisanceModelAMulti_mlp, self).__init__()
        self.binary_A = binary_A
        self.network = nn.Sequential(
            nn.Linear(input_dim, 4*l),
            nn.ReLU(),
            nn.Linear(4*l, 4*l),
            nn.ReLU(),
            nn.Linear(4*l, l)
            # Output: A prediction
        )
    
    def forward(self, X):
        # softmax to standardize the output
        outputs = F.softmax(self.network(X), dim=1)
        return outputs
    
class NuisanceModelA(nn.Module):
    """
    Model to estimate E[A | X].
    """
    def __init__(self, input_dim, binary_A=False):
        super(NuisanceModelA, self).__init__()
        self.binary_A = binary_A
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, 1)
            # Output: A prediction
        )
    
    def forward(self, X):
        outputs = self.network(X).squeeze()
        if self.binary_A:
            # If A is binary, output probability
            outputs = torch.sigmoid(outputs)
        return outputs

class NuisanceModelWp_mlp(nn.Module):
    """
    Model to estimate Wp = logit(E[Y | A, X]).
    """
    def __init__(self, input_dim):
        super(NuisanceModelWp_mlp, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, 1) # Output:logit(Wp)
        )
    
    def forward(self, A, X):
        if len(A.shape) == 1:
            inputs = torch.cat([A.unsqueeze(1), X], dim=1)
        else:
            inputs = torch.cat([A, X], dim=1)
        
        return self.network(inputs).squeeze()
    
class NuisanceModeltnk(nn.Module):
    """
    Model to estimate logits ~ X.
    """
    def __init__(self, input_dim):
        super(NuisanceModeltnk, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, 1) # Output: logit(Wp)
        )
    
    def forward(self, X):
        return self.network(X).squeeze()

class NuisanceModeltnk_tr(nn.Module):
    """
    Transformer-based model to estimate logits ~ X.
    
    Each scalar feature in X is treated as a token. Tokens are embedded to a d_model-dimensional space,
    combined with positional embeddings, processed through a Transformer encoder, and then aggregated via mean pooling.
    The final linear layer outputs a single logit.
    """
    def __init__(self, input_dim, d_model=16, nhead=2, num_layers=2, dim_feedforward=128):
        """
        Args:
            input_dim (int): Number of features in X.
            d_model (int): Embedding dimension for each token.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network in the encoder.
        """
        super(NuisanceModeltnk_tr, self).__init__()
        
        # Embed each scalar feature (token) from dimension 1 to d_model.
        self.token_embedding = nn.Linear(1, d_model)
        
        # Learnable positional embeddings for each token.
        self.positional_embedding = nn.Parameter(torch.randn(input_dim, d_model))
        
        # Define the Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer to produce a scalar logit.
        self.output_linear = nn.Linear(d_model, 1)
    
    def forward(self, X):
        """
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size,).
        """
        # Reshape X to treat each scalar feature as a token: (batch_size, input_dim, 1)
        tokens = X.unsqueeze(-1)
        
        # Embed each token from dimension 1 to d_model.
        embedded_tokens = self.token_embedding(tokens)
        
        # Add positional embeddings (broadcast along the batch dimension).
        embedded_tokens = embedded_tokens + self.positional_embedding.unsqueeze(0)
        
        # Transformer expects input shape: (sequence_length, batch_size, d_model)
        transformer_input = embedded_tokens.transpose(0, 1)
        transformer_output = self.transformer(transformer_input)
        
        # Mean pooling over the sequence (token) dimension.
        pooled = transformer_output.mean(dim=0)  # (batch_size, d_model)
        
        # Final linear layer produces a single logit.
        logits = self.output_linear(pooled).squeeze(-1)
        return logits


class NuisanceModelAY0Multi_mlp(nn.Module):
    """
    Model to estimate E[A | Y=0, X], which is a vector of length l, l is given
    """
    def __init__(self, input_dim, l, binary_A=False):
        super(NuisanceModelAY0Multi_mlp, self).__init__()
        self.binary_A = binary_A
        self.network = nn.Sequential(
            nn.Linear(input_dim, 4*l),
            nn.ReLU(),
            nn.Linear(4*l, 4*l),
            nn.ReLU(),
            nn.Linear(4*l, l)
            # Output: A prediction
        )
    
    def forward(self, X):
        # softmax to standardize the output
        outputs = F.softmax(self.network(X), dim=1)
        return outputs



class NuisanceModelAY0(nn.Module):
    """
    Model to estimate E[A | Y=0, X].
    """
    def __init__(self, input_dim, binary_A=False):
        super(NuisanceModelAY0, self).__init__()
        self.binary_A = binary_A
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, 1)  # Output: A prediction
        )
    
    def forward(self, X):
        outputs = self.network(X).squeeze()
        if self.binary_A:
            # If A is binary, output probability
            outputs = torch.sigmoid(outputs)
        return outputs

class NuisanceModelAY0_tr(nn.Module):
    """
    Transformer-based model to estimate E[A | Y=0, X].
    
    Each scalar in X is treated as a token. The tokens are embedded into a d_model-dimensional space,
    combined with positional embeddings, and processed through a Transformer encoder. Mean pooling aggregates
    the token outputs, and a final linear layer produces the scalar output.
    """
    def __init__(self, input_dim, d_model=16, nhead=2, num_layers=2, dim_feedforward=128, binary_A=False):
        """
        Args:
            input_dim (int): Number of features in X.
            d_model (int): Embedding dimension for each token.
            nhead (int): Number of attention heads in the Transformer encoder.
            num_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network in each Transformer layer.
            binary_A (bool): If True, apply a sigmoid to the output (for binary A).
        """
        super(NuisanceModelAY0_tr, self).__init__()
        self.binary_A = binary_A
        
        # Embed each scalar feature (token) from dimension 1 to d_model.
        self.token_embedding = nn.Linear(1, d_model)
        
        # Learnable positional embeddings (one per token).
        self.positional_embedding = nn.Parameter(torch.randn(input_dim, d_model))
        
        # Define the Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer to produce a scalar output.
        self.output_linear = nn.Linear(d_model, 1)
    
    def forward(self, X):
        """
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,) with the estimated A.
        """
        # Convert each scalar feature in X into a token: (batch_size, input_dim, 1)
        tokens = X.unsqueeze(-1)
        
        # Embed each token: from 1-dim to d_model-dim.
        embedded_tokens = self.token_embedding(tokens)
        
        # Add positional embeddings (broadcasting along the batch dimension).
        embedded_tokens = embedded_tokens + self.positional_embedding.unsqueeze(0)
        
        # Transformer expects input shape: (sequence_length, batch_size, d_model)
        transformer_input = embedded_tokens.transpose(0, 1)
        transformer_output = self.transformer(transformer_input)
        
        # Mean pooling over the sequence (token) dimension.
        pooled = transformer_output.mean(dim=0)  # shape: (batch_size, d_model)
        
        # Final linear layer to produce a single scalar output.
        output = self.output_linear(pooled).squeeze(-1)  # shape: (batch_size)
        
        # If A is binary, convert the output to a probability.
        if self.binary_A:
            output = torch.sigmoid(output)
        return output



class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss  # Because we want to minimize validation loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased. Saving model to {self.path}')
        # torch.save(model.state_dict(), self.path)
        self.best_model_state = model.state_dict()

# def train_Wp_nested(Y, A, X, K_inner=5, binary_A=False, device='cuda',
#                       early_stopping_patience=10, early_stopping_delta=1e-4, 
#                       early_stopping_verbose=False):
#     """
#     Estimate Wp = logit(E[Y | A, X]) using nested K-fold cross-fitting.
    
#     Parameters:
#     - Y (torch.Tensor): Outcome variable, shape (n_samples,)
#     - A (torch.Tensor): Treatment variable, shape (n_samples,)
#     - X (torch.Tensor): Covariates, shape (n_samples, p)
#     - K_inner (int): Number of inner folds for cross-fitting
#     - binary_A (bool): If True, treat A as binary
#     - device (str): Device to use ('cuda' or 'cpu')
#     - early_stopping_patience (int): Patience for Early Stopping
#     - early_stopping_delta (float): Minimum improvement for Early Stopping
#     - early_stopping_verbose (bool): Verbosity for Early Stopping
    
#     Returns:
#     - Wp_pred (torch.Tensor): Estimated Wp values, shape (n_samples,)
#     """
#     n_samples, p = X.shape
#     Wp_pred = torch.zeros(n_samples, device=device)
    
#     # Define outer K-fold (same as K_inner here for simplicity)
#     kf_outer = KFold(n_splits=K_inner, shuffle=True, random_state=42)
    
#     for fold_outer, (train_idx_outer, val_idx_outer) in enumerate(kf_outer.split(range(n_samples)), 1):
#         print(f"  Inner Fold {fold_outer}/{K_inner}:")
        
#         # Split data
#         X_train_outer, A_train_outer, Y_train_outer = X[train_idx_outer], A[train_idx_outer], Y[train_idx_outer]
#         X_val_outer, A_val_outer, Y_val_outer = X[val_idx_outer], A[val_idx_outer], Y[val_idx_outer]
        
#         # Initialize Wp model
#         if binary_A:
#             model_Wp = NuisanceModelWp(input_dim=p).to(device)
#         else:
#             model_Wp = NuisanceModelWp(input_dim=p).to(device)
        
#         optimizer_Wp = optim.Adam(model_Wp.parameters(), lr=1e-3)
#         criterion_Wp = nn.BCEWithLogitsLoss()
        
#         # Initialize EarlyStopping
#         early_stopping_Wp = EarlyStopping(patience=early_stopping_patience, 
#                                          verbose=early_stopping_verbose, 
#                                          delta=early_stopping_delta, 
#                                          path=f'best_model_Wp_inner_fold_{fold_outer}.pt')
        
#         # Create DataLoader
#         dataset_Wp = torch.utils.data.TensorDataset(A_train_outer, Y_train_outer, X_train_outer)
#         loader_Wp = torch.utils.data.DataLoader(dataset_Wp, batch_size=64, shuffle=True)
        
#         # Create validation DataLoader
#         dataset_Wp_val = torch.utils.data.TensorDataset(A_val_outer, Y_val_outer, X_val_outer)
#         loader_Wp_val = torch.utils.data.DataLoader(dataset_Wp_val, batch_size=64, shuffle=False)
        
#         # Training loop
#         for epoch in range(100):  # Inner epochs
#             model_Wp.train()
#             for batch_A, batch_Y, batch_X in loader_Wp:
#                 optimizer_Wp.zero_grad()
#                 outputs = model_Wp(batch_A, batch_X)
#                 loss = criterion_Wp(outputs, batch_Y)
#                 loss.backward()
#                 optimizer_Wp.step()
            
#             # Validation loss
#             model_Wp.eval()
#             val_losses = []
#             with torch.no_grad():
#                 for batch_A_val, batch_Y_val, batch_X_val in loader_Wp_val:
#                     outputs_val = model_Wp(batch_A_val, batch_X_val)
#                     loss_val = criterion_Wp(outputs_val, batch_Y_val)
#                     val_losses.append(loss_val.item())
#             val_loss = np.mean(val_losses)
#             early_stopping_Wp(val_loss, model_Wp)
#             print(f"    Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")
#             if early_stopping_Wp.early_stop:
#                 print("      Early stopping triggered.")
#                 break
        
#         # Load the best model
#         model_Wp.load_state_dict(torch.load(f'best_model_Wp_inner_fold_{fold_outer}.pt'))
        
#         # Predict on validation set
#         model_Wp.eval()
#         with torch.no_grad():
#             logits_Y_val = model_Wp(A_val_outer, X_val_outer)
#             # Compute Wp = logit(E[Y | A, X]) = logits_Y
#             Wp_pred_val = logits_Y_val  # Already in logit scale
#             Wp_pred[val_idx_outer] = Wp_pred_val
    
#     return Wp_pred


def train_nuisance_model(model, optimizer, criterion, A_train, Y_train, X_train, 
                         epochs=100, batch_size=64, binary_A=False, 
                         early_stopping_patience=10, early_stopping_delta=0.0, 
                         early_stopping_verbose=False):
    """
    Train a nuisance parameter model with Early Stopping.
    
    Parameters:
    - model: nn.Module, the model to train
    - optimizer: torch.optim.Optimizer, optimizer
    - criterion: loss function
    - A_train: torch.Tensor, treatment variable (for Y prediction)
    - Y_train: torch.Tensor, outcome variable (for Y prediction)
    - X_train: torch.Tensor, covariates
    - epochs: int, number of training epochs
    - batch_size: int, size of each training batch
    - binary_A: bool, if True, treat A as binary
    - early_stopping_patience: int, number of epochs with no improvement after which training will be stopped
    - early_stopping_delta: float, minimum change in the monitored quantity to qualify as an improvement
    - early_stopping_verbose: bool, if True, prints a message for each validation loss improvement
    """
    model.train()
    
    # Split the training data into training and validation sets (80-20 split)
    dataset = torch.utils.data.TensorDataset(A_train, Y_train, X_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, 
                                   verbose=early_stopping_verbose, 
                                   delta=early_stopping_delta, 
                                   path='best_model.pt')
    
    for epoch in range(epochs):
        model.train()
        for batch_A, batch_Y, batch_X in train_loader:
            optimizer.zero_grad()
            if isinstance(model, NuisanceModelY):
                outputs = model(batch_A, batch_X)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_Y.dim() == 0:
                    batch_Y = batch_Y.unsqueeze(0)
                loss = criterion(outputs, batch_Y)
            elif isinstance(model, NuisanceModelA) or isinstance(model, NuisanceModelAY0) or isinstance(model, NuisanceModelAtemp): #or isinstance(model, NuisanceModelAMulti) or isinstance(model, NuisanceModelAY0Multi) or isinstance(model, NuisanceModelAtempMulti): #TODO change later
                outputs = model(batch_X)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_A.dim() == 0:
                    batch_A = batch_A.unsqueeze(0)
                loss = criterion(outputs, batch_A)  
            elif isinstance(model, NuisanceModelWp_mlp):
                outputs = model(batch_A, batch_X)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_Y.dim() == 0:
                    batch_Y = batch_Y.unsqueeze(0)
                loss = criterion(outputs, batch_Y)  # Y_train here represents Wp targets
            elif isinstance(model, NuisanceModeltnk):
                outputs = model(batch_X)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_Y.dim() == 0:
                    batch_Y = batch_Y.unsqueeze(0)
                loss = criterion(outputs, batch_Y)
            else:
                raise ValueError("Unknown model type.")
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_A, batch_Y, batch_X in val_loader:
                if isinstance(model, NuisanceModelY):
                    outputs = model(batch_A, batch_X)
                    # if outputs and batch_Y are both 1D tensors, BCEWithLogitsLoss expects them to have the same shape
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    loss = criterion(outputs, batch_Y)
                elif isinstance(model, NuisanceModelA) or isinstance(model, NuisanceModelAY0) or isinstance(model, NuisanceModelAtemp): #or isinstance(model, NuisanceModelAMulti) or isinstance(model, NuisanceModelAY0Multi) or isinstance(model, NuisanceModelAtempMulti):
                    outputs = model(batch_X)
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    loss = criterion(outputs, batch_A)
                elif isinstance(model, NuisanceModelWp_mlp):
                    outputs = model(batch_A, batch_X)
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    loss = criterion(outputs, batch_Y)
                elif isinstance(model, NuisanceModeltnk):
                    outputs = model(batch_X)
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    loss = criterion(outputs, batch_Y)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        early_stopping(val_loss, model)
        # if epoch % 50 == 0:
        #     print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.6f}')
        
        if early_stopping.early_stop:
            #print("Early stopping triggered.")
            break
    
    # Load the best model state
    model.load_state_dict(early_stopping.best_model_state)

