import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import KFold

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
        inputs = torch.cat([A.unsqueeze(1), X], dim=1)
        return torch.sigmoid(self.network(inputs).squeeze())
    
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

class NuisanceModelWp(nn.Module):
    """
    Model to estimate Wp = logit(E[Y | A, X]).
    """
    def __init__(self, input_dim):
        super(NuisanceModelWp, self).__init__()
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
        inputs = torch.cat([A.unsqueeze(1), X], dim=1)
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
            elif isinstance(model, NuisanceModelA) or isinstance(model, NuisanceModelAY0) or isinstance(model, NuisanceModelAtemp):
                outputs = model(batch_X)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_A.dim() == 0:
                    batch_A = batch_A.unsqueeze(0)
                loss = criterion(outputs, batch_A)  
            elif isinstance(model, NuisanceModelWp):
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
                elif isinstance(model, NuisanceModelA) or isinstance(model, NuisanceModelAY0) or isinstance(model, NuisanceModelAtemp):
                    outputs = model(batch_X)
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    loss = criterion(outputs, batch_A)
                elif isinstance(model, NuisanceModelWp):
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

