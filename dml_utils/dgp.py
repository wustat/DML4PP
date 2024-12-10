import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def r0(X, linear = False):
    """
    Compute the function r0(X).
    
    Parameters:
    - X: torch.Tensor, shape (n_samples, p)
    
    Returns:
    - Z: torch.Tensor, shape (n_samples,)
    """
    term1 = 0.1 * (X[:, 0] * X[:, 1] * X[:, 2] + X[:, 3] * X[:, 4] + X[:, 5] ** 3)
    term2 = 0.5 * (-torch.sin(X[:, 6]) ** 2 + torch.cos(X[:, 7]))
    term3 = 1 / (1 + X[:, 8] ** 2)
    term4 = -1 / (1 + torch.exp(X[:, 9]))
    term5 = 0.25 * ((X[:, 10] > 0).float() - (X[:, 11] > 0).float())
    
    Z = term1 + term2 + term3 + term4 + term5
    return Z if not linear else torch.sum(X, dim = 1)

def A0(X, linear = False):
    """
    Compute the function A0(X).
    
    Parameters:
    - X: torch.Tensor, shape (n_samples, p)
    
    Returns:
    - EA: torch.Tensor, shape (n_samples,)
    """
    term1 = 1 / (1 + torch.exp(X[:, 0])) - 1 / (1 + torch.exp(X[:, 1]))
    term2 = 0.5 * (torch.sin(X[:, 2]) + torch.cos(X[:, 3]))
    term3 = 0.25 * ((X[:, 4] > 0).float() - (X[:, 5] > 0).float())
    term4 = 0.1 * (X[:, 6] * X[:, 7] + X[:, 8] * X[:, 9])
    
    EA = term1 + term2 + term3 + term4
    return EA if not linear else torch.sum(X, dim = 1)

def GD(beta, n=1000, p=20, Sigma=None, binary_A=False, random_state=42, linear = False):
    """
    Generate synthetic data for causal inference.
    
    Parameters:
    - beta: float, coefficient for A in the outcome model
    - n: int, number of samples (default: 1000)
    - p: int, number of covariates/features (default: 20)
    - Sigma: torch.Tensor or None, covariance matrix of X (default: 0.2 * ones with diagonal 1)
    - binary_A: bool, if True, make treatment A binary
    - random_state: int or None, for reproducibility
    
    Returns:
    - X: torch.Tensor, shape (n, p), covariates
    - Y: torch.Tensor, shape (n,), binary outcomes
    - A: torch.Tensor, shape (n,), treatments (binary or continuous)
    """
    torch.manual_seed(random_state)
    
    if Sigma is None:
        Sigma = 0.2 * torch.ones((p, p), device=device)
        Sigma.fill_diagonal_(torch.tensor(1.0, device=device))
    else:
        Sigma = Sigma.to(device)
        if Sigma.shape != (p, p):
            raise ValueError("Sigma must be a square matrix with shape (p, p).")
    
    # Generate X ~ MVN(0, Sigma)
    X = torch.distributions.MultivariateNormal(
        loc=torch.zeros(p, device=device),
        covariance_matrix=Sigma
    ).sample((n,))
    
    # Truncate X to [-2, 2]
    X = torch.clamp(X, -2, 2)
    
    # Compute EA = A0(X)
    EA = A0(X, linear)
    # Generate err ~ N(0,1), truncate to [-2, 2]
    err = torch.randn(n, device=device)
    err = torch.clamp(err, -2, 2)
    if binary_A:
        # Binary Treatment
        # Use sigmoid to map EA + noise to probability, then sample Bernoulli
        # Alternatively, threshold at 0 to make A binary
        # Here, we use sigmoid and Bernoulli
        prob_A = torch.sigmoid(EA + err)
        A = torch.bernoulli(prob_A)
    else:
        # Continuous Treatment
        A = EA + err
    
    # Compute ErX = r0(X)
    ErX = r0(X, linear)
    
    # Compute P = expit(beta * A + ErX)
    linear_combination = beta * A + ErX
    P = torch.sigmoid(linear_combination)
    
    # Generate Y ~ Bernoulli(P)
    Y = torch.bernoulli(P)
    
    return X, Y, A


