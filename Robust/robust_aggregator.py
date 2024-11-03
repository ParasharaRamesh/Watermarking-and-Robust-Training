import random

import torch
from tqdm import tqdm


def filtering_algorithm(gradients, eps_threshold=9 * 39275, show_progress=False):
    """
    Implements the filtering algorithm to update the set Y by removing or reducing outliers.

    Parameters:
    - gradients (torch.Tensor): The input tensor of shape (n, d), where n is the number of vectors.
    - eps_threshold (float): The threshold value for filtering (not used directly in this implementation).
    - show_progress (bool): Whether to display a progress bar for the filtering process.

    Returns:
    - updated_Y (torch.Tensor): The filtered set Y with updated weights applied to each vector.
    """
    # Step 1: Initialize equal weights c_i for each vector
    n, d = gradients.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gradients = gradients.to(device)
    c = torch.ones(n, device=device) / n  # Initialize equal weights, c_i = 1/n for all i

    pbar = tqdm(total=n) if show_progress else None

    for i in range(n):
        # Step 2: Compute the weighted mean µ_c of Y
        mu_c = (c.unsqueeze(1) * gradients).sum(dim=0)

        # Step 3: Calculate the covariance matrix of Y and find the largest eigenvector
        # This is Cov(Y) = Σ_i c_i * (y_i - µ_c) * (y_i - µ_c)^T
        diff_Y = gradients - mu_c  # Centering Y by subtracting the weighted mean
        Cov_Y = calculate_covariance_matrix(diff_Y)  # Weighted covariance matrix
        Cov_Y += torch.eye(d, device=device) * 1e-7  # Regularization to ensure numerical stability
        lambdas, eigenvectors = torch.linalg.eigh(Cov_Y)  # Get eigenvectors (sorted in ascending order)
        v = eigenvectors[:, -1]  # Largest eigenvector (corresponds to max variance)
        spectral_norm = lambdas[-1].item()

        should_prune = spectral_norm > eps_threshold
        if should_prune:
            # Step 4: Compute τ_i for each vector y_i, τ_i = ⟨y_i - µ_c, v⟩^2
            tau = (diff_Y @ v).pow(2)  # Square the projections to get τ_i

            # Step 5: Update weights based on τ_i, c_i := c_i * (1 - τ_i / τ_max)
            tau_max = tau.max()  # Get the maximum τ_i value
            c = c * (1 - (tau / tau_max))  # Adjust each weight based on its τ_i value

            # Step 6: Count non-zero weights and filter out zero-weight vectors
            nonzero_mask = c != 0  # Identify vectors with non-zero weights
            cnt = nonzero_mask.sum().item()  # Count how many non-zero weights there are
            gradients = gradients[nonzero_mask]  # Retain only the vectors with non-zero weights
            c = c[nonzero_mask]  # Retain the corresponding weights

            # Step 7: Normalize the new weights so that their sum equals 1
            c /= c.sum()

            # Step 8: Apply weights to each vector in the filtered set to form the final output
            gradients = c.unsqueeze(1) * gradients  # Apply the weights to each vector in Y
        else:
            break

        if show_progress and pbar:
            # Update the description of the tqdm bar
            pbar.set_description(f"pruned gradient shape: {gradients.shape}")

            # Update the progress bar
            pbar.update(1)

    return gradients


def robust_aggregator(gradients,
                      eps_threshold=9 * 39275,
                      show_progress=False,
                      use_filtering_algorithm=False
                      ):  # hardcoding the eps threshold to k* SIGMA provided in question
    # Clone the original gradients to avoid modifying the input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gradients.to(device)

    # use the filtering algorithm mentioned in the paper instead
    if use_filtering_algorithm:
        return filtering_algorithm(gradients, eps_threshold, show_progress)

    n = gradients.shape[0]

    pbar = tqdm(total=n) if show_progress else None

    for i in range(n):
        # Calculate the covariance matrix
        cov = calculate_covariance_matrix(gradients)

        # hack to ensure numerical stability of eigh
        torch.backends.cuda.preferred_linalg_library('cusolver')

        # Tikhonov regularization to prevent problems with eigh not converging in CUDA
        cov += torch.eye(cov.size(0), device=cov.device) * 1e-8
        lambdas, U = torch.linalg.eigh(cov)
        spectral_norm = lambdas[-1].item()

        max_var_eigen_vector = U[:, -1]
        norm = torch.norm(max_var_eigen_vector)
        max_var_direction = max_var_eigen_vector / norm

        should_prune = spectral_norm > eps_threshold
        if should_prune:
            mean_gradient = gradients.mean(dim=0)

            # Project the distance of every gradient wrt mean gradient along the max variance direction
            diff_gradient = gradients - mean_gradient
            projections_on_max_var = diff_gradient.to(device) @ max_var_direction.to(device)  # shape (n,)

            # Find the index of the gradient with the maximum absolute distance
            outlier_index = projections_on_max_var.argmax()

            # Remove the outlier gradient
            gradients = torch.cat((gradients[:outlier_index], gradients[outlier_index + 1:]), dim=0)
        else:
            # can stop here since we have already pruned it
            # print(f"Pruned gradients now has shape of {gradients.shape} after iteration {iter}")
            break

        if show_progress and pbar:
            # Update the description of the tqdm bar
            pbar.set_description(f"pruned gradient shape: {gradients.shape}")

            # Update the progress bar
            pbar.update(1)

    # return the pruned gradients
    return gradients


def calculate_covariance_matrix(X):
    ''' This calculates the covariance matrix in the same way prescribed in the slides without using a direct cov function present in torch'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    n = X.shape[0]
    d = X.shape[1]

    # approach 0; use torch's internal cov function
    cov_matrix = torch.cov(X.T)

    # approach 1. possibly incorrect as this will make X[i]
    # mu = X.mean(dim=0)  # Shape (d,)
    # X = X - mu  # Shape (n, d) - centered along the mean
    # cov_matrix = (X.T @ X) / (n - 1)  # cov matrix of shape d,d

    # approach 2: the dot product is calculated for each X[i] wrt itself and then summed and divided as opposed to approach 1 since there are differences
    # cov_matrix = torch.zeros((d,d)).to(device)
    # for i in range(n):
    #     x = X[i].unsqueeze(0) # shape (1,d)
    #     xt = x.transpose(0,1) # shape (d,1)
    #     cov_matrix += xt @ x # shape (d,d)
    #
    # cov_matrix /= n-1

    # approach 3: matrix form of doing approach 2 and was slow on cpu
    # X_centered_expanded = X.unsqueeze(1)  # Shape (n, 1, d)
    # outer_products = X_centered_expanded.transpose(1, 2) @ X_centered_expanded # Shape (n, d, d)
    # cov_matrix_other = outer_products.sum(dim=0) / (n - 1)  # Shape (d, d)

    # is_psd = is_matrix_psd(cov_matrix)
    return cov_matrix


def is_matrix_psd(X):
    eigenvalues, eigenvectors = torch.linalg.eigh(X)
    return torch.all(eigenvalues.max_cov >= -1e-10).item()


if __name__ == '__main__':
    # Setup
    torch.set_printoptions(precision=10, sci_mode=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # shape is (1545, 2, 768)
    gradients = torch.load("all_gradients.pt")

    # reshape to (1545, 1536)
    gradients = gradients.view(gradients.size(0), -1)
    n = gradients.shape[0]
    d = gradients.shape[1]

    benign_var = 39275
    k = 9
    eps_threshold = k * benign_var

    # ---------------------- Q 2.1.1 ----------------------
    '''
    Compute the covariance matrix Σ. What is the maximum variance? What is the direction of maximum variance (giving the first 3 and last 3 values is sufficient)
    '''
    print("Part 2: Question 1 =>")
    cov = calculate_covariance_matrix(gradients)

    # NOTE: this does it in ascending order, meaning the last one is the highest lambda
    lambdas, U = torch.linalg.eigh(cov)

    max_cov = lambdas[-1]  # max variance

    max_var_eigen_vector = U[:, -1]
    norm = torch.norm(max_var_eigen_vector)

    max_var_direction = max_var_eigen_vector / norm  # shape (d,) unit vector

    print(f"max variance eigen vector is {max_cov}")  # 19781812
    print(
        f"first 3 values of max var direction is [{max_var_direction[:3]}]")  # [2.5262700394e-02, 1.0088919662e-02, 3.7055097520e-02]
    print(
        f"last 3 values of max var direction is [{max_var_direction[-3:]}]")  # [-3.7027940154e-02, -1.1205702089e-02, 4.4668824412e-03]
    print("-" * 60)
    print()

    # ---------------------- Q 2.1.2 ----------------------
    '''
    Compute the absolute distance of every gradient vector to the mean, with respect to the direction of max variance. Which gradient seems the most likely to be an outlier?
    Report the index number of this gradient (an integer in [0, 1544]), and give the first 3 and last 3 values of the gradient vector.
    '''
    print("Part 2: Question 2 =>")
    # step1: finding the mean gradient
    mean_gradient = gradients.mean(dim=0)  # shape (d,)

    # step2: projecting the distance of every gradient wrt mean gradient along the max var direction
    diff_gradient = gradients - mean_gradient  # shape (n,d)
    projections_on_max_var = diff_gradient.to(device) @ max_var_direction.to(device)  # shape (n,)

    # step3: Find the index of the gradient with the maximum absolute distance
    outlier_index = projections_on_max_var.argmax()
    outlier_gradient = gradients[outlier_index]

    print(f"The most likely outlier gradient index is: {outlier_index}")  # 872
    print(
        f"First 3 values of the outlier gradient: {outlier_gradient[:3]}")  # [-0.0000000000e+00, 2.4866737425e-02, -1.1044409275e+00]
    print(
        f"Last 3 values of the outlier gradient: {outlier_gradient[-3:]}")  # [1.0991185904e+00, -2.6810422540e-01, 7.0607316494e-01]
    print("-" * 60)
    print()

    # ---------------------- Q 2.1.3 ----------------------
    '''Remove the gradient vector that seems most likely to be an outlier. Compute and report the maximum variance among the remaining gradient vectors. 
    By how much did the maximum variance change? '''
    print("Part 2: Question 3 =>")
    # Step 1: Remove the outlier gradient
    remaining_gradients = torch.cat((gradients[:outlier_index], gradients[outlier_index + 1:]), dim=0)  # Shape (n-1, d)

    # Step 2: Compute the covariance matrix of the remaining gradients
    new_cov = calculate_covariance_matrix(remaining_gradients)

    # Step 3: Determine the maximum variance from the new covariance matrix
    new_lambdas, new_U = torch.linalg.eigh(new_cov)

    new_max_cov = new_lambdas[-1]  # max variance
    print(f"New maximum covariance among remaining gradients is: {new_max_cov}")

    # Step 4: Calculate the change in maximum variance
    variance_change = new_max_cov - max_cov
    print(
        f"Change in maximum variance: {variance_change}")  # -64558.0 (i.e. new variance is smaller than the old max variance)
    print("-" * 60)
    print()

    # ---------------------- Q 2.1.4 ----------------------
    '''Repeat this procedure until max variance of the pruned set of gradients is below threshold of E. How many poisoned gradients did you detect? 
        Suppose I told you there were 673 poisoned gradients, what percentage did you detect?'''
    print("Part 2: Question 4 =>")

    pruned_gradients = robust_aggregator(gradients, show_progress=True)
    num_poisoned = gradients.shape[0] - pruned_gradients.shape[0]
    print(
        f"percentage of poisoned gradients: {(num_poisoned / 673) * 100} % ")  # got 82.76374442793461% (afer 18 mins) i.e. 557/673
