import torch
from tqdm import tqdm

# For question 2, it would be helpful to define a function that returns the set of pruned gradients
# def robust_aggregator(gradients):
#     # Run pruning procedure
#     pruned_gradients = gradients
#     return pruned_gradients
#

def robust_aggregator(gradients,
                      eps_threshold=9 * 39275,
                      show_progress=False):  # hardcoding the eps threshold to k* SIGMA provided in question
    # Clone the original gradients to avoid modifying the input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gradients = gradients.clone()
    n = gradients.shape[0]

    pbar = tqdm(total=n) if show_progress else None

    for i in range(n):
        # Calculate the covariance matrix
        cov = calculate_covariance_matrix(gradients)

        # old approach just using the max cov value
        # max_cov = cov.max()

        # Determine the maximum covariance and its direction
        lambdas, U = torch.linalg.eig(cov)
        spectral_norm = torch.abs(lambdas).max().item() # taking abs because the eigen value can be complex

        # old approach: no need to really sort anything here as we just pick the largest already
        # # Sort lambdas and U accordingly ( typically it is not needed, but the EVD had complex entries so this is just a precaution )
        # _, sorted_indices = torch.sort(torch.abs(lambdas), descending=True)
        # lambdas = lambdas[sorted_indices]
        # U = U[:, sorted_indices]
        #
        # # need to compare with the spectral norm which corresponds to the largest eigen value which is in the first position (based on the provided algorithm )
        # spectral_norm = torch.abs(lambdas[0]).item() # taking abs because the eigen value can be complex

        if spectral_norm > eps_threshold:
            mean_gradient = gradients.mean(dim=0)

            # Project the distance of every gradient wrt mean gradient along the max variance direction
            diff_gradient = gradients - mean_gradient
            diff_gradient = diff_gradient.to(torch.complex64)
            max_var_eigen_vector = U[0]
            projections_on_max_var = diff_gradient.to(device) @ max_var_eigen_vector.to(device)

            # Find the index of the gradient with the maximum absolute distance
            outlier_index = torch.abs(projections_on_max_var).argmax()

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
    mu = X.mean(dim=0)  # Shape (d,)
    X = X - mu  # Shape (n, d) - centered along the mean

    # approach 1. possibly incorrect as this will make X[i]
    # cov_matrix = (X.T @ X) / (n - 1)  # cov matrix of shape d,d

    # approach 2: the dot product is calculated for each X[i] wrt itself and then summed and divided as opposed to approach 1 since there are differences
    cov_matrix = torch.zeros((d,d)).to(device)
    for i in range(n):
        x = X[i].unsqueeze(0) # shape (1,d)
        xt = x.transpose(0,1) # shape (d,1)
        cov_matrix += xt @ x # shape (d,d)

    cov_matrix /= n-1

    # approach 3: matrix form of doing approach 2 (but still not getting the exact same answer) and was slow on cpu
    # X_centered_expanded = X.unsqueeze(1)  # Shape (n, 1, d)
    # outer_products = X_centered_expanded.transpose(1, 2) @ X_centered_expanded # Shape (n, d, d)
    # cov_matrix_other = outer_products.sum(dim=0) / (n - 1)  # Shape (d, d)

    # is_psd = is_matrix_psd(cov_matrix)
    return cov_matrix

def is_matrix_psd(X):
    eigenvalues, eigenvectors = torch.linalg.eig(X)
    return torch.all(eigenvalues.real >= -1e-10).item()

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

    benign_var = 39275  # spectral norm of the covariance matrix (i.e sqrt(SIGMA) )
    k = 9
    eps_threshold = k * benign_var

    # ---------------------- Q 2.1.1 ----------------------
    '''
    Compute the covariance matrix Σ. What is the maximum variance? What is the direction of maximum variance (giving the first 3 and last 3 values is sufficient)
    '''
    print("Part 2: Question 1 =>")
    cov = calculate_covariance_matrix(gradients)

    # step1: max value of covariance
    max_cov = cov.max()
    print(f"max covariance is {max_cov}") # 39616.3515625
    max_cov_ind = cov.argmax()

    # step2: Convert the linear index to 2D coordinates (i, j)
    i, j = divmod(max_cov_ind.item(), d)
    print(f"max cov index is {i}, {j}") #296,296

    # step3: what is the direction of the max variance (unit vector along max eigen vector), since cov is positive semi definite no need to sort as the first eigen value is already max

    lambdas, U = torch.linalg.eig(cov)

    # Get the indices that would sort lambdas by their absolute values in descending order (TODO.x review the PSD issue with covariance matrix)
    _, sorted_indices = torch.sort(torch.abs(lambdas), descending=True)

    # Sort lambdas and U accordingly ( typically it is not needed, but the EVD had complex entries so this is just a precaution )
    lambdas = lambdas[sorted_indices]
    U = U[:, sorted_indices]
    max_var_eigen_vector = U[0]
    norm = torch.norm(max_var_eigen_vector, p=2)
    max_var_direction = max_var_eigen_vector / norm  # shape (d,)
    print(f"max variance eigen vector is {max_var_direction}")
    print(f"first 3 values of max var direction is [{max_var_direction[:3]}]") #[tensor([-3.6010164768e-02+0.j, 3.4746624529e-02+0.j, 2.3515963927e-02+0.j],device='cuda:0')]
    print(f"last 3 values of max var direction is [{max_var_direction[-3:]}]") #[tensor([ 1.2595599230e-07+0.j,  3.4827093032e-07+0.j, -9.0192429525e-08+0.j],device='cuda:0')]
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
    diff_gradient = diff_gradient.to(
        torch.complex64)  # making it a complex tensor because the max var direction contains imaginary components shape (n,d)
    projections_on_max_var = diff_gradient.to(device) @ max_var_direction.to(device)  # shape (n,)

    # step3: Find the index of the gradient with the maximum absolute distance
    outlier_index = torch.abs(projections_on_max_var).argmax()
    outlier_gradient = gradients[outlier_index]

    print(f"The most likely outlier gradient index is: {outlier_index}") #872
    print(f"First 3 values of the outlier gradient: {outlier_gradient[:3]}") #tensor([-0.0000000000e+00, 2.4866737425e-02, -1.1044409275e+00])
    print(f"Last 3 values of the outlier gradient: {outlier_gradient[-3:]}") #tensor([1.0991185904e+00, -2.6810422540e-01, 7.0607316494e-01])
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
    new_max_cov = new_cov.max()
    print(f"New maximum covariance among remaining gradients is: {new_max_cov}") # 39492.1875

    # Step 4: Calculate the change in maximum variance
    variance_change = new_max_cov - max_cov
    print(f"Change in maximum variance: {variance_change}") # -124.1640625 , it has decreased from old value
    print("-" * 60)
    print()

    # ---------------------- Q 2.1.4 ----------------------
    '''Repeat this procedure until max variance of the pruned set of gradients is below threshold of E. How many poisoned gradients did you detect? 
        Suppose I told you there were 673 poisoned gradients, what percentage did you detect?'''
    print("Part 2: Question 4 =>")

    pruned_gradients = robust_aggregator(gradients, show_progress=True)
    num_poisoned = gradients.shape[0] - pruned_gradients.shape[0]
    print(f"percentage of poisoned gradients: {(num_poisoned / 673) * 100} % ") # got 82.76374442793461% (afer 18 mins) i.e. 557/673
