# TODO.Q1 Put any functions you write for Question 1 here.
import torch


# For question 2, it would be helpful to define a function that returns the set of pruned gradients
# def robust_aggregator(gradients):
#     # Run pruning procedure
#     pruned_gradients = gradients
#     return pruned_gradients
#

def robust_aggregator(gradients,
                      eps_threshold=9 * 39275):  # hardcoding the eps threshold to k* SIGMA provided in question
    # Clone the original gradients to avoid modifying the input
    gradients = gradients.clone()

    while True:
        # Calculate the covariance matrix
        cov = calculate_covariance_matrix(gradients)
        max_cov = cov.max()

        # Determine the maximum covariance and its direction
        lambdas, U = torch.linalg.eig(cov)
        max_var_eigen_vector = U[0]
        mean_gradient = gradients.mean(dim=0)

        if max_cov >= eps_threshold:
            # Project the distance of every gradient wrt mean gradient along the max variance direction
            diff_gradient = gradients - mean_gradient
            projections_on_max_var = diff_gradient @ max_var_eigen_vector

            # Find the index of the gradient with the maximum absolute distance
            outlier_index = torch.abs(projections_on_max_var).argmax()

            # Remove the outlier gradient
            gradients = torch.cat((gradients[:outlier_index], gradients[outlier_index + 1:]), dim=0)
        else:
            # can stop here since we have already pruned it
            print(f"Pruned gradients now has shape of {gradients.shape}")
            break

    # return the pruned gradients
    return gradients


def calculate_covariance_matrix(X):
    ''' This calculates the covariance matrix in the same way prescribed in the slides without using a direct cov function present in torch'''
    X = X.clone()
    n = X.shape[0]
    mu = X.mean(dim=0, keepdim=True)  # Shape (1, d)
    X = X - mu  # Shape (n, d) - centered along the mean
    cov_matrix = (X.T @ X) / (n - 1)  # cov matrix of shape d,d
    return cov_matrix


if __name__ == '__main__':
    # Setup
    torch.set_printoptions(precision=10, sci_mode=True)

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
    print(f"max covariance is {max_cov}")
    max_cov_ind = cov.argmax()

    # step2: Convert the linear index to 2D coordinates (i, j)
    i, j = divmod(max_cov_ind.item(), d)
    print(f"max cov index is {i}, {j}")

    # step3: what is the direction of the max variance (unit vector along max eigen vector), since cov is positive semi definite no need to sort as the first eigen value is already max
    lambdas, U = torch.linalg.eig(cov)
    max_var_eigen_vector = U[0]
    norm = torch.norm(max_var_eigen_vector, p=2)
    max_var_direction = max_var_eigen_vector / norm  # shape (d,)
    print(f"max variance eigen vector is {max_var_direction}")
    print(f"first 3 values of max var direction is [{max_var_direction[:3]}]")
    print(f"last 3 values of max var direction is [{max_var_direction[-3:]}]")
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
    projections_on_max_var = diff_gradient @ max_var_direction  # shape (n,)

    # step3: Find the index of the gradient with the maximum absolute distance
    outlier_index = torch.abs(projections_on_max_var).argmax()
    outlier_gradient = gradients[outlier_index]

    print(f"The most likely outlier gradient index is: {outlier_index}")
    print(f"First 3 values of the outlier gradient: {outlier_gradient[:3]}")
    print(f"Last 3 values of the outlier gradient: {outlier_gradient[-3:]}")
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
    print(f"New maximum covariance among remaining gradients is: {new_max_cov}")

    # Step 4: Calculate the change in maximum variance
    variance_change = new_max_cov - max_cov
    print(f"Change in maximum variance: {variance_change}")
    print("-" * 60)
    print()

    # ---------------------- Q 2.1.4 ----------------------
    '''Repeat this procedure until max variance of the pruned set of gradients is below threshold of E. How many poisoned gradients did you detect? 
        Suppose I told you there were 673 poisoned gradients, what percentage did you detect?'''
    pruned_gradients = robust_aggregator(gradients)
    num_poisoned = gradients.shape[0] - pruned_gradients.shape[0]
    print(f"percentage of poisoned gradients: {(num_poisoned/673) * 100} % ")
