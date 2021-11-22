from utils import *

import numpy as np
import matplotlib.pyplot as plt

# Global Variables: Dimensions
N, D = 0, 0
best_val_score = float('-inf')
best_iterations = 0
best_learning_rate = 0

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.

    # Goal is to calculate log-likelihood of question j being answered correctly by student i
    for i, c_ij in enumerate(data['is_correct']):
        # From 2a), theta_i is ith student
        theta_i = theta[data['user_id'][i]]
        # From 2b), beta_j is jth question
        beta_j = beta[data['question_id'][i]]

        log_lklihood += (c_ij * (theta_i - beta_j)) - np.log(1 + np.exp(theta_i - beta_j))

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    derivative_wrt_theta = np.zeros(theta.shape[0])
    derivative_wrt_beta = np.zeros(beta.shape[0])

    for i, c_ij in enumerate(data['is_correct']):
        # From 2a), theta_i is ith student
        theta_i = theta[data['user_id'][i]]
        # From 2b), beta_j is jth question
        beta_j = beta[data['question_id'][i]]

        # Use equation from 2a)
        derivative_wrt_theta[data['user_id'][i]] += (c_ij - sigmoid(theta_i - beta_j))
        derivative_wrt_beta[data['question_id'][i]] += (sigmoid(theta_i - beta_j) - c_ij)

    # Update theta and beta for gradient descent
    theta += (lr * derivative_wrt_theta)
    beta += (lr * derivative_wrt_beta)

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # Set global scope for these variables since we want to find best learning rate and iterations
    global best_val_score
    global best_iterations
    global best_learning_rate

    # Initialize theta and beta.
    theta = np.zeros(N)
    beta = np.zeros(D)

    # Store results in these arrays
    val_acc_lst = []
    train_acc_lst = []
    val_negative_logs = []
    train_negative_logs = []

    val_score = 0

    for i in range(iterations):
        # Record training negative log likelihood
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_negative_logs.append(train_neg_lld)

        # Record validation negative log likelihood
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_negative_logs.append(val_neg_lld)

        # Record training accuracy score
        train_score = evaluate(data=data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)

        # Record validation accuracy score
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)

        print("NLLK: {} \t Score: {}".format(train_neg_lld, val_score))

        theta, beta = update_theta_beta(data, lr, theta, beta)

    # If val_score better than global best_val_score, replace
    if val_score > best_val_score:
        best_val_score = val_score
        best_learning_rate = lr
        best_iterations = iterations

    return theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Set global variables for dimensions
    global N
    N = sparse_matrix.shape[0]  # i = 0, 1, ..., N
    global D
    D = sparse_matrix.shape[1]  # j = 0, 1, ..., D

    # Uncomment section to find best learning rate and best number of iteration
    # Caution: This takes a while, but the best learning rate is 0.0025 with iterations of 50
    # -----------------------------------------------------------------------------------------------------------------
    # Learning rate typically small, between 0.01 and 0.0001
    # learning_rates = [0.01, 0.0075, 0.005, 0.0025, 0.001]

    # Let's try different iterations
    # iterations = [10, 25, 50, 75, 100]

    # for lr in learning_rates:
    #     for iteration in iterations:
    #         print("Using Learning rate: {} for {} iterations".format(lr, iteration))
    #         theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs = irt(train_data, val_data, lr, iteration)
    #
    # global best_learning_rate
    # global best_iterations
    # print("The best learning rate is {} and the best number of iterations to use is {}".format(
    #     best_learning_rate,
    #     best_iterations
    # ))
    # -----------------------------------------------------------------------------------------------------------------

    learning_rate = 0.0025
    iterations = 50

    theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs = irt(
        train_data,
        val_data,
        learning_rate,
        iterations
    )
    # Q2 c) Report final validation accuracy
    print("Validation Accuracy: {}".format(val_acc_lst[-1]))

    # Q2 b) Plot training curve that shows the training and validation log-likelihoods as a function of iteration
    plt.plot(np.arange(iterations), train_negative_logs, label="Training")
    plt.plot(np.arange(iterations), val_negative_logs, label="Validation")
    plt.ylabel("Training and Validation Log Likelihood")
    plt.xlabel("Iterations")
    plt.title("Training and Validation Log Likelihoods as a Function of Iterations")
    plt.legend()
    plt.show()

    theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs = irt(
        train_data,
        test_data,
        learning_rate,
        iterations
    )

    # Q2 c) Report final test accuracy
    print("Test Accuracy: {}".format(val_acc_lst[-1]))

    # Q2 d)

if __name__ == "__main__":
    main()
