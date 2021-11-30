from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z, n, reg):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :para n: # of data points to sample
    :return: (u, z)
    """
    # Randomly select a pair (user_id, question_id).
    for k in range(n):
        i = \
            np.random.choice(len(train_data["question_id"]), 1)[0]

        c = train_data["is_correct"][i]     # C_nm
        n = train_data["user_id"][i]        # n for u_n
        q = train_data["question_id"][i]    # m for z_m but labelled q

        error = c - np.dot(u[n], z[q])
        u[n] = u[n] - lr * (error * -z[q] + reg * u[n])
        z[q] = z[q] - lr * (error * -u[n] + reg * z[q])

    return u, z


def als(train_data, k, lr, num_iteration, n, reg):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param n: # of data points to sample
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z, n, reg)

    mat = np.matmul(u, z.T)
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # print("=== Singular Value Decomposition ===")
    k_values = [1, 6, 11, 16, 21, 26]
    # accuracies = []
    # for k in k_values:
    #     matrix = svd_reconstruct(train_matrix, k)
    #     acc = sparse_matrix_evaluate(val_data, matrix)
    #     accuracies.append(acc)
    #     print(f"k = {k}: Validation accuracy = {acc}")
    #
    # opt_k = k_values[np.argmax(accuracies)]
    # test_acc = sparse_matrix_evaluate(test_data, svd_reconstruct(train_matrix, opt_k))
    # print(f"The optimal k is {opt_k} with a validation accuracy of {max(accuracies)} and test accuracy of {test_acc}")

    print("=== ALS ==")

    num_iterations = 10
    learning_rate = 0.01
    n = int(len(train_data["question_id"]) * 0.7)
    reg = 0.025
    accuracies = []
    for k in k_values:
        prediction = als(train_data, k, learning_rate, num_iterations, n, reg)
        acc = sparse_matrix_evaluate(val_data, prediction)
        accuracies.append(acc)
        print(f"k = {k}: Validation accuracy = {acc}")

    opt_k = k_values[np.argmax(accuracies)]
    test_acc = sparse_matrix_evaluate(test_data, als(train_data, opt_k, learning_rate, num_iterations, n, reg))
    print(f"The optimal k is {opt_k} with a validation accuracy of {max(accuracies)} and a test accuracy of {test_acc}")

    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(opt_k),
                          size=(len(set(train_data["user_id"])), opt_k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(opt_k),
                          size=(len(set(train_data["question_id"])), opt_k))

    train_losses = []
    val_losses = []
    for i in range(num_iterations):
        u, z = update_u_z(train_data, learning_rate, u, z, n, reg)
        train_loss = squared_error_loss(train_data, u, z) / len(train_data["question_id"])
        train_losses.append(train_loss)
        val_loss = squared_error_loss(val_data, u, z) / len(val_data["question_id"])
        val_losses.append(val_loss)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.xlabel("# iterations")
    plt.ylabel("Squared Error Loss")
    plt.title("Squared Error Loss vs. Iterations")
    plt.show()


if __name__ == "__main__":
    main()
