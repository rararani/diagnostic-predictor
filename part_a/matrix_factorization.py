from utils import *
from scipy.linalg import sqrtm

import numpy as np


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


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    for j in range(len(train_data["user_id"])):
        i = \
            np.random.choice(len(train_data["question_id"]), 1)[0]

        c = train_data["is_correct"][i]     # C[n,m] (C^(i) in notes)
        n = train_data["user_id"][i]    # can obtain U[n] (U^(i) in notes)
        q = train_data["question_id"][i]    # can obtain Z[q] (Z^(i) in notes)

        u[n] += lr * ((c - np.dot(u[n], z[q])) * z[q])
        z[q] += lr * ((c - np.dot(u[n], z[q])) * u[n])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    mat = None
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
    mat = np.matmul(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    k_values = [1, 5, 10, 15, 18]
    # print("=== Singular Value Decomposition ===")
    # accuracies = []
    # for k in k_values:
    #     mat = svd_reconstruct(train_matrix, k)
    #     acc = sparse_matrix_evaluate(val_data, mat)
    #     print("Validation Accuracy for k={}: {}".format(k, acc))
    #     accuracies.append(acc)
    #
    # opt_k = k_values[np.argmax(accuracies)]
    # print("The optimal k is {} with a validation accuracy of {}".format(opt_k, max(accuracies)))
    # mat = svd_reconstruct(train_matrix, opt_k)
    # test_acc = sparse_matrix_evaluate(test_data, mat)
    # print("Test Accuracy: {}".format(test_acc))

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    print("=== ALS ===")
    # for k in k_values:
    #     matrix = als(train_data, k, 0.01, 10)
    #     acc = sparse_matrix_evaluate(val_data, matrix)
    #     print(acc)

    matrix = als(train_data, 5, 0.01, 10)
    print(sparse_matrix_evaluate(test_data, matrix))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
