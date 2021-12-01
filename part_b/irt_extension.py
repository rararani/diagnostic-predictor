from utils import *

import numpy as np
import matplotlib.pyplot as plt

# Global Variables: Dimensions
N, D = 0, 0
best_val_score = float('-inf')
best_iterations = 0
best_learning_rate = 0


def _load_student_metadata_csv(path, get_student_meta=False):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    if get_student_meta:
        data["data_of_birth"] = []
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                if get_student_meta:
                    data["user_id"].append(int(row[0]))
                    # data["gender"].append(int(row[1]))
                    data["data_of_birth"].append(row[2])
                    # data["premium_pupil"].append(int(row[3]))
                else:
                    data["question_id"].append(int(row[0]))
                    data["user_id"].append(int(row[1]))
                    data["is_correct"].append(int(row[2]))

            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_student_metadata(root_dir="/data"):
    path = os.path.join(root_dir, "student_meta.csv")
    return _load_student_metadata_csv(path, get_student_meta=True)


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    log_lklihood = 0.

    # Goal is to calculate log-likelihood of question j being answered correctly by student i
    for i, c_ij in enumerate(data['is_correct']):
        # From 2a), theta_i is ith student
        theta_i = theta[data['user_id'][i]]
        # From 2b), beta_j is jth question
        beta_j = beta[data['question_id'][i]]

        # TODO: add alpha here
        log_lklihood += (c_ij * (alpha*(theta_i - beta_j))) - np.log(1 + np.exp(alpha*(theta_i - beta_j)))

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, alpha):
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
    :param alpha: Vector
    :return: tuple of vectors
    """
    derivative_wrt_theta = np.zeros(theta.shape[0])
    derivative_wrt_beta = np.zeros(beta.shape[0])
    derivative_wrt_alpha = np.zeros(alpha.shape[0])

    for i, c_ij in enumerate(data['is_correct']):
        # From 2a), theta_i is ith student
        theta_i = theta[data['user_id'][i]]
        # From 2b), beta_j is jth question
        beta_j = beta[data['question_id'][i]]

        alpha_i = alpha[data['question_id'][i]]

        # Use equation from 2a)
        # TODO: update using alpha
        derivative_wrt_theta[data['user_id'][i]] += (c_ij - sigmoid(alpha_i*(theta_i - beta_j)))
        derivative_wrt_beta[data['question_id'][i]] += (sigmoid(alpha_i*(theta_i - beta_j)) - c_ij)
        derivative_wrt_alpha[data['question_id'][i]] += (c_ij * theta_i) - (c_ij * beta_j) - \
                                                        (sigmoid(alpha_i * (theta_i - beta_j)) * (theta_i - beta_j))

    # Update theta and beta for gradient descent
    theta += (lr * derivative_wrt_theta)
    beta += (lr * derivative_wrt_beta)
    alpha += (lr * derivative_wrt_alpha)

    return theta, beta, alpha


def irt(data, val_data, lr, iterations, alt_beta=None, alt_theta=None):
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
    # TODO: initialize alpha vector
    alpha = np.zeros(D)

    # If alt_theta and alt_beta are provided
    if alt_beta and alt_theta:
        theta = np.zeros(alt_theta)
        beta = np.zeros(alt_beta)

    # Store results in these arrays
    val_acc_lst = []
    train_acc_lst = []
    val_negative_logs = []
    train_negative_logs = []

    val_score = 0

    for i in range(iterations):
        # Record training negative log likelihood
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        train_negative_logs.append(train_neg_lld)

        # Record validation negative log likelihood
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha)
        val_negative_logs.append(val_neg_lld)

        # Record training accuracy score
        train_score = evaluate(data=data, theta=theta, beta=beta, alpha=alpha)
        train_acc_lst.append(train_score)

        # Record validation accuracy score
        val_score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
        val_acc_lst.append(val_score)

        print("NLLK: {} \t Score: {}".format(train_neg_lld, val_score))

        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha)

    # If val_score better than global best_val_score, replace
    if val_score > best_val_score:
        best_val_score = val_score
        best_learning_rate = lr
        best_iterations = iterations

    return theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs


def evaluate(data, theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        # TODO: calculate x using alpha
        x = ((alpha[q] * theta[u]) - (alpha[q] * beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def partition_outliers(student_metadata, train_data):
    """
    Split dataset into categories based on student metadata.
    """
    early = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    mid = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    late = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    early_user_id = []
    mid_user_id = []
    late_user_id = []
    # TODO: split user_id into early mid late based on month of birth
    for i, user_id in enumerate(student_metadata["user_id"]):
        dob = student_metadata["data_of_birth"][i]
        if dob:
            birth_month = int(dob[5:7])
            if birth_month <= 4:
                early_user_id.append(user_id)
            elif birth_month <= 8:
                mid_user_id.append(user_id)
            else:
                late_user_id.append(user_id)

    for i, user_id in enumerate(train_data["user_id"]):
        if user_id in early_user_id:
            early["user_id"].append(train_data["user_id"][i])
            early["question_id"].append(train_data["question_id"][i])
            early["is_correct"].append(train_data["is_correct"][i])
        if user_id in mid_user_id:
            mid["user_id"].append(train_data["user_id"][i])
            mid["question_id"].append(train_data["question_id"][i])
            mid["is_correct"].append(train_data["is_correct"][i])
        if user_id in late_user_id:
            late["user_id"].append(train_data["user_id"][i])
            late["question_id"].append(train_data["question_id"][i])
            late["is_correct"].append(train_data["is_correct"][i])
    return early, mid, late


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # TODO: load student metadata
    student_metadata = load_student_metadata("../data")
    early_born_students, mid_born_students, late_born_students = partition_outliers(student_metadata, train_data)

    # Set global variables for dimensions
    global N
    N = sparse_matrix.shape[0]  # i = 0, 1, ..., N
    global D
    D = sparse_matrix.shape[1]  # j = 0, 1, ..., D

    # Uncomment section to find best learning rate and best number of iteration
    # Caution: This takes a while, but the best learning rate is 0.0025 with iterations of 50
    # -----------------------------------------------------------------------------------------------------------------
    # # Learning rate typically small, between 0.01 and 0.0001
    # learning_rates = [0.01, 0.0075, 0.005, 0.0025, 0.001]
    #
    # # Let's try different iterations
    # iterations = [10, 25, 50, 75, 100]
    #
    # for lr in learning_rates:
    #     for iteration in iterations:
    #         print("Using Learning rate: {} for {} iterations".format(lr, iteration))
    #         theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs = irt(train_data, val_data, lr, iteration)
    #         print("Accuracy: {}".format(val_acc_lst[-1]))
    #
    # global best_learning_rate
    # global best_iterations
    # print("The best learning rate is {} and the best number of iterations to use is {}".format(
    #     best_learning_rate,
    #     best_iterations
    # ))
    # -----------------------------------------------------------------------------------------------------------------

    learning_rate = 0.0075
    iterations = 50

    theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs = irt(
        early_born_students,
        val_data,
        learning_rate,
        iterations
    )
    print("Validation Accuracy For Early Born Students: {}".format(val_acc_lst[-1]))

    theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs = irt(
        mid_born_students,
        val_data,
        learning_rate,
        iterations
    )
    print("Validation Accuracy For Mid Born Students: {}".format(val_acc_lst[-1]))

    theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs = irt(
        late_born_students,
        val_data,
        learning_rate,
        iterations
    )
    print("Validation Accuracy For Late Born Students: {}".format(val_acc_lst[-1]))

    # Q2 b) Plot training curve that shows the training and validation log-likelihoods as a function of iteration
    # plt.plot(np.arange(iterations), train_negative_logs, label="Training")
    # plt.plot(np.arange(iterations), val_negative_logs, label="Validation")
    # plt.ylabel("Training and Validation Log Likelihood")
    # plt.xlabel("Iterations")
    # plt.title("Training and Validation Log Likelihoods as a Function of Iterations")
    # plt.legend()
    # plt.show()
    #
    # theta, test_beta, test_val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs = irt(
    #     train_data,
    #     test_data,
    #     learning_rate,
    #     iterations
    # )
    #
    # # Q2 c) Report final test accuracy
    # print("Test Accuracy: {}".format(test_val_acc_lst[-1]))
    #
    # # Q2 d) Select three questions and plot the curves as functions of theta
    # selected_questions = [13, 14, 15]
    # for question in selected_questions:
    #     probabilities = []
    #     q_id = val_data["question_id"][question]
    #     x_axis = [i for i in range(-5, 6)]
    #     for theta in x_axis:
    #         probabilities.append(sigmoid(theta-beta[q_id]))
    #     plt.plot(x_axis, probabilities, label="Question {}".format(str(question)))
    # plt.ylabel("Probability of correctness (p(c_ij))")
    # plt.xlabel("Theta")
    # plt.title("Training and Validation Log Likelihoods as a Function of Iterations")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
