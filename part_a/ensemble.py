from part_a.item_response import irt
from utils import *


def bootstrap_new_datasets(train_data, m: int):
    """
    Returns m new datasets from train dataset sampled with replacement
    """
    n = len(train_data["user_id"])
    new_datasets = []
    for _ in range(m):
        # Randomly choose an index, replace=True since we want to sample with replacement
        i = np.random.choice(n, n, replace=True)
        new_dataset = {"user_id": (np.array(train_data["user_id"])[i]).tolist(),
                    "question_id": (np.array(train_data["question_id"])[i]).tolist(),
                    "is_correct": (np.array(train_data["is_correct"])[i]).tolist()}
        new_datasets.append(new_dataset)
    return new_datasets


def predict_using_irt(new_train_datasets, val_data, test_data):
    val_accuracies = []
    test_accuracies = []

    # These were found to be best from Q2
    learning_rate = 0.0025
    iterations = 50

    # for each bootstrapped dataset, predict accuracy using irt
    for i in range(len(new_train_datasets)):
        train_data = new_train_datasets[i]

        theta, beta, val_acc_lst, train_acc_lst, val_negative_logs, train_negative_logs = irt(
            train_data,
            val_data,
            learning_rate,
            iterations,
            alt_theta=542,  # From sparse matrix size i.e. N
            alt_beta=1774  # From sparse matrix size i.e. D
        )
        val_accuracies += val_acc_lst

        theta, beta, test_acc_lst, train_acc_lst, test_negative_logs, train_negative_logs = irt(
            train_data,
            test_data,
            learning_rate,
            iterations,
            alt_theta=542,  # From sparse matrix size i.e. N
            alt_beta=1774  # From sparse matrix size i.e. D
        )
        test_accuracies += test_acc_lst

    return val_accuracies, test_accuracies


def main():
    # For reproducible effects
    np.random.seed(13)

    # This our dataset D with n examples
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # First, bootstrap from data to create m new datasets
    # i.e. sample with replacement n training examples from D with replacement
    m = 3
    new_train_datasets = bootstrap_new_datasets(train_data, m)

    # Next, predict accuracy on m new datasets
    predicted_val_accuracies, predicted_test_accuracies = predict_using_irt(new_train_datasets, val_data, test_data)

    # Finally, report average predictions of models
    final_val_accuracy = sum(predicted_val_accuracies) / len(predicted_val_accuracies)
    final_test_accuracy = sum(predicted_test_accuracies) / len(predicted_test_accuracies)
    print("Final validation accuracy: {}".format(final_val_accuracy))
    print("Final test accuracy: {}".format(final_test_accuracy))


if __name__ == '__main__':
    main()
