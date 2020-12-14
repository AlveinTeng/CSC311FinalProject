from utils import *
from knn import knn_impute_by_user
from knn import knn_impute_by_item
import matplotlib.pyplot as plt
import numpy as np
import random


def resample(sparse_matrix, M):
    """
    Resample M samples from sparse_matrix
    :param sparse_matrix:
    :param M:
    :return: list of samples
    """
    results = []
    N = len(sparse_matrix)
    for i in range(M):
        random_indices = random.sample(range(N), N)
        resampled = np.zeros(sparse_matrix.shape)
        for j in range(len(random_indices)):
            new_index = random_indices[j]
            resampled[j] += sparse_matrix[new_index]
        results.append(resampled)
    return results


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # number of sample
    N = len(sparse_matrix)
    M = 3
    k_value = [1, 6, 11, 16, 21, 26]
    results_user = []
    results_item = []
    for k in k_value:
        models = resample(sparse_matrix, M)
        correct_user = 0
        correct_item = 0
        for m in range(len(models)):
            correct_user += np.round(knn_impute_by_user(models[m], val_data, k) * N)
            correct_item += np.round(knn_impute_by_item(models[m], val_data, k) * N)
        results_user.append(correct_user / (3*N))
        results_item.append(correct_item / (3*N))

    plt.plot(k_value, results_user, color="red", label='student')
    plt.plot(k_value, results_item, color='blue', linestyle='--', label='item')
    plt.xlabel('k value')
    plt.ylabel('validation accuracy')
    plt.title('validation accuracy vs k value')
    plt.legend(loc='lower right')
    plt.show()

    # Test performance
    user_based = []
    item_based = []
    for k in k_value:
        user_based.append(knn_impute_by_user(sparse_matrix, test_data, k))
        item_based.append(knn_impute_by_item(sparse_matrix, test_data, k))
    plt.plot(k_value, user_based, color="red", label='student')
    plt.plot(k_value, item_based, color='blue', linestyle='--', label='item')
    plt.xlabel('k value')
    plt.ylabel('Test accuracy')
    plt.title('test accuracy vs k value')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()