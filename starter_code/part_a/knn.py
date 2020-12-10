from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy by student: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.transpose())
    acc = sparse_matrix_evaluate_item(valid_data, mat)
    print("Validation Accuracy by item: {}".format(acc))
    # Implement the function as described in the docstring.             #
    #####################################################################
    # acc = None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc
def sparse_matrix_evaluate_item(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_question_id, cur_user_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_question_id, cur_user_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    k_value = [1, 6, 11, 16, 21, 26]
    # Validation data
    # results_user = []
    # results_item = []
    # for k in k_value:
    #     results_user.append(knn_impute_by_user(sparse_matrix,val_data, k))
    #     results_item.append(knn_impute_by_item(sparse_matrix, val_data, k))
    # # for k in  k_value:
    # #     results_item.append(knn_impute_by_item(sparse_matrix, val_data, k))
    # plt.plot(k_value, results_user, color = "red", label = 'student')
    # plt.plot(k_value, results_item, color = 'blue', linestyle = '--' , label = 'item')
    # plt.xlabel('k value')
    # plt.ylabel('validation accuracy')
    # plt.title('validation accuracy cs k value')
    # plt.legend(loc= 'lower right')
    # plt.show()
    # Test performance
    user_based = []
    item_based = []
    for k in k_value:
        user_based.append(knn_impute_by_user(sparse_matrix,test_data, k))
        item_based.append(knn_impute_by_item(sparse_matrix, test_data, k))
    # for k in  k_value:
    #     results_item.append(knn_impute_by_item(sparse_matrix, val_data, k))
    plt.plot(k_value, user_based, color = "red", label = 'student')
    plt.plot(k_value, item_based, color = 'blue', linestyle = '--' , label = 'item')
    plt.xlabel('k value')
    plt.ylabel('validation accuracy')
    plt.title('validation accuracy cs k value')
    plt.legend(loc= 'lower right')
    plt.show()

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
