from utils import *
from item_response import irt, sigmoid
from neural_network import load_data, AutoEncoder, train
import numpy as np
import random
import torch
from torch.autograd import Variable
from sklearn.impute import KNNImputer


def resample_matrix(sparse_matrix, M):
    """
    Resample M samples from sparse_matrix with replacement
    :param sparse_matrix:
    :param M: number of samples needed
    :return: list of samples
    """
    results = []
    N = len(sparse_matrix)
    if not isinstance(sparse_matrix, np.ndarray):
        sparse_matrix = np.asarray(sparse_matrix)
    for i in range(M):
        random_indices = random.sample(range(N), N)
        resampled = np.zeros(sparse_matrix.shape)
        for j in range(len(random_indices)):
            new_index = random_indices[j]
            resampled[j] += sparse_matrix[new_index]
        results.append(resampled)
    return results


def resample_data(train_data):
    """
    Resample len(train_data) elements from train_data with replacement
    """
    results = {"user_id": [], "question_id": [], "is_correct": []}
    N = len(train_data)
    random_indices = random.sample(range(N), N)
    for i in range(N):
        new_index = random_indices[i]
        results["user_id"].append(train_data["user_id"][new_index])
        results["question_id"].append(train_data["question_id"][new_index])
        results["is_correct"].append(train_data["is_correct"][new_index])
    return results


def resample_tensor(sparse_matrices):
    """
    Resample M samples from sparse_matrix with replacement
    :param sparse_matrices: list of sparse_matrix
    :return: list of matrices
    """
    N = len(sparse_matrices[0])
    for i in range(len(sparse_matrices)):
        if not isinstance(sparse_matrices[i], np.ndarray):
            sparse_matrices[i] = np.asarray(sparse_matrices[i])
    random_indices = random.sample(range(N), N)
    resampled_mat1 = (sparse_matrices[0])[random_indices, :]
    resampled_mat2 = (sparse_matrices[1])[random_indices, :]

    return torch.FloatTensor(resampled_mat1), torch.FloatTensor(resampled_mat2)


def predict_knn(train_matrix, data, k):
    predictions = []
    nbrs = KNNImputer(n_neighbors=k)
    knn_sample = resample_matrix(train_matrix, 1)
    mat = nbrs.fit_transform(knn_sample[0])
    for i in range(len(data["is_correct"])):
        uid = data["user_id"][i]
        qid = data["question_id"][i]
        predictions.append(mat[uid, qid])
    return predictions


def predict_irt(train_data, data, lr, num_epoch):
    predictions = []
    theta, beta, val_acc_lst, train_acc_lst = irt(train_data, data, lr, num_epoch)
    for i in range(len(data["is_correct"])):
        uid = data["user_id"][i]
        qid = data["question_id"][i]
        x = (theta[uid] - beta[qid]).sum()
        p_a = sigmoid(x)
        predictions.append(p_a >= 0.5)
    return predictions


def predict_nn(data, lr, num_epoch, lamb):
    predictions = []
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    zero_train_matrix, train_matrix = resample_tensor([zero_train_matrix, train_matrix])
    nn_sample = torch.FloatTensor(train_matrix)
    nn_model = AutoEncoder(nn_sample.shape[1], 9)
    train(nn_model, lr, lamb, nn_sample, zero_train_matrix, data, num_epoch)

    for i in range(len(data["is_correct"])):
        uid = data["user_id"][i]
        qid = data["question_id"][i]
        inputs = Variable(zero_train_matrix[uid]).unsqueeze(0)
        outputs = nn_model(inputs)
        pred = outputs[0][qid]
        predictions.append(pred)
    return predictions


def evaluate(data, predictions):
    total = 0
    correct = 0
    threshold = 0.5

    for i, pred in enumerate(predictions):
        if pred >= threshold and data["is_correct"][i]:
            correct += 1
        if pred < threshold and not data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / total


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # knn base model
    knn_predictions_val = predict_knn(sparse_matrix, val_data, k=11)
    knn_predictions_test = predict_knn(sparse_matrix, test_data, k=11)
    print("KNN Accuracy for validation set is {}".format(evaluate(val_data, knn_predictions_val)))
    print("KNN Accuracy for test set is {}".format(evaluate(test_data, knn_predictions_test)))

    # irt model
    irt_predictions_val = predict_irt(train_data, val_data, 0.001, 120)
    irt_predictions_test = predict_irt(train_data, test_data, 0.001, 120)
    print("IRT Accuracy for validation set is {}".format(evaluate(val_data, irt_predictions_val)))
    print("IRT Accuracy for test set is {}".format(evaluate(test_data, irt_predictions_test)))

    # nn model
    nn_predictions_val = predict_nn(val_data, lr=0.06, num_epoch=9, lamb=0.001)
    nn_predictions_test = predict_nn(test_data, lr=0.06, num_epoch=9, lamb=0.001)
    print("NN Accuracy for validation set is {}".format(evaluate(val_data, nn_predictions_val)))
    print("NN Accuracy for test set is {}".format(evaluate(test_data, nn_predictions_test)))

    sum_up_val = [knn_predictions_val[i] + irt_predictions_val[i] + nn_predictions_val[i]
                  for i in range(len(val_data["is_correct"]))]
    ensemble_pred_val = [pred / 3 for pred in sum_up_val]
    ensemble_acc_val = evaluate(val_data, ensemble_pred_val)
    print("Ensemble Accuracy for validation set is {}".format(ensemble_acc_val))

    sum_up_test = [knn_predictions_test[i] + irt_predictions_test[i] + nn_predictions_test[i]
                  for i in range(len(test_data["is_correct"]))]
    ensemble_pred_test = [pred / 3 for pred in sum_up_test]
    ensemble_acc_test = evaluate(test_data, ensemble_pred_test)
    print("Ensemble Accuracy for test set is {}".format(ensemble_acc_test))

    # # irt model
    # predictions_val = []
    # predictions_test = []
    # for i in range(3):
    #     irt_predictions_val = predict_irt(train_data, val_data, 0.001, 120)
    #     irt_predictions_test = predict_irt(train_data, test_data, 0.001, 120)
    #     predictions_val.append(irt_predictions_val)
    #     predictions_test.append(irt_predictions_test)
    #     print("IRT Model{} Accuracy for validation set is {}".format(i+1, evaluate(val_data, irt_predictions_val)))
    #     print("IRT Model{} Accuracy for test set is {}".format(i+1, evaluate(test_data, irt_predictions_test)))
    #
    # sum_up_val = []
    # for j in range(len(val_data["is_correct"])):
    #     sum_up = sum(predictions_val[i][j] for i in range(3))
    #     sum_up_val.append(sum_up)
    # ensemble_pred_val = [pred / 3 for pred in sum_up_val]
    # ensemble_acc_val = evaluate(val_data, ensemble_pred_val)
    # print("Ensemble Accuracy for validation set is {}".format(ensemble_acc_val))
    #
    # sum_up_test = []
    # for j in range(len(test_data["is_correct"])):
    #     sum_up = sum(predictions_test[i][j] for i in range(3))
    #     sum_up_test.append(sum_up)
    # ensemble_pred_test = [pred / 3 for pred in sum_up_test]
    # ensemble_acc_test = evaluate(test_data, ensemble_pred_test)
    # print("Ensemble Accuracy for test set is {}".format(ensemble_acc_test))


if __name__ == "__main__":
    main()
