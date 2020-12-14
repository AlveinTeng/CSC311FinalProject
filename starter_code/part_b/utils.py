from scipy.sparse import load_npz
from utils import *
import numpy as np
import csv
import os
import ast
from scipy.sparse import csr_matrix
import random


def _load_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": [],
        "subject_id": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["user_id"].append(int(row[1]))
                data["is_correct"].append(int(row[2]))
                data["subject_id"].append([])
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def _create_question(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "question_id": [],
        "subject_id": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["subject_id"].append(ast.literal_eval(row[1]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def create_data(root_dir="/data"):
    train_data = load_train_csv(root_dir)
    question_data = load_qestion_meta(root_dir)
    valid_data = load_valid_csv(root_dir)
    test_data = load_public_test_csv(root_dir)
    # Record the time that the subject is countered
    new_train_matrix_nominator = np.empty((542, 388))
    new_train_matrix_nominator[:] = np.NaN
    # Record the time that the question is correctly answered
    new_train_matrix_denominator = np.empty((542, 388))
    new_train_matrix_denominator[:] = np.NaN
    # The new train matrix
    new_train_matrix = np.empty((542, 388))
    new_train_matrix[:] = np.NaN

    new_sparse_train_matrix = np.empty((542, 388))
    new_sparse_train_matrix[:] = np.NaN

    for i, q in enumerate(train_data["user_id"]):
        question_id = train_data["question_id"][i]
        subject_list = question_data["subject_id"][question_id]
        train_data["subject_id"][i] = subject_list
        is_correct = train_data["is_correct"][i]

        for j, sub in enumerate(subject_list):
            # print(np.isnan(new_train_matrix_denominator[q][sub]))
            if np.isnan(new_train_matrix_denominator[q][sub]):
                # print(1)
                # print(new_train_matrix_denominator[q][sub])
                new_train_matrix_denominator[q][sub] = 1
                # print(new_train_matrix_denominator[q][sub])
            else:
                new_train_matrix_denominator[q][sub] += 1
        for j, sub in enumerate(subject_list):
            if np.isnan(new_train_matrix_nominator[q][sub]):
                new_train_matrix_nominator[q][sub] = is_correct
            else:
                new_train_matrix_nominator[q][sub] += is_correct
            # new_sparse_train_matrix[q][sub] = is_correct
    # print(new_train_matrix_nominator)
    # print(new_train_matrix_denominator)
    for i in range(new_train_matrix.shape[0]):
        for j in range(new_train_matrix.shape[1]):
            if new_train_matrix_denominator[i][j] != np.NaN:
                new_train_matrix[i][j] = new_train_matrix_nominator[i][j] / new_train_matrix_denominator[i][j]
    new_sparse_train_matrix = new_train_matrix.copy()
    for i, q in enumerate(valid_data["user_id"]):
        question_id = valid_data["question_id"][i]
        subject_list = question_data["subject_id"][question_id]
        valid_data["subject_id"][i] = subject_list

    for i, q in enumerate(test_data["user_id"]):
        question_id = test_data["question_id"][i]
        subject_list = question_data["subject_id"][question_id]
        test_data["subject_id"][i] = subject_list

    random.seed(0)
    for i in range(542):
        for j in range(388):
            if random.uniform(0, 1) > 0.5:
                new_sparse_train_matrix[i][j] = np.NaN

    return new_train_matrix, new_sparse_train_matrix, valid_data, test_data


def load_train_sparse(root_dir="/data"):
    """ Load the training data as a spare matrix representation.

    :param root_dir: str
    :return: 2D sparse matrix
    """
    path = os.path.join(root_dir, "train_sparse.npz")
    if not os.path.exists(path):
        raise Exception("The specified path {} "
                        "does not exist.".format(os.path.abspath(path)))
    matrix = load_npz(path)
    return matrix


def load_train_csv(root_dir="/data"):
    """ Load the training data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "train_data.csv")
    return _load_csv(path)


def load_valid_csv(root_dir="/data"):
    """ Load the validation data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "valid_data.csv")
    return _load_csv(path)


def load_public_test_csv(root_dir="/data"):
    """ Load the test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "test_data.csv")
    return _load_csv(path)


def load_private_test_csv(root_dir="/data"):
    """ Load the private test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: an empty list.
    """
    path = os.path.join(root_dir, "private_test_data.csv")
    return _load_csv(path)


def load_qestion_meta(root_dir="/data"):
    """ Load the qestion_meta

    :param root_dir: str
    :return: A dictionary {question_id: list, subject_id: list}
        WHERE
        question_id: a list of question id.
        subject_id: a list of list of subject id.
    """
    path = os.path.join(root_dir, "question_meta.csv")
    return _create_question(path)


def save_private_test_csv(data, file_name="private_test_result.csv"):
    """ Save the private test data as a csv file.

    This should be your submission file to Kaggle.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    :param file_name: str
    :return: None
    """
    if not isinstance(data, dict):
        raise Exception("Data must be a dictionary.")
    cur_id = 1
    valid_id = ["0", "1"]
    with open(file_name, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "is_correct"])
        for i in range(len(data["user_id"])):
            if str(int(data["is_correct"][i])) not in valid_id:
                raise Exception("Your data['is_correct'] is not in a valid format.")
            writer.writerow([str(cur_id), str(int(data["is_correct"][i]))])
            cur_id += 1
    return


def evaluate(data, predictions, threshold=0.5):
    """ Return the accuracy of the predictions given the data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param predictions: list
    :param threshold: float
    :return: float
    """
    if len(data["is_correct"]) != len(predictions):
        raise Exception("Mismatch of dimensions between data and prediction.")
    if isinstance(predictions, list):
        predictions = np.array(predictions).astype(np.float64)
    return (np.sum((predictions >= threshold) == data["is_correct"])
            / float(len(data["is_correct"])))


def sparse_matrix_evaluate(data, matrix, threshold=0.5):
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
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def sparse_matrix_predictions(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the predictions.

    This function can be used for submitting Kaggle competition.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: list
    """
    predictions = []
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold:
            predictions.append(1.)
        else:
            predictions.append(0.)
    return predictions
