from utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch

from utils import load_public_test_csv, load_valid_csv, \
    load_train_sparse, create_data


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    new_train_matrix, new_sparse_train_matrix, valid_data, test_data = create_data(base_path)
    # print(new_train_matrix)
    # print(new_sparse_train_matrix)
    # train_matrix = load_train_sparse(base_path).toarray()
    # valid_data = load_valid_csv(base_path)
    # test_data = load_public_test_csv(base_path)

    # print(len(valid_data["user_id"]))
    # print(len(test_data["user_id"]))

    # zero_train_matrix = train_matrix.copy()
    zero_train_matrix = new_sparse_train_matrix.copy()
    # Fill in the missing entries to 0.
    # zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix[np.isnan(new_sparse_train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    # train_matrix = torch.FloatTensor(train_matrix)
    train_matrix = torch.FloatTensor(new_sparse_train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        # self.l1 = nn.Linear(k, k)
        # self.l4 = nn.Linear(k, k)
        self.h = nn.Linear(k, num_question)


    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out1 = F.sigmoid(self.g(inputs))
        # out2 = F.relu(self.l1(out1))
        # out5 = F.relu(self.l4(out2))
        output = F.sigmoid(self.h(out1))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return output


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    cost = []
    val_acc = []

    for epoch in range(num_epoch):
        train_loss = 0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())

            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        reg = (lamb / 2) * model.get_weight_norm()
        train_loss += reg
        # print("Epoch: {} \tTraining Cost: {:.6f}".format(epoch, train_loss))

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t Valid Acc: {}".format(epoch, train_loss, valid_acc))

        cost.append(train_loss)
        val_acc.append(valid_acc)

    return cost, val_acc


    # # Tell PyTorch you are training the model.
    # model.train()
    #
    # # Define optimizers and loss function.
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # num_student = train_data.shape[0]
    #
    # cost = []
    # val_acc = []
    #
    # for epoch in range(0, num_epoch):
    #     train_loss = 0.
    #
    #     for user_id in range(num_student):
    #         inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
    #         target = inputs.clone()
    #
    #         optimizer.zero_grad()
    #         output = model(inputs)
    #
    #         # Mask the target to only compute the gradient of valid entries.
    #         nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
    #         target[0][nan_mask] = output[0][nan_mask]
    #
    #         loss = torch.sum((output - target) ** 2.)
    #         loss.backward()
    #
    #         train_loss += loss.item()
    #         optimizer.step()
    #
    #     reg = (lamb / 2) * model.get_weight_norm()
    #     train_loss += reg
    #     valid_acc = evaluate(model, zero_train_data, valid_data)
    #     print("Epoch: {} \tTraining Cost: {:.6f}\t "
    #           "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    #
    #     cost.append(train_loss)
    #     val_acc.append(valid_acc)
    #
    # return cost, val_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = 0
        for j, sub in enumerate(valid_data["subject_id"][i]):
            guess += output[0][sub].item()
            # if output[0][sub].item() >= 0.5:
            #     guess += 1
        # guess = (guess / len(valid_data["subject_id"][i])) >= 0.5
        # print(guess / len(valid_data["subject_id"][i]))
        guess = 1 if (guess / len(valid_data["subject_id"][i])) >= 0.6 else 0

        # guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)
    #     correct = guess
    # return correct


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.

    subjects_num = train_matrix.shape[1]

    k_values = [10, 50, 100, 200, 500]
    epochs = np.arange(50)

    # Set optimization hyperparameters.
    lr = 0.01
    lambds = [0.001, 0.01, 0.1, 1]

    k = k_values[2]
    print("K is {}".format(k))
    # for lamb in lambds:
    lamb = 0.001
    print("Lambda is {}".format(lamb))
    model = AutoEncoder(subjects_num, k)
    costs, validation_acc = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, len(epochs))
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print('Test accuracy is {}'.format(test_acc))

    plt.figure()
    plt.plot(epochs, validation_acc, color ='red', label = 'validation accuracy')
    plt.title('validation accuracy vs epoches')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.figure()
    plt.plot(epochs, costs, color ='blue', label = 'train costs')
    plt.title('train cost vs epoches')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend(loc='lower right')
    plt.show()


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()