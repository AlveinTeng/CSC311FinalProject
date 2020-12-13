from utils import *
import matplotlib.pyplot as plt
import numpy as np


def logsumexp_stable(a, axis=None):
    '''
    Compute the logsumexp of the numpy array x along an axis.

    Parameters
    ----------
    a : array_like
        Elements to logsumexp.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.

    Returns
    -------
    logsumexp_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.
    '''
    m = np.max(a, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(a - m), axis=axis)) + np.squeeze(m, axis=axis)


def sigmoid(x, alpha):
    """ Apply sigmoid function.
    """
    # alpha = 1.7 * alpha
    return np.exp(np.multiply(alpha, x)) / (1 + np.exp(np.multiply(alpha, x)))


def log_p_cij(theta, beta, q_id_j, u_id_i, is_c_ij, alpha):
    a = np.array([0, alpha[q_id_j] * (theta[u_id_i] - beta[q_id_j])])
    if is_c_ij:
        return alpha[q_id_j] * (theta[u_id_i] - beta[q_id_j]) - logsumexp_stable(a)
    return -logsumexp_stable(a)


def neg_log_likelihood(data, theta, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]

    N = len(is_correct)

    log_lklihood = np.sum([log_p_cij(theta, beta, question_id[i], user_id[i], is_correct[i], alpha) for i in range(N)])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
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
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]

    N = len(is_correct)
    theta_gradient = np.zeros(len(theta))
    beta_gradient = np.zeros(len(beta))
    alpha_gradient = np.zeros(len(alpha))
    for i in range(N):
        z_i = theta[user_id[i]] - beta[question_id[i]]
        y_i = sigmoid(z_i, alpha[question_id[i]])
        theta_gradient[user_id[i]] += alpha[question_id[i]] * (is_correct[i] - y_i)
        beta_gradient[question_id[i]] += alpha[question_id[i]] * (y_i - is_correct[i])
        alpha_gradient[question_id[i]] += z_i * (is_correct[i] - y_i)
    theta += lr * theta_gradient
    beta += lr * beta_gradient
    alpha += lr * alpha_gradient
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


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
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)
    alpha = np.random.rand(1774)

    # print(alpha)

    val_acc_lst = []
    train_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        val_score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
        train_score = evaluate(data=data, theta=theta, beta=beta, alpha=alpha)
        val_acc_lst.append(val_score)
        train_acc_lst.append(train_score)
        print("i: {}, NLLK: {} \t Score: {}".format(i, neg_lld, val_score))
        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_acc_lst, alpha


def evaluate(data, theta, beta, alpha):
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
        # print("alpha q:{}".format(alpha[q]))
        p_a = sigmoid(x, alpha[q])
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc_lst, train_acc_lst, alpha = irt(train_data, val_data, 0.005, 200)
    plt.plot(val_acc_lst, label="validation acc")
    plt.plot(train_acc_lst, label="training acc")
    plt.title("Training Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    valid_acc = evaluate(val_data, theta, beta, alpha)
    test_acc = evaluate(test_data, theta, beta, alpha)
    print("Validation Accuracy:", valid_acc)
    print("Test Accuracy:", test_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    for j in [0, 1, 2, 3, 4]:
        beta_j = beta[j]
        alpha_j = alpha[j]
        sorted_theta = np.sort(theta)
        p_correct = np.exp(-np.logaddexp(0, alpha_j * (beta_j - sorted_theta)))
        plt.plot(sorted_theta, p_correct, label="j={}".format(j))
    plt.title("probability of the correct response vs theta")
    plt.ylabel("probability")
    plt.xlabel("theta")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
