import argparse
import numpy as np

import matplotlib.pyplot as plt
from visualize import plot_decision_regions

# Some inspiration taken from:
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy_loss(y_true, y_pred):
    return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


def cross_entropy_gradient(y_true, y_pred):
    return (- y_true + y_pred) / (y_pred - y_pred ** 2)


class MLP:
    def __init__(self, learning_rate=1, n_hidden_nodes=8, init_std=0.1):
        self.W1 = np.random.normal(0, init_std, [3, n_hidden_nodes])
        self.W2 = np.random.normal(0, init_std, [1 + n_hidden_nodes, 1])

        self.activation = sigmoid
        self.lr = learning_rate

    def predict(self, X):
        if X.shape[1] == self.W1.shape[0] - 1:
            X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
        self.params = {'X': X}

        tmp = X.dot(self.W1)
        self.params['W1 @ X'] = tmp

        tmp = self.activation(tmp)
        tmp = np.concatenate([np.ones([X.shape[0], 1]), tmp], axis=1)
        self.params['g(W1 @ X)'] = tmp

        tmp = tmp.dot(self.W2)
        self.params['W2 @ g(W1 @ X)'] = tmp

        tmp = self.activation(tmp)
        self.params['g(W2 @ g(W1 @ X))'] = tmp

        return tmp

    def update_weights(self, y_true, y_pred):
        assert self.params != {}

        # ∂L/∂pred
        pd_loss = cross_entropy_gradient(y_true, y_pred)
        print("y_true", y_true)
        print("y_pred", y_pred)
        print("pd_loss", pd_loss, pd_loss.shape)
        # ∂sigmoid(x)/∂x
        pd_second_sigmoid = sigmoid_gradient(self.params['W2 @ g(W1 @ X)'])
        print("self.params['W2 @ g(W1 @ X)']", self.params['W2 @ g(W1 @ X)'], self.params['W2 @ g(W1 @ X)'].shape)
        print("pd_second_sigmoid", pd_second_sigmoid, pd_second_sigmoid.shape)
        # ∂(W2 @ g(W1 @ X))/∂ W2
        pd_matmul = self.params['g(W1 @ X)']
        print("pd_matmul", pd_matmul, pd_matmul.shape)

        # save the post_w2_loss that captures cross entropy and final sigmoid
        post_w2_loss = pd_loss * pd_second_sigmoid
        print("post_w2_loss", post_w2_loss, post_w2_loss.shape)
        temp = post_w2_loss * pd_matmul
        print("temp_w2_update", temp, temp.shape)
        
        w2_update = np.mean(temp, axis=0, keepdims=True)
        print("w2_update", w2_update, w2_update.shape)
        
        # ∂(W2 @ g(W1 @ X))/∂ g(W1 @ X)
        pd_W2_g = self.W2
        print("pd_W2_g", pd_W2_g, pd_W2_g.shape)
        # ∂sigmoid(x)/∂x
        pd_first_sigmoid = sigmoid_gradient(self.params['W1 @ X'])
        print("pd_first_sigmoid", pd_first_sigmoid, pd_first_sigmoid.shape)
        # ∂(W1 @ X) /∂W1
        pd_W1_X = self.params['X']
        print("pd_W1_X", pd_W1_X, pd_W1_X.shape)

        w1_update = np.zeros_like(self.W1)
        print("w1_update", w1_update.shape)

        # for each node in the hidden layer ...
        for i in range(self.W1.shape[1]):
            # ... only grab the losses for one specific hidden node
            w1_loss = post_w2_loss * pd_W2_g[i + 1] * pd_first_sigmoid[:, (i, )]
            print("pd_first_sigmoid[:, (i, )]", i, pd_first_sigmoid[:, (i, )], pd_first_sigmoid[:, (i, )].shape)
            print("w1_loss", i, w1_loss, w1_loss.shape)
            w1_update[:, i] = np.mean(
                w1_loss * pd_W1_X, axis=0, keepdims=True)

        self.params = {}
        self.W1 -= self.lr * w1_update
        self.W2 -= self.lr * w2_update.T

    def fit(self, X, y, steps=10000, quiet=True):
        if X.shape[1] == self.W1.shape[1] - 1:
            X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
        
        for i in range(steps):
            y_pred = self.predict(X)
            loss = cross_entropy_loss(y, y_pred)
            self.update_weights(y, y_pred)
            if not quiet and (i + 1) % (steps // 10) == 0:
                print(i + 1, np.mean(loss))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--init_std", type=float, default=0.1)
    parser.add_argument("--bonus", action="store_true")
    parser.add_argument("--n_hidden_nodes", type=int, default=2)
    parser.add_argument("--n_iters", type=int, default=10000)
    parser.add_argument("--plot_before", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # xor dataset
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]],
                 dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    if args.bonus:
        X_bonus = np.array([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]])
        y_bonus = np.array([[0], [1], [0]])
        X = np.concatenate([X, X_bonus], axis=0)
        y = np.concatenate([y, y_bonus], axis=0)

    mlp = MLP(learning_rate=args.lr,
              init_std=args.init_std,
              n_hidden_nodes=args.n_hidden_nodes)

    if args.plot_before:
        plot_decision_regions(X, y, mlp)
        plt.show()

    mlp.fit(X, y, quiet=not args.verbose, steps=args.n_iters)

    print("{:.0f}% accuracy".format(
        100 * np.mean((mlp.predict(X) > 0.5) == y)))
    print(np.round(mlp.W1, 2))
    print(np.round(mlp.W2, 2))

    # plot the new decision boundaries
    plot_decision_regions(X, y, mlp)

    if args.save:
        fn = []
        for key in ["seed", "lr", "init_std", "bonus", "n_hidden_nodes", "n_iters"]:
            fn.append(f"{key}{getattr(args, key)}")
        fn = "-".join(fn)
        plt.savefig(f"plots/{fn}.png")
    else:
        plt.show()
    plt.close('all')


# python backprop.py --bonus --verbose --n_iter 10000
# python backprop.py --bonus --verbose --n_iter 20000
if __name__ == "__main__":
    main()
