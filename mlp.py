import numpy as np
from utils import show_data, show_history
import time

np.set_printoptions(suppress=True)


class TrainingHistory:
    def __init__(self):
        self.history = {'loss': [], 'val_loss': []}

    def add_training_error(self, error):
        self.history['loss'].append(error)

    def add_validation_error(self, error):
        self.history['val_loss'].append(error)


class MLP:
    def __init__(self, layers=(2, 24, 1), activation_functions=("tanh", "linear")):
        assert len(layers) - 1 == len(activation_functions)
        self.forward_layers = len(layers) - 1
        weight_matrices_dims = list(zip(layers[1:], layers[:-1]))
        self.weights = [
            np.random.randn(rows, cols + 1) if rows != 1 else np.random.randn(cols + 1)
            for rows, cols in weight_matrices_dims
        ]
        self.activation_functions = activation_functions
        self.data = np.loadtxt('mlp_train.txt')

    @staticmethod
    def add_bias(x):  # x is a vector
        return np.concatenate(([1], x))

    def train(self, num_epochs=1000, alpha=0.01, k_fold=10):
        history = TrainingHistory()
        for epoch in range(1, num_epochs + 1, k_fold):
            shuffled_data = np.random.permutation(self.data)

            # create ranges for k-fold groups
            rows, cols = shuffled_data.shape
            s = int(rows / k_fold)
            index = 0
            indices = []
            while index < rows - s:
                indices.append(index)
                index += s
            indices.append(rows)
            ranges = list(zip(indices[:-1], indices[1:]))

            for idx, rng in enumerate(ranges):
                print("Epoch no. {} started".format(epoch + idx))

                # split data into training and validation set
                validation_set = np.array(shuffled_data[rng[0]:rng[1]])
                training_set = np.array([row for idx, row in enumerate(shuffled_data) if idx < rng[0] or rng[1] <= idx])

                # train the model on the training set
                training_error = 0
                for entry in training_set:
                    entry_input = entry[:cols - 1]
                    entry_output = entry[cols - 1]

                    # forward propagation
                    intermediate_steps = []
                    predicted_output = entry_input
                    for i in range(self.forward_layers):
                        with_bias = self.add_bias(predicted_output)
                        computed = self.weights[i] @ with_bias
                        predicted_output = self.activation_function(self.activation_functions[i], computed)
                        intermediate_steps.append(computed)

                    # backpropagation
                    back_prop = entry_output - predicted_output
                    training_error += back_prop ** 2
                    for i in reversed(range(self.forward_layers)):
                        if i == self.forward_layers - 1:
                            computed = intermediate_steps[i]
                        else:
                            computed = self.add_bias(intermediate_steps[i])
                        delta = back_prop * self.derivative_of_activation_function(
                            self.activation_functions[i], computed
                        )
                        # remove bias weight from delta only from hidden layers and not output layer
                        if i != self.forward_layers - 1:
                            delta = np.array(delta[1:])
                        if i != 0:
                            self.weights[i] = self.weights[i] + (
                                alpha * np.outer(delta, self.activation_function(
                                        self.activation_functions[i - 1], self.add_bias(intermediate_steps[i - 1])
                                    )
                                )
                            )
                        else:  # special case, since there is no activation function for input layer
                            self.weights[i] = self.weights[i] + (alpha * (np.outer(delta, self.add_bias(entry_input))))
                        if type(delta) != np.ndarray:
                            delta = np.array([delta])
                        back_prop = self.weights[i].T @ delta

                        # this reshape is needed if the output of the NN should be a single number instead of 1x1 matrix
                        if i == self.forward_layers - 1 and self.weights[i].shape[0] == 1:
                            self.weights[i] = self.weights[i].reshape(self.weights[i].shape[1])

                # compute validation error
                validation_set_input_data = validation_set[:, :cols - 1]
                validation_set_output_data = validation_set[:, cols - 1]
                validation_error = self.compute_error(validation_set_input_data, validation_set_output_data)

                # add errors to training history
                history.add_training_error(training_error / self.data.shape[0])
                history.add_validation_error(validation_error)

                print("Epoch no. {} finished\n".format(epoch + idx))
        return history

    @staticmethod
    def activation_function(activation_function, vec):
        if activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-vec))
        elif activation_function == 'linear':
            return vec
        elif activation_function == 'relu':
            return np.maximum(0, vec)
        elif activation_function == 'tanh':
            return (np.exp(vec) - np.exp(-vec)) / (np.exp(vec) + np.exp(-vec))
        else:
            NotImplementedError("Activation function {} not implemented\n".format(activation_function))

    def derivative_of_activation_function(self, activation_function, vec):
        if activation_function == 'sigmoid':
            return self.activation_function('sigmoid', vec) * (1 - self.activation_function('sigmoid', vec))
        elif activation_function == 'linear':
            return 1
        elif activation_function == 'relu':
            return (vec > 0) * 1
        elif activation_function == 'tanh':
            return 1 - (self.activation_function('tanh', vec) ** 2)
        else:
            NotImplementedError("Activation function {} not implemented\n".format(activation_function))

    # just computes forward propagation
    def predict(self, x: np.array):  # x is vector
        output = x
        for i in range(self.forward_layers):
            with_bias = self.add_bias(output)
            output = self.activation_function(self.activation_functions[i], self.weights[i] @ with_bias)
        return output

    def compute_error(self, input_data, output_data):
        predicted = np.array(list(map(self.predict, input_data)))
        return np.mean((predicted - output_data) ** 2)

    def evaluate_training(self):
        cols = self.data.shape[1]
        input_data = self.data[:, :cols - 1]
        output_data = self.data[:, cols - 1]
        error = self.compute_error(input_data, output_data)

        print("Average error on training set was {}\n".format(error))
        return error

    def evaluate(self, file_path):
        data = np.loadtxt(file_path)
        cols = data.shape[1]
        input_data = data[:, :cols - 1]
        output_data = data[:, cols - 1]
        error = self.compute_error(input_data, output_data)

        print("Average error on test set was {}\n".format(error))
        return error

    def save_weights(self, file_path):
        with open(file_path, 'w') as outfile:
            outfile.write("{}\n".format(len(self.weights)))  # first line in weights' file is number of weight matrices
            for weights in self.weights:
                outfile.write("{}\n".format(weights.shape))  # we store the shape of all weight matrices so we can reconstruct it
                np.savetxt(outfile, weights)

    def load_weights(self, file_path):
        with open(file_path) as file:
            no_weights = int(file.readline())  # first line in weights' file is number of weight matrices
            weights = []
            for i in range(no_weights):
                layer_weights = []
                shape = file.readline()  # for each weight matrix the first line in
                shape_splitted = shape[1:-2].split(",")  # removes parenthesis and end line character and splits shape into dimensions
                assert len(shape_splitted) == 2  # all weight matrices should be 2D
                if shape_splitted[1] == '':  # shape of numpy vector is (x,), which we interpret as (x, 1)
                    shape_splitted[1] = '1'
                [rows, cols] = list(map(int, shape_splitted))
                for j in range(rows):
                    line = file.readline()
                    w = list(map(float, line.split()))
                    assert len(w) == cols  # all rows in matrix should have the same length
                    if cols == 1:
                        layer_weights.append(w[0])
                    else:
                        layer_weights.append(np.array(w))
                weights.append(np.array(layer_weights))
            self.weights = weights

    def show_data(self):
        cols = self.data.shape[1]
        show_data(self.data[:, :cols - 1], self.data[:, cols - 1])


if __name__ == "__main__":
    print("Hi and welcome to my MLP project!\n")

    mlp = MLP()
    mlp.show_data()

    # UNCOMMENT WHEN YOU WANT TO PLAY AROUND WITH THE MODEL
    train_or_load = input("Would you like to train the model or do you want to load previously trained weights?  ("
                          "train | load)\n").lower()
    while train_or_load != "train" and train_or_load != "load":
        train_or_load = input("Would you like to train the model or do you want to load previously trained weights?  ("
                              "train | load)\n").lower()
    if train_or_load == "train":
        start = time.time()
        training_history = mlp.train(num_epochs=1000)
        end = time.time()
        print("Time spent on training the model: {}".format(end - start))
        show_history(training_history)
    else:
        load_weights_file = input("What is the name of the file which you want to load weights from?\n")
        mlp.load_weights(load_weights_file)

    evaluate_file = input("What is the name of the file to evaluate the model on?\n")
    mlp.evaluate_training()
    mlp.evaluate(evaluate_file)

    save_weights = input("Do you want to save weights of the model?   (yes | no)\n").lower()
    while save_weights != "yes" and save_weights != "no":
        save_weights = input("Do you want to save weights of the model?   (yes | no)\n").lower()
    if save_weights == "yes":
        save_weights_file = input("Please provide file name where the weights should be saved to:\n")
        mlp.save_weights(save_weights_file)

    # UNCOMMENT WHEN YOU WANT TO FINALLY TEST THE MODEL
    # # First round
    # print("First round:")
    # training_history = mlp.train()
    # show_history(training_history)
    # training_error_first_round = mlp.evaluate_training()
    # evaluate_file = input("What is the name of the file to evaluate the model on?\n")
    # testing_error_first_round = mlp.evaluate(evaluate_file)
    # first_round_error = (training_error_first_round * 0.3) + (testing_error_first_round * 0.7)
    # print("Total error in first round (training (30%), testing (70%)) was: {}\n".format(first_round_error))
    #
    # # Second round
    # print("Second round:")
    # load_weights_file = input("What is the name of the file which you want to load weights from?\n")
    # mlp.load_weights(load_weights_file)
    # training_error_second_round = mlp.evaluate_training()
    # testing_error_second_round = mlp.evaluate(evaluate_file)
    # second_round_error = (training_error_second_round * 0.3) + (testing_error_second_round * 0.7)
    # print("Total error in second round (training (30%), testing (70%)) was: {}\n".format(second_round_error))
    #
    # err = (first_round_error * 0.7) + (second_round_error * 0.3)
    # print("Total error in both rounds combined (first (70%), error (30%)) was: {}".format(first_round_error))
    # points = 18.0 - (60 * err)
    # print("Points given for the neural net itself are therefore: {}\n".format(round(points, 2)))
