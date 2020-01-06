# [MLP Project](https://github.com/filipjurcak/ItAI_project)

This is a README to MLP project as a part of
**[Introduction to Artificial Intelligence class](http://dai.fmph.uniba.sk/w/Course:Introduction_to_artificial_intelligence)**.

In `main` function, you can play around with the model, create your own instance and test it how well it does on
training data. When you're done, you can uncomment the last part of the main function, which takes your model and also
loads model from file, evaluates both of them on test data and assign points based on the results.

## Installation
In order to run this project, you will need `numpy` and `matplotlib`. In case you don't have them, run
```
pip3 install numpy matplotlib
```

## Implementation

Class MLP takes care of training and evaluating of the model.
Model is trained on the data in `mlp_train.txt`.

### Class methods specification:
```
def __init__(self, layers=(2, 32, 32, 1),
			 activation_functions=("tanh", "tanh", "linear"))
```
- Initializes class with layers and activation functions specified. Weights for each layer are then initialized with
bias included in every layer.
- #### Parameters:
    - **layers**: tuple of number of neurons in each layers
    - **activation functions**: tuple of strings specifying activation functions for each layer of the network
        - activation functions currently implemented: `sigmoid, linear, relu, tanh`

- It must hold that `len(layers) + 1 == len(activation_functions)`

```
def train(self, num_epochs=1000, alpha=0.01, validation_set_size=0.2,
              training_set=None, validation_set=None, clipnorm=5.0)
```
- Trains the model on the training data, performs forward propagation and backpropagation
- #### Parameters:
	- **num_epochs**: number of iterations through full training dataset
    - **alpha**: learning rate
    - **validation_set_size**: coefficient for splitting dataset for training into training and validation set with
        validation set as `(validation_set_size * 100)%` of the dataset, rest as training set if *training_set* and
        *validation_set* options are `None`
    - **training_set**: specific training set that we want to train the model on, if this option and *validation_set*
        options are specified, *validation_set_size* option is ignored
    - **validation_set**: specific validation set that we want to validate model on, if this option and *training_set*
    options are specified, *validation_set_size* option is ignored
    - **clipnorm**: paramter for clipping delta vector in backpropagation if it's too large, prevents exploding gradients
- #### Returns:
	- **training histrory**: training and validation error in each epoch, for visualizing purposes

```
def kfold(self, k=5, num_epochs=200, alpha=0.01)
```
- performs **k-fold cross validation** on the specified model
- #### Parameters:
    - **k**: number of folds
    - **num_epochs**: number of iterations through full training dataset, will be send as option to *train* method
    - **alpha**: learning rate, will be send as option to *train* method
    
```
def evaluate_training(self):
```
- evaluate model on training data, returns computed error

```
def evaluate(self, file_path):
```
- evaluate model on data specified in the `file_path` file, returns computed error

```
def save_weights(self, file_path):
```
- save weights of the model into `file_path` file

```
def load_weights(self, file_path):
```
- load weights from the `file_path` file and sets the model to architecture specified in the file

```
def show_data(self):
```
- shows training data
    
## Initialize the model
Instance of MLP class can be defined for example as
```
mlp = MLP(layers=(2, 24, 1), activation_functions=("tanh", "linear"))
```
or in case of the default MLP just as 
```	
mlp = MLP()
```
    
## Running the model
To train the model on training data with default parameters just call
```
mlp.train()
``` 
or for example with different number of epoch and learning rate
```
mlp.train(num_epochs=500, alpha=0.001)
```    
## Evaluate the model
To get training error of the model just call
```
training_error = mlp.evaluate_training()
```
To get testing error of the model on your specified file (for example, `'test.txt'`), call
```
testing_error = mlp.evaluate('test.txt')
```

## Architecture of the best model

As we have 2 inputs and 1 output, input layer will have 2 neurons and output layer will have 1 neuron.
Because output of our function can be any real number, we chose linear function as the activation function for the output layer.

We'll run `5-fold cross validation` on different architectures and compare results and explain why we chose our architecture.

```
mlp1 = MLP(layers=(2, 2, 1), activation_functions=("tanh", "linear"))
Average validation error on MLP1 was 0.6144455038689021

mlp2 = MLP(layers=(2, 8, 1), activation_functions=("tanh", "linear"))
Average validation error on MLP2 was 0.20262778008312848

mlp3 = MLP(layers=(2, 24, 1), activation_functions=("tanh", "linear"))
Average validation error on MLP3 was 0.10666788143048782

mlp4 = MLP(layers=(2, 160, 1), activation_functions=("tanh", "linear"))
Average validation error on MLP4 was 0.41729354213383835

mlp5 = MLP(layers=(2, 320, 1), activation_functions=("tanh", "linear"))
Average validation error on MLP5 was 34.581125317943396

mlp6 = MLP(layers=(2, 32, 32, 1), activation_functions=("tanh", "tanh", "linear"))
val_errors_mlp6 = mlp6.kfold(alpha=0.001)
Average validation error on MLP6 was 0.07572071474452033

mlp7 = MLP(layers=(2, 32, 32, 1), activation_functions=("tanh", "tanh", "linear"))
val_errors_mlp7 = mlp7.kfold(alpha=0.1)
Average validation error on MLP7 was 32.742871026182534

mlp8 = MLP(layers=(2, 32, 32, 1), activation_functions=("sigmoid", "sigmoid", "linear"))
Average validation error on MLP8 was 0.11910417430128759

mlp9 = MLP(layers=(2, 32, 32, 1), activation_functions=("tanh", "tanh", "linear"))
Average validation error on MLP9 was 0.04047494245853781
```

All of these MLPs were run with `num_epochs = 200` and `alpha=0.01` except for the mlp6 & mlp7, which were run with
`alpha=0.001` and `alpha=0.1`.

As we can see, different architectures yields different results. `mlp9` represents our best model, which mean of
validation errors is low even with small number of epochs. 

We demonstrate importance of number of neurons in the network. From `mlp1` through to `mlp3` and from `mlp8` to `mlp9`
we can see that increasing number of neurons in hidden layers and even increasing number of hidden layers themselves
shrinks down validation error of models. As a result, we found that having
`2 hidden layers with 32 neurons in each hidden layer` works best for our use case.

However, as we can see, we can't just add more and more layers into the network, as this will result in overfitting, as
we can see in `mlp4` and `mlp5`, where the network didn't generalize well, so we need to choose reasonable number of
neurons in each layer.

In `mlp6` and `mlp7` we demonstrate why we chose `alpha=0.01` as our default value for `alpha`. If we choose higher
`alpha` like `alpha=0.1` or lower `alpha` like `alpha=0.001` we end up with higher validation error than with default value.

Good activation function can also reduce test error. We show this effect in `mlp8` with `sigmoid` and in `mlp9` with
`tanh` as activation functions. Average validation error got lower as a result of better properties of `tanh` function.

As consequence of all mentioned above, we decided to stick with
```
mlp = MLP(layers=(2, 32, 32, 1), activation_functions=("tanh", "tanh", "linear"))
```
with `alpha=0.01`.


### Best weights
Best weights saved from the model are stored in `weights.txt` file. When evaluating model for points
please use this file as weights file.