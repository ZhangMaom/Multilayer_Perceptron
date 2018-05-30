# coding: UTF-8

"""
multilayer perceptron

Author: Maomao Zhang
Email: 2908642732@qq.com

"""

import random
import math

class Neuron:
    """
    """
    def __init__(self, inputs=None, bias=0, weight=None):
        self.inputs = inputs
        self.bias = bias
        self.weight = weight

    # calculate weight * inputs + bias
    def neuron_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total = total + self.inputs[i] * self.weight[i]
        return total + self.bias

    # activation function: sigmoid function
    def activation_function(self, neuron_input):
        return 1 / (1 + math.exp(-neuron_input))

    # calculate output of neuron
    def neuron_output(self):
        return self.activation_function(self.neuron_input())


class InputNeuronLayer:
    """
    """
    def __init__(self, inputs=None):
        self.inputs = inputs
        self.outputs = inputs


class HiddenNeuronLayer:
    """
    """
    def __init__(self, num_neurons=0, weight=[], bias=[], inputs=None):
        self.num_neurons = num_neurons
        self.weight = weight
        self.bias = bias
        self.neurons = []
        self.outputs = []
        self.inputs = inputs
        # used when update weight and bias of previous_layer
        self.value_back_propagation_previous = []
        # used when update weight and bias of current_layer
        self.value_current_layer = []

    def init_weight_bias(self):
        self.create_random_weight()
        self.create_random_bias()
        for i in range(self.num_neurons):
            self.neurons.append(Neuron(self.inputs, self.bias[i], self.weight[i]))

    def create_random_bias(self):
        if not self.bias:
            for i in range(self.num_neurons):
                self.bias.append(random.random())

    def create_random_weight(self):
        if not self.weight:
            for i in range(self.num_neurons):
                random_weight = []
                for j in range(len(self.inputs)):
                    random_weight.append(random.random())
                self.weight.append(random_weight)

    def neuron_layer_output(self):
        self.init_weight_bias()
        for neuron in self.neurons:
            self.outputs.append(neuron.neuron_output())

    def value_back_propagation_output_input(self):
        back_value = []
        # sigmoid function: f(x) = 1/(1 + exp(-x))
        # d(f) / d(x) = (1-f(x))*f(x)
        for i in range(self.num_neurons):
            back_value.append((1-self.outputs[i]) * self.outputs[i])
        return back_value

    def value_current(self, value_back_propagation_next_current):
        back_value = self.value_back_propagation_output_input()
        for num_neuron_current in range(self.num_neurons):
            value = back_value[num_neuron_current] * value_back_propagation_next_current[num_neuron_current]
            self.value_current_layer.append(value)

    def value_back_propagation_current_previous(self, value_back_propagation_next_current):
        self.value_current(value_back_propagation_next_current)
        for num_neuron_previous in range(len(self.inputs)):
            value_previous = 0
            for num_neuron_current in range(self.num_neurons):
                value=self.value_current_layer[num_neuron_current]*self.weight[num_neuron_current][num_neuron_previous]
                value_previous += value
            self.value_back_propagation_previous.append(value_previous)

    def update_weight_bias(self, value_back_propagation_next_current, learning_rate):
        self.value_back_propagation_current_previous(value_back_propagation_next_current)
        # update bias and weight
        for num_neuron in range(self.num_neurons):
            self.bias[num_neuron] -= learning_rate * self.value_current_layer[num_neuron]
            for i in range(len(self.inputs)):
                self.weight[num_neuron][i] -= learning_rate * self.inputs[i] * self.value_current_layer[num_neuron]


class OutputNeuronLayer(HiddenNeuronLayer):
    """
    """
    def __init__(self, num_neurons=0, weight=[], bias=[], inputs=None):
        super(OutputNeuronLayer, self).__init__(num_neurons, weight, bias, inputs)
        self.value_loss_output = []

    def loss(self, training_data_label):
        loss = 0
        for num_neuron in range(self.num_neurons):
            loss = loss + (self.outputs[num_neuron] - training_data_label[num_neuron]) ** 2
        return 1 / 2 * loss

    def value_back_propagation_loss_output(self, training_data_label):
        for num_neuron in range(self.num_neurons):
            self.value_loss_output.append(self.outputs[num_neuron] - training_data_label[num_neuron])

    def update_weight_bias(self, training_data_label, learning_rate):
        self.value_back_propagation_loss_output(training_data_label)
        HiddenNeuronLayer.update_weight_bias(self, self.value_loss_output, learning_rate)


# class MultilayerPerceptron:
#     """
#     """
#
#     def __init__(self):
#         self.layers = []
#
#     def train(self, train_data, train_data_label, learning_rate):
#         # input neuron layer
#         # self.layers[0].inputs = train_data
#         self.feed_forward()
#         self.back_propagation(train_data_label, learning_rate)
#
#     def test(self, test_data):
#         #self.layers[0] = InputNeuronLayer(test_data)
#         self.feed_forward()
#
#     def feed_forward(self):
#         for index in range(1, len(self.layers)):
#             self.layers[index].inputs = self.layers[index-1].outputs
#             self.layers[index].neuron_layer_output()
#
#     def back_propagation(self, train_data_label, learning_rate):
#         self.layers[len(self.layers)-1].update_weight_bias(train_data_label, learning_rate)
#         for i in range(len(self.layers)-2, 0, -1):
#             self.layers[i].update_weight_bias(self.layers[i+1].value_back_propagation_previous, learning_rate)
#
#
# class NeuralNetwork:
#     """
#
#     """
#     def __init__(self, learning_rate=0.01, epoch=1000):
#         self.learning_rate = learning_rate
#         self.epoch = epoch
#         self.network_architecture = MultilayerPerceptron()
#
#     def train(self, training_data_set, training_data_set_label):
#         for i in range(self.epoch):
#             index = 0#random.randint(0, len(training_data_set) - 1)
#             self.network_architecture.layers[0] = InputNeuronLayer(training_data_set[index])
#
#             weight = []
#             bias = []
#
#             for j in range(1, len(self.network_architecture.layers) - 1):
#                 weight.append(self.network_architecture.layers[j].weight)
#                 bias.append(self.network_architecture.layers[j].bias)
#                 self.network_architecture.layers[j] = HiddenNeuronLayer(self.network_architecture.layers[j].num_neurons,
#                     self.network_architecture.layers[j].weight, self.network_architecture.layers[j].bias,
#                     self.network_architecture.layers[j-1].outputs)
#
#             weight.append(self.network_architecture.layers[len(self.network_architecture.layers)-1].weight)
#             bias.append(self.network_architecture.layers[len(self.network_architecture.layers)-1].bias)
#
#             self.network_architecture.layers[len(self.network_architecture.layers)-1] = OutputNeuronLayer(
#                 self.network_architecture.layers[len(self.network_architecture.layers)-1].num_neurons,
#                 self.network_architecture.layers[len(self.network_architecture.layers)-1].weight,
#                 self.network_architecture.layers[len(self.network_architecture.layers)-1].bias,
#                 self.network_architecture.layers[len(self.network_architecture.layers)-2].outputs)
#             self.network_architecture.train(training_data_set[index], training_data_set_label[index],self.learning_rate)
#
#             print(i, self.network_architecture.layers[len(self.network_architecture.layers)-1].loss_value,
#                   self.network_architecture.layers[len(self.network_architecture.layers)-1].outputs, weight, bias)
#
#     def test(self, test_data_set):
#         for i in range(len(test_data_set)):
#             self.network_architecture.layers[0] = InputNeuronLayer(test_data_set[i])
#             self.network_architecture.test(test_data_set[i])

class MultilayerPerceptron:
    """

    """

    def __init__(self, learning_rate=0.01, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.layers = []

    def train(self, training_data_set, training_data_set_label):
        for i in range(self.epoch):
            index = random.randint(0, len(training_data_set) - 1)
            self.layers[0] = InputNeuronLayer(training_data_set[index])

            weight = []
            bias = []

            for j in range(1, len(self.layers) - 1):
                weight.append(self.layers[j].weight)
                bias.append(self.layers[j].bias)
                self.layers[j] = HiddenNeuronLayer(self.layers[j].num_neurons,
                                                   self.layers[j].weight,
                                                   self.layers[j].bias,
                                                   self.layers[j-1].outputs)

            weight.append(self.layers[len(self.layers) - 1].weight)
            bias.append(self.layers[len(self.layers) - 1].bias)

            self.layers[len(self.layers) - 1] = OutputNeuronLayer(
                                self.layers[len(self.layers)-1].num_neurons,
                                self.layers[len(self.layers)-1].weight,
                                self.layers[len(self.layers)-1].bias,
                                self.layers[len(self.layers)-2].outputs)

            self.feed_forward()
            self.back_propagation(training_data_set_label[index])
            print(i, self.total_loss(training_data_set, training_data_set_label),
                  self.layers[len(self.layers) - 1].loss(training_data_set_label[index]),
                  self.layers[len(self.layers) - 1].outputs, weight, bias)



    def test(self, test_data_set, test_data_set_outputs):
        num_correct = 0
        for i in range(len(test_data_set)):
            self.layers[0] = InputNeuronLayer(test_data_set[i])

            for j in range(1, len(self.layers) - 1):
                self.layers[j] = HiddenNeuronLayer(self.layers[j].num_neurons,
                                                   self.layers[j].weight,
                                                   self.layers[j].bias,
                                                   self.layers[j-1].outputs)

            self.layers[len(self.layers) - 1] = OutputNeuronLayer(
                                self.layers[len(self.layers)-1].num_neurons,
                                self.layers[len(self.layers)-1].weight,
                                self.layers[len(self.layers)-1].bias,
                                self.layers[len(self.layers)-2].outputs)

            self.feed_forward()
            if(self.layers[len(self.layers) - 1].outputs == test_data_set_outputs[i]):
                num_correct += 1
        return num_correct / len(test_data_set)

    def total_loss(self, training_data_set, training_data_set_label):
        total_loss = 0
        for i in range(len(training_data_set)):
            self.layers[0] = InputNeuronLayer(training_data_set[i])

            for j in range(1, len(self.layers) - 1):
                self.layers[j] = HiddenNeuronLayer(self.layers[j].num_neurons,
                                                   self.layers[j].weight,
                                                   self.layers[j].bias,
                                                   self.layers[j-1].outputs)

            self.layers[len(self.layers) - 1] = OutputNeuronLayer(
                                self.layers[len(self.layers)-1].num_neurons,
                                self.layers[len(self.layers)-1].weight,
                                self.layers[len(self.layers)-1].bias,
                                self.layers[len(self.layers)-2].outputs)

            self.feed_forward()

            total_loss += self.layers[len(self.layers)-1].loss(training_data_set_label[i])
        return total_loss

    def feed_forward(self):
        for index in range(1, len(self.layers)):
            self.layers[index].inputs = self.layers[index - 1].outputs
            self.layers[index].neuron_layer_output()

    def back_propagation(self, train_data_label):
        self.layers[len(self.layers) - 1].update_weight_bias(train_data_label, self.learning_rate)
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].update_weight_bias(self.layers[i + 1].value_back_propagation_previous, self.learning_rate)


if __name__ == "__main__":
    learning_rate = 0.01
    epoch = 10000
    training_data_set = [[1, 1], [1, 0], [0, 1], [0, 0]]
    training_data_set_label = [[0], [1], [1], [0]]
    mlp = MultilayerPerceptron()

    mlp.layers.append(InputNeuronLayer())
    mlp.layers.append(HiddenNeuronLayer(2))
    mlp.layers.append(OutputNeuronLayer(1))

    mlp.train(training_data_set, training_data_set_label)
    print(mlp.test([[1, 1]], [0]))
