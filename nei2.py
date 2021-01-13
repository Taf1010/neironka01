import numpy as np
from colorama import init
from colorama import Fore, Back, Style
init()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print("Случайные инициализирующие веса:")
print(synaptic_weights)

for i in range(20000):
    input_layer = training_inputs
    outputs = sigmoid( np.dot(input_layer, synaptic_weights))

    err = training_outputs - outputs
    adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)) )

    synaptic_weights += adjustments

print("Веса после обучения: ")
print(synaptic_weights)

print("Результат списков для обучения: ")
print(outputs)

#print("Введите ряд из трех чисел(1 или 0): ")
#a=input(int)
#b=input(int)
#c=input(int)
print(Fore.YELLOW)
print("Ряд чисел 1,1,0\nНейросеть определяет шансы того что превое число равно 1,\nСудя по предыдущим рядам обучения:\n0,0,1\n1,1,1\n1,0,1\n0,1,1")
#new_inputs2 = new_inputs + int(a)
new_inputs = np.array([1,1,0])
output = sigmoid( np.dot( new_inputs, synaptic_weights))

print("Ряд 1,1,0: ")
print(output)