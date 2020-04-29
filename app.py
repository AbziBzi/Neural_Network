import matplotlib.pyplot as plt
import math
import numpy as np
from neural_net import NeuralNetwork
from norm import norm, standDev, mean, denorm

FILE_PATH = "data/sunspot.txt"
N = 4

# Read data from given file path
# Returns list of lines
def read_data(file_path):
    file = open(file_path, "r")
    if file.mode == 'r':
        data = file.readlines()
        return data

# Splits given data by \t
# Returns two lists: list of years and list of spots
def split_data(data):
    years = []
    spots = []
    for d in data:
        splited_data = d.split('\t')
        year = splited_data[0]
        spots_count = splited_data[1].split('\n')[0]
        years.append(int(year))
        spots.append(int(spots_count))
    return years, spots

# Creates matrix of given data. Matrix dimension is LEN-N x N
def create_input_matrix(sunspot, n):
    training_in = []
    i = 0
    while i < len(sunspot) - n:
        line = sunspot[i:n+i]
        training_in.append(line)
        i += 1
    return np.array(training_in)

# readed data from file
data = read_data(FILE_PATH)
years, spots = split_data(data)

# normalise data
norm_spots = norm(spots, 0, 1)

# split for training and validation
train_spots = norm_spots[:200]
valid_spots = norm_spots[200:]

# converted data to matrices - ready to work with it
training_out = np.array(train_spots[N:]).reshape(-1, 1)
training_in = create_input_matrix(train_spots, N)

# learn
neur_net = NeuralNetwork(N)
neur_net.train(training_in, training_out, 100000, 0.0001)
print("Weights after training:")
print(neur_net.synaptic_weights)

# test data preparation 
valid_test_out = np.array(valid_spots[N:]).reshape(-1, 1)
valid_test_in = create_input_matrix(valid_spots, N)

# count error
res = []
for i, line in enumerate(valid_test_in):
    prediciton = neur_net.think(line)
    error = valid_test_out[i-1] - prediciton
    res.append(error**2)
    print(denorm(valid_test_out[i], spots), denorm(prediciton, spots))

# count error sum
sum = 0
for i in res:
    sum += i[0]

print(res)

# print error sum
print(sum/len(valid_test_in))
# print error denorm sum
print(denorm(sum, spots))


# plt.plot(res)
# plt.show()
