import math

def mean(data):
    sum = 0
    for d in data:
        sum += d
    return sum/len(data)

def standDev(data):
    m = mean(data)
    sum = 0
    for d in data:
        sum += (d - m) **2
    return math.sqrt(sum/len(data))

def stand(data):
    m = mean(data)
    sd = standDev(data)
    norm_data = []
    for d in data:
        norm = (d - m) / sd
        norm_data.append(norm)
    return norm_data

def norm(data, LOW, HIGH):
    new_data = []
    for d in data:
        temp = (d - min(data))/(max(data)-min(data)) * (HIGH - LOW) + LOW
        new_data.append(temp)
    return new_data

def denorm(x, data):
    return x * (max(data)-min(data)) + min(data)