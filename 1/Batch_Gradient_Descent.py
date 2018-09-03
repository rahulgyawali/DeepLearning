import numpy as np

def activation_function(val):
    return 1 if val > 0 else 0

def train_split(filename):
    dataset = np.loadtxt(filename + ".csv", delimiter=',')
    rows = len(dataset)
    cols = len(dataset[0])
    x = dataset[:rows, : cols - 1]
    y = dataset[:rows, cols - 1 : cols]
    temp2 = []
    for i in range(rows):
        temp1 = list(x[i])
        temp1.append(1)
        temp2.append(temp1)
    x = np.array(temp2)
    
    y = np.array(y)
    return x, y

def test(x1, x2, weights):
    val = x1 * weights[0] + x2 * weights[1] + weights[2]
    val = activation_function(val)
    return val

def batchGD(training_input, training_output, weights, learning_rate, epochs):
    m = len(training_output)
    training_input=np.asarray(training_input)
    training_output=np.asarray(training_output)
    for _ in range(epochs):
        y_hat = np.dot(training_input, weights)
        weights = weights - learning_rate * (1.0/m) * np.dot(training_input.T, y_hat-training_output)
    return weights

def main():
    filename = input("Input Gate\n")

    weights = np.random.uniform(-0.5, 0.5, (3,1))
    learning_rate = 0.001
    epochs = 500
    training_input, training_output = train_split(filename)
    print("\nInitial Weights: ", weights, "\n")
    weights=batchGD(training_input, training_output, weights, learning_rate, epochs)
    print("Weights After Training: ", weights, "\n")
    x1, x2 = [float(i) for i in input("Enter input for testing : ").split()]
    pred=test(x1, x2, weights)
    print("output :", pred)

if __name__ == '__main__':
    main()
