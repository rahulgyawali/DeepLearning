import numpy as np

def activation_function(val):
    if val > 0 :
        return 1
    else:
        return 0

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
    return x, y

def test(weights):
    print("Enter Input for testing:")
    x, y = input().split()
    x  = float(x)
    y  = float(y)
    val = x * weights[0] + y * weights[1] + weights[2]
    print("Output: ")
    print(activation_function(val))

def main():         
    filename = input("Input Gate\n")
    weights = np.random.uniform(-0.5, 0.5, 3)  
    learning_rate = 0.01
    epochs = 500
    training_input, training_output = train_split(filename)

    print("\nInitial Weights: ", weights, "\n")
    for i in range(epochs):
        for j in range(len(training_output)):
            value = np.dot(training_input[j], np.transpose(weights))
            actual_output = activation_function(value)
            error = training_output[j] - actual_output
            weights += learning_rate * error * np.asarray(training_input[j])
    print("Weights After Training: ", weights, "\n")

    test(weights)

if __name__ == '__main__':
    main()