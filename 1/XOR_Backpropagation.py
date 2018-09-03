import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

def split_data(dataset):
    n_samples = len(dataset)
    n_features = len(dataset[0])
    train_X = dataset[ : , : n_features - 1]
    train_Y = dataset[ : , n_features - 1 : n_features]
    return train_X, train_Y

def test(model, X, Y):
    m = X.shape[0]
    prediction = model.predict(X)
    result = [round(var[0]) for var in prediction]
    print("\nAFTER TESTING :\n")
    print("X_1\t X_2\t predicted\t actual")
    acc = 0
    for i in range(m):
        print(X[i][0],"\t",X[i][1],"\t",result[i],"\t\t",Y[i][0])
        if result[i] == Y[i][0] :
            acc += 1
    acc = (acc / m) * 100
    print("Accuracy:", acc, "%")

def get_model(train_X, train_Y):
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation="tanh"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.2))
    model.fit(train_X, train_Y, batch_size=8, epochs=500)
    return model
    
def main():
    dataset = np.loadtxt("xor.csv", delimiter=',')
    train_X, train_Y = split_data(dataset)
    model = get_model(train_X, train_Y)    
    dataset = np.loadtxt("test.csv", delimiter=",")
    test_X, test_Y = split_data(dataset)
    test(model, test_X, test_Y)

if __name__ == '__main__':
    main()