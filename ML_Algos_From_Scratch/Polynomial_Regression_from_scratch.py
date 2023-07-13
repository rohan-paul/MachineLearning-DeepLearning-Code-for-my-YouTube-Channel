import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

X = np.random.rand(1000,1)
y = 5*((X)**(2)) + np.random.rand(1000,1)

def loss_function(y_true, y_hat):
    loss = np.mean((y_hat - y_true)**2)
    return loss

def calculate_gradients(X, y_true, y_pred):
    num_rows = X.shape[0]
    dw = (1/num_rows)*np.dot(X.T, (y_pred - y_true))
    db = (1/num_rows)*np.sum((y_pred - y_true))
    return dw, db

def create_polynomial_feature_set(X, degrees):
    t = X.copy()
    for i in degrees:
        X = np.append(X, t**i, axis=1)
    return X

def train(X, y, batch_size, degrees, epochs, lr):
    x = create_polynomial_feature_set(X, degrees)
    m, n = x.shape
    w = np.zeros((n,1))
    b = 0
    y = y.reshape(m,1)
    losses = []
    for epoch in range(epochs):
        for i in range((m-1)//batch_size + 1):
            start_i = i*batch_size
            end_i = start_i + batch_size
            x_batch = x[start_i:end_i]
            y_batch = y[start_i:end_i]
            y_hat = np.dot(x_batch, w) + b
            dw, db = calculate_gradients(x_batch, y_batch, y_hat)
            w -= lr*dw
            b -= lr*db
        l = loss_function(y, np.dot(x, w) + b)
        losses.append(l)
    return w, b, losses

def predict(X, w, b, degrees):
    x1 = create_polynomial_feature_set(X, degrees)
    return np.dot(x1, w) + b

w_trained, b_trained, losses_trained = train(X, y,
                                             batch_size=100,
                                             degrees=[2],
                                             epochs=1000,
                                             lr=0.01)

y_hat = predict(X, w_trained, b_trained, [2])

fig = plt.figure(figsize=(8,6))
plt.plot(X, y, 'y.')
plt.plot(X, y_hat, 'r.')
plt.legend(["True Data Points", "Preds from Polynomial Regression"])
plt.xlabel('X - Input')
plt.ylabel('y - target / true')
plt.title('Polynomial Regression')
plt.show()

def r2_score(y, y_hat):
    sse = np.sum((np.array(y_hat)-np.array(y))**2)
    tss = np.sum((np.array(y)-np.mean(np.array(y)))**2)
    return 1 - (sse / tss )

y_pred = predict(X, w_trained, b_trained, [2])

r2_score(y, y_pred)
