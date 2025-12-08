# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Some useful utilities

def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)

def gaussian_mech(v, sensitivity, epsilon, delta):
    return v + np.random.normal(loc=0, scale=sensitivity * np.sqrt(2*np.log(1.25/delta)) / epsilon)

def gaussian_mech_vec(v, sensitivity, epsilon, delta):
    return v + np.random.normal(loc=0, scale=sensitivity * np.sqrt(2*np.log(1.25/delta)) / epsilon, size=len(v))

def pct_error(orig, priv):
    return np.abs(orig - priv)/orig * 100.0

def z_clip(xs, b):
    return [min(x, b) for x in xs]

def g_clip(v):
    n = np.linalg.norm(v, ord=2)
    if n > 1:
        return v / n
    else:
        return v

# %% [md]
"""
# Setup
Here we want to load our dataset, preprocessing, and split into train and test for our model
"""
# %%
# Step 1: load the data
default_of_credit_clients = pd.read_csv("default_of_credit_card_clients.csv")

cols = default_of_credit_clients.iloc[0].tolist()
cols[-1] = "default"

df = default_of_credit_clients[1:].copy()
df.columns = cols

X = df.drop(columns=["default"]).reset_index(drop=True)
y = df["default"].astype(int)


# Step 2: Split into training and testing data
training_size = int(X.shape[0] * 0.8)

x_train = X[:training_size]
x_test = X[training_size:]

y_train = y[:training_size]
y_test = y[training_size:]

## Step 3: Convert X_train/X_test to StandardScaler 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Step 4: Make sure our labels are correct
y_train = [(-1 if x == 0 else 1) for x in y_train]
y_test = [(-1 if x == 0 else 1) for x in y_test]


# %%
print(X_train[:10], X_test[:10], y_train[:10], y_test[:10])


# %% [md]
"""
# Using Scikit-Learn
This is going to be our baseline model that we want to compare against a 
differentially private gradient descent model
"""

# %%

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=100).fit(X_train, y_train)
model.predict(X_test)
model.intercept_[0], model

# %%
np.sum(model.predict(X_test) == y_test)/X_test.shape[0]

# %% [md]
"""
# Model Prediction
"""
# %%

def predict(xi, theta, bias=0.0):
    return np.sign(xi @ theta + bias)

def accuracy(theta, bias=0.0):
    return np.sum(predict(X_test, theta, bias) == y_test) / X_test.shape[0]

accuracy(model.coef_[0], model.intercept_[0])
# %% [md]
"""
# Gradient Descent Model

This differs slightly from what we did in class since including bias seems to make a very big difference
"""

# %%

def gradient(theta, bias, xi, yi):
    z = yi * (np.dot(xi, theta) + bias)

    if z >= 0:
        exp_neg_z = np.exp(-z)
        sigma = 1 / (1 + exp_neg_z)
    else:
        exp_z = np.exp(z)
        sigma = exp_z / (1 + exp_z)

    grad_theta = -yi * xi * (1 - sigma)
    grad_bias = -yi * (1 - sigma)
    return grad_theta, grad_bias

def avg_grad(theta, bias, X, y):
    Gt = []
    Gb = []
    for xi, yi in zip(X, y):
        gt, gb = gradient(theta, bias, xi, yi)
        Gt.append(gt)
        Gb.append(gb)
    return np.mean(Gt, axis=0), np.mean(Gb)

def gradient_descent(iterations):
    theta = np.zeros(X_train.shape[1])
    bias = 0.0

    for _ in range(iterations):
        gtheta, gbias = avg_grad(theta, bias, X_train, y_train)
        theta -= gtheta
        bias -= gbias
    return theta, bias

theta, bias = gradient_descent(10)
accuracy(theta, bias)

# %%
def gradient_vec(theta, bias, X, y):
    y = np.array(y)
    z = y * (X @ theta + bias)

    z = np.clip(z, -500, 500)
    sigma = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    grad_theta = -y[:, np.newaxis] * X * (1 - sigma)[:, np.newaxis]
    grad_bias = -y * (1 - sigma)

    return grad_theta, grad_bias

# We want to clip all gradients at once for speed
def L2_clip(gt, gb, b):
    gs = np.hstack([gt, gb[:, np.newaxis]])

    norms = np.linalg.norm(gs, axis=1, keepdims=True)

    clip_factors = np.minimum(1.0, b / (norms + 1e-10))
    return gs * clip_factors


def grad_sum_vec(theta, bias, X, y, b):
    gt, gb = gradient_vec(theta, bias, X, y)

    grads_clipped = L2_clip(gt, gb, b)

    sum_grad = np.sum(grads_clipped, axis=0)
    
    return sum_grad[:-1], sum_grad[-1]

def perform_iter(theta, bias, X, y, b, epsilon, delta, learning_rate):
    gtheta, gbias = grad_sum_vec(theta, bias, X, y, b)

    g = np.concatenate([gtheta, np.array([gbias])])

    noisy = gaussian_mech_vec(g, b, epsilon, delta)

    noisy_theta = noisy[:-1]
    noisy_bias = noisy[-1]

    n = len(X)

    theta = theta - learning_rate * noisy_theta / n
    bias = bias - learning_rate * noisy_bias / n
    return theta, bias

def noisy_gradient_descent(iterations, epsilon, delta):
    theta = np.zeros(X_train.shape[1])
    sensitivity = 5.0
    bias = 0.0
    learning_rate = 0.5

    eps_i = epsilon / (iterations + 1)
    delta_i = delta / iterations
    
    noisy_count = laplace_mech(X_train.shape[0], 1, eps_i)

    for i in range(iterations):
        theta, bias = perform_iter(theta, bias, X_train, y_train, sensitivity, eps_i, delta_i, learning_rate)

    return theta, bias

def get_avg(iters, eps, delta):
    results = [noisy_gradient_descent(iters, eps, delta) for _ in range(50)]
    thetas, biases = zip(*results)
    return np.mean([accuracy(theta, bias) for theta, bias in zip(thetas, biases)])

get_avg(10, 0.1, 1e-5)

# %%
delta = 1e-5

epsilons = [0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1]
accs     = [get_avg(10, epsilon, delta) for epsilon in epsilons]

plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.plot(epsilons, accs);
