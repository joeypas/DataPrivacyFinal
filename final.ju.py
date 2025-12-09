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

# The loss function measures how good our model is. The training goal is to minimize the loss.
# This is the logistic loss function.
def loss(theta, bias, xi, yi):
    exponent = -np.array(yi) * (xi.dot(theta) + bias)
    exponent = np.clip(exponent, -500, 500)
    return np.log(1 + np.exp(exponent))

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

    z = np.clip(z, -500, 500)
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
    grads = [gradient(theta, bias, xi, yi) for xi, yi in zip(X, y)]
    gt, gb = zip(*grads)
    return np.mean(gt, axis=0), np.mean(gb)

def gradient_descent(iterations):
    theta = np.zeros(X_train.shape[1])
    bias = 0.0

    for _ in range(iterations):
        gtheta, gbias = avg_grad(theta, bias, X_train, y_train)
        theta -= gtheta
        bias -= gbias
    return theta, bias

# %% [md]

# %%
theta, bias = gradient_descent(10)
accuracy(theta, bias)

# %%
# We can also log out steps 
def gradient_descent_log(iterations):
    theta = np.zeros(X_train.shape[1])
    bias = 0.0

    training_loss = []
    testing_loss = []
    training_acc = []
    testing_acc = []

    for _ in range(iterations):
        gtheta, gbias = avg_grad(theta, bias, X_train, y_train)
        theta -= gtheta
        bias -= gbias
        training_loss.append(np.mean(loss(theta, bias, X_train, y_train)))
        testing_loss.append(np.mean(loss(theta, bias, X_test, y_test)))
        training_acc.append(accuracy(theta, bias))
        testing_acc.append(accuracy(theta, bias))
    return theta, bias, training_loss, testing_loss, training_acc, testing_acc

iterations = 20

theta, bias, training_loss, testing_loss, training_acc, testing_acc = gradient_descent_log(iterations)

iterations_range = np.arange(iterations)

fig, ax1 = plt.subplots(figsize=(8, 5))

# --- Loss on left y-axis ---
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(iterations_range, training_loss, label='Training Loss', color='tab:blue')
ax1.plot(iterations_range, testing_loss, label='Testing Loss', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# --- Accuracy on right y-axis ---
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:green')
ax2.plot(iterations_range, training_acc, label='Training Accuracy', color='tab:green')
ax2.plot(iterations_range, testing_acc, label='Testing Accuracy', color='tab:olive')
ax2.tick_params(axis='y', labelcolor='tab:green')

# Combined legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='center right')

plt.title("Loss and Accuracy vs Iterations")
plt.tight_layout()
plt.show()


# %% [md]
"""
# Differentially Private Gradient Descent
Here we want to add gausian noise to both the weights and the bias.

What we are doing here:
- For clipping:
    - We have to clip both bias and weights which differs from the textbook since we don't consider bias in the texbook examples

"""

# %%

def L2_clip(v, b):
    norm = np.linalg.norm(v, ord=2)
    
    if norm > b:
        return b * (v / norm)
    else:
        return v

def gradient_sum(theta, bias, X, y, bound):
    sum_theta = np.zeros_like(theta)
    sum_bias = 0.0
    for xi, yi in zip(X, y):
        gt, gb = gradient(theta, bias, xi, yi)

        grad_vec = np.append(gt, gb)
        clipped = L2_clip(grad_vec, bound)

        sum_theta += clipped[:-1]
        sum_bias += clipped[-1]

    return sum_theta, sum_bias


def perform_iter_priv(theta, bias, X, y, b, epsilon, delta):
    gtheta, gbias = gradient_sum(theta, bias, X, y, b)

    g = np.concatenate([gtheta, np.array([gbias])])

    noisy = gaussian_mech_vec(g, b, epsilon, delta)

    noisy_theta = noisy[:-1]
    noisy_bias = noisy[-1]

    n = X_train.shape[0]

    theta = theta - (noisy_theta / n)
    bias = bias - (noisy_bias / n)
    return theta, bias

def noisy_gradient_descent(iterations, epsilon, delta):
    theta = np.zeros(X_train.shape[1])
    sensitivity = 1.0
    bias = 0.0

    eps_i = epsilon / (iterations + 1)
    delta_i = delta / iterations
    
    noisy_count = laplace_mech(X_train.shape[0], 1, eps_i)

    training_loss = []
    testing_loss = []
    training_acc = []
    testing_acc = []

    for i in range(iterations):
        theta, bias = perform_iter_priv(theta, bias, X_train, y_train, sensitivity, eps_i, delta_i)
        training_loss.append(np.mean(loss(theta, bias, X_train, y_train)))
        testing_loss.append(np.mean(loss(theta, bias, X_test, y_test)))
        training_acc.append(accuracy(theta, bias))
        testing_acc.append(accuracy(theta, bias))

    return theta, bias

theta, bias = noisy_gradient_descent(20, 0.1, 1e-5)
accuracy(theta, bias)

# %%
delta = 1e-5

epsilons = [0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1]
results   = [noisy_gradient_descent(10, epsilon, delta) for epsilon in epsilons]
accs     = [accuracy(theta, bias) for theta, bias in results]

plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.plot(epsilons, accs);

# %%
def noisy_gradient_descent_log(iterations, epsilon, delta):
    theta = np.zeros(X_train.shape[1])
    sensitivity = 1.0
    bias = 0.0

    eps_i = epsilon / (iterations + 1)
    delta_i = delta / iterations
    
    noisy_count = laplace_mech(X_train.shape[0], 1, eps_i)

    training_loss = []
    testing_loss = []
    training_acc = []
    testing_acc = []

    for i in range(iterations):
        theta, bias = perform_iter_priv(theta, bias, X_train, y_train, sensitivity, eps_i, delta_i)
        training_loss.append(np.mean(loss(theta, bias, X_train, y_train)))
        testing_loss.append(np.mean(loss(theta, bias, X_test, y_test)))
        training_acc.append(accuracy(theta, bias))
        testing_acc.append(accuracy(theta, bias))

    return theta, bias, training_loss, testing_loss, training_acc, testing_acc



iterations = 20

theta, bias, training_loss, testing_loss, training_acc, testing_acc = noisy_gradient_descent_log(iterations, 0.1, 1e-5)

iterations_range = np.arange(iterations)

fig, ax1 = plt.subplots(figsize=(8, 5))

# --- Loss on left y-axis ---
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(iterations_range, training_loss, label='Training Loss', color='tab:blue')
ax1.plot(iterations_range, testing_loss, label='Testing Loss', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# --- Accuracy on right y-axis ---
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:green')
ax2.plot(iterations_range, training_acc, label='Training Accuracy', color='tab:green')
ax2.plot(iterations_range, testing_acc, label='Testing Accuracy', color='tab:olive')
ax2.tick_params(axis='y', labelcolor='tab:green')

# Combined legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='center right')

plt.title("Loss and Accuracy vs Iterations")
plt.tight_layout()
plt.show()
