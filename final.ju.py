# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

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

# %% [md]
"""
## Step 1: load the data
"""
# %%
default_of_credit_clients = pd.read_csv("default_of_credit_card_clients.csv")

cols = default_of_credit_clients.iloc[0].tolist()
cols[-1] = "default"

df = default_of_credit_clients[1:].copy()
df.columns = cols

df = df.apply(pd.to_numeric, errors="coerce")

# %% [md]
"""
## Step 2: Split train/test data
"""
# %%
from sklearn.model_selection import train_test_split

X = df.drop(columns=["default"]).reset_index(drop=True)
y = df["default"].astype(int)

assert isinstance(X, pd.DataFrame)
assert isinstance(y, pd.Series)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

training_size = int(X.shape[0] * 0.8)

X_train = X.iloc[:training_size]
X_test = X.iloc[training_size:]

y_train = y.iloc[:training_size]
y_test = y.iloc[training_size:]

# %% [md]
"""
## Step 3: Convert X_train/X_test to StandardScaler 
"""
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# %% [md]
"""
## Step 4: Convert to numpy
"""
# %%
X_train = X_train_scaled.astype(float)
X_test = X_test_scaled.astype(float)

y_train = y_train.to_numpy().astype(float)
y_test = y_test.to_numpy().astype(float)


# %% [md]
"""
# Using Scikit-Learn
This is going to be our baseline model that we want to compare against a 
differentially private gradient descent model
"""

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=20000).fit(X_train, y_train)
model.predict(X_test)

# %%
np.sum(model.predict(X_test) == y_test)/X_test.shape[0]

# %% [md]
"""
# Model Prediction
"""
# %%

theta = np.zeros(X_train.shape[1])

def predict(xi, theta, bias=0):
    label = np.sign(xi @ theta + bias)
    return label

def accuracy(theta):
    return np.sum(predict(X_test, theta) == y_test)/X_test.shape[0]


# %% [md]
"""
# Gradient Descent Model
"""

# %%

def loss(theta, xi, yi):
    exponent = - yi * (xi.dot(theta))
    return np.log(1 + np.exp(exponent))

# %%
np.mean([loss(theta, x_i, y_i) for x_i, y_i in zip(X_test, y_test)])

# %%
def logistic(x):
    return 1 / (1 + np.exp(-x))


def gradient(theta, xi, yi):
    z = yi * np.dot(xi, theta)

    if z >= 0:
        exp_neg_z = np.exp(-z)
        sigma = 1 / (1 + exp_neg_z)
    else:
        exp_z = np.exp(z)
        sigma = exp_z / (1 + exp_z)

    return -yi * xi * (1 - sigma)

def avg_grad(theta, X, y):
    grads = [gradient(theta, xi, yi) for xi, yi in zip(X, y)]
    return np.mean(grads, axis=0)

def gradient_descent(iterations):
    theta = np.zeros(X_train.shape[1])

    for _ in range(iterations):
        theta = theta - avg_grad(theta, X_train, y_train)

    return theta

theta = gradient_descent(10)
accuracy(theta)
