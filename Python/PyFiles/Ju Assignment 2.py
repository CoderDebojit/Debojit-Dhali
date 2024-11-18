import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Read the data
df = pd.read_csv("./Datasets/weatherHistory.csv")
# Feature selection
x = df[['Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']]
y = df['Temperature (C)']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)

####### Linear Regression #######

# Step 1: Add a column of 1's (Intercept term) to x_train
x_train_with_intercept = pd.concat([pd.Series(1, index=x_train.index, name='Intercept'), x_train], axis=1)

# Step 2: Compute X^T (transpose of x_train_with_intercept)
x_train_transpose = x_train_with_intercept.T

# Step 3: Compute X^T * X (Matrix multiplication)
xtx = np.dot(x_train_transpose.values, x_train_with_intercept.values)

# Step 4: Compute the inverse of (X^T * X)
xtx_inverse = np.linalg.inv(xtx)

# Step 5: Compute (X^T * X)^-1 * X^T (Matrix multiplication)
xtx_inv_xt = np.dot(xtx_inverse, x_train_transpose.values)

# Step 6: Multiply (X^T * X)^-1 * X^T * y_train to get the coefficients (beta)
beta = np.dot(xtx_inv_xt, y_train)

####### Ridge Regression #######

# Regularization strength for Ridge Regression
lambda_ridge = 1.0  
'''
    Ridge Regression formula:
    # β_ridge = (X^T * X + λI)^-1 * X^T * y
    # Where:
    # X^T = Transpose of x_train_with_intercept
    # I = Identity matrix (of size equal to the number of features in x_train_with_intercept)
    # λ = Regularization strength
    # y = Target values
'''
# Create the identity matrix for regularization
ridge_term = lambda_ridge * np.identity(x_train_with_intercept.shape[1])

# Compute X^T * X
xt_x = np.matmul(x_train_with_intercept.T, x_train_with_intercept)

# Add regularization term (λ * I) to X^T * X
regularized_xt_x = xt_x + ridge_term

# Invert the regularized matrix
inverse_regularized_xt_x = np.linalg.inv(regularized_xt_x)

# Compute X^T * y
xt_y = np.matmul(x_train_with_intercept.T, y_train)

# Compute β_ridge = (X^T * X + λI)^-1 * X^T * y
beta_ridge = np.matmul(inverse_regularized_xt_x, xt_y)


####### Losso Regression #######

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_with_intercept)

# Initialize variables
lambda_lasso = 0.01
alpha = 0.001  # Reduce the learning rate if necessary
iterations = 100
beta_lasso = np.zeros(x_train_scaled.shape[1])  # Shape (7,)
losses = []

# Gradient Descent Loop
for iteration in range(iterations):
    y_pred = np.dot(x_train_scaled, beta_lasso)  # Predictions
    loss = np.mean((y_train - y_pred) ** 2)  # Compute Mean Squared Error (MSE)
    loss_gradient = -np.dot(x_train_scaled.T, (y_train - y_pred)) / len(y_train)
    penalty = lambda_lasso * np.sign(beta_lasso)
    beta_lasso -= alpha * (loss_gradient + penalty)  # Update coefficients

    # Store the loss to monitor the optimization process
    losses.append(loss)
    
    # Check if the loss is increasing, break if so (early stopping)
    if iteration > 0 and losses[iteration] > losses[iteration - 1]:
        print("Loss started increasing at iteration", iteration)
        break

    # Optionally print the loss at every 10th iteration
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss: {loss}")


# The resulting beta represents the coefficients
print("Coefficients (beta):", beta)
print("Ridge Coefficients:", beta_ridge)
# Final Lasso Coefficients and Monitoring the Loss
print("Final Lasso Coefficients:", beta_lasso)
