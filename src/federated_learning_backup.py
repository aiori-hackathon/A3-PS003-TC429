
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned data
df = pd.read_csv("clean_traffic_data.csv")

# Prepare previous 5 readings to predict the next reading
traffic = df["767523"].to_numpy()

X = []
y = []

for i in range(5, len(traffic)):
    X.append(traffic[i-5:i])
    y.append(traffic[i])

X = np.array(X)
y = np.array(y)

# Split data by time
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Divide training data among 3 clients
client_X = np.array_split(X_train, 3)
client_y = np.array_split(y_train, 3)

# Train local models
models = []

for i in range(3):
    model = LinearRegression()
    model.fit(client_X[i], client_y[i])
    models.append(model)
    print("Client", i + 1, "trained.")

# Combine local coefficients using sample-count weights
weights = np.array([len(part) for part in client_X], dtype=float)
weights = weights / weights.sum()

global_coef = sum(
    weights[i] * models[i].coef_ for i in range(3)
)
global_intercept = sum(
    weights[i] * models[i].intercept_ for i in range(3)
)

# Predict using the combined parameters
predictions = X_test @ global_coef + global_intercept

print("\nFederated model evaluation")
print("MAE:", round(mean_absolute_error(y_test, predictions), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 3))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# Save federated model predictions
federated_results = pd.DataFrame({
    "Actual Traffic": y_test,
    "Federated Prediction": predictions
})

federated_results.to_csv(
    "federated_prediction_results.csv", index=False
)

# Save metric comparison
comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Baseline": [1.407, 2.556, 0.512],
    "Federated": [
        mean_absolute_error(y_test, predictions),
        np.sqrt(mean_squared_error(y_test, predictions)),
        r2_score(y_test, predictions)
    ]
})

comparison.to_csv("model_comparison.csv", index=False)

print("\nFederated predictions saved!")
print("Model comparison saved!")

# Measure communication cost
number_of_clients = len(models)
parameters_per_client = len(models[0].coef_) + 1

bytes_per_parameter = 8

upload_bytes = (
    number_of_clients
    * parameters_per_client
    * bytes_per_parameter
)

download_bytes = (
    number_of_clients
    * parameters_per_client
    * bytes_per_parameter
)

total_bytes = upload_bytes + download_bytes

print("\nCommunication Cost Evaluation")
print("Number of clients:", number_of_clients)
print("Parameters per client:", parameters_per_client)
print("Upload communication:", upload_bytes, "bytes")
print("Download communication:", download_bytes, "bytes")
print("Total communication:", total_bytes, "bytes")
print("Total communication:", round(total_bytes / 1024, 3), "KB")

# Save communication-cost results
communication_results = pd.DataFrame({
    "Metric": [
        "Number of Clients",
        "Parameters Per Client",
        "Upload Bytes",
        "Download Bytes",
        "Total Communication Bytes",
        "Total Communication KB"
    ],
    "Value": [
        number_of_clients,
        parameters_per_client,
        upload_bytes,
        download_bytes,
        total_bytes,
        round(total_bytes / 1024, 3)
    ]
})

communication_results.to_csv(
    "communication_cost.csv", index=False
)

print("\nCommunication results saved to communication_cost.csv")
