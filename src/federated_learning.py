
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv("clean_traffic_data.csv")
traffic = df["767523"].to_numpy()

# 2. Use previous 5 readings to predict the next
X = np.array([traffic[i-5:i] for i in range(5, len(traffic))])
y = traffic[5:]

# 3. Split data by time
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Scale input features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Divide training data among 3 clients
number_of_clients = 3
client_X = np.array_split(X_train, number_of_clients)
client_y = np.array_split(y_train, number_of_clients)

# 6. Initialize global model
global_coef = np.zeros(X_train.shape[1])
global_intercept = 0.0
local_epochs = 5
learning_rate = 0.001
federated_rounds = 10

# 7. Federated training
for round_no in range(federated_rounds):
    local_parameters = []
    client_sizes = []

    for i in range(number_of_clients):
        model = SGDRegressor(
            loss="squared_error",
            penalty=None,
            max_iter=1,
            tol=None,
            learning_rate="constant",
            eta0=learning_rate,
            random_state=42
        )

        # Initialize local model with global parameters
        model.coef_ = global_coef.copy()
        model.intercept_ = np.array([global_intercept])
        model.n_features_in_ = X_train.shape[1]

        for epoch in range(local_epochs):
            model.partial_fit(client_X[i], client_y[i])

        local_parameters.append(
            (model.coef_.copy(), model.intercept_[0])
        )
        client_sizes.append(len(client_X[i]))

    # Weighted average (FedAvg)
    weights = np.array(client_sizes, dtype=float)
    weights /= weights.sum()

    global_coef = sum(
        weights[i] * local_parameters[i][0]
        for i in range(number_of_clients)
    )
    global_intercept = sum(
        weights[i] * local_parameters[i][1]
        for i in range(number_of_clients)
    )

    print(f"Federated round {round_no + 1} completed.")

# 8. Evaluate federated model
federated_pred = X_test @ global_coef + global_intercept

federated_mae = mean_absolute_error(y_test, federated_pred)
federated_rmse = np.sqrt(mean_squared_error(y_test, federated_pred))
federated_r2 = r2_score(y_test, federated_pred)

print("\nFederated Model Evaluation")
print(f"MAE: {federated_mae:.3f}")
print(f"RMSE: {federated_rmse:.3f}")
print(f"R2 Score: {federated_r2:.3f}")

# 9. Save federated predictions
pd.DataFrame({
    "Actual Traffic": y_test,
    "Federated Prediction": federated_pred
}).to_csv("federated_prediction_results.csv", index=False)

# 10. Calculate baseline metrics from saved predictions
baseline = pd.read_csv("prediction_results.csv")
actual = baseline["Actual Traffic"].to_numpy()
baseline_pred = baseline["Predicted Traffic"].to_numpy()

comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Baseline": [
        mean_absolute_error(actual, baseline_pred),
        np.sqrt(mean_squared_error(actual, baseline_pred)),
        r2_score(actual, baseline_pred)
    ],
    "Federated": [
        federated_mae, federated_rmse, federated_r2
    ]
})
comparison.to_csv("model_comparison.csv", index=False)

# 11. Estimate communication cost
parameters_per_model = len(global_coef) + 1
bytes_per_parameter = 8

upload_bytes = (
    number_of_clients * parameters_per_model
    * bytes_per_parameter * federated_rounds
)
download_bytes = upload_bytes
total_bytes = upload_bytes + download_bytes

communication_results = pd.DataFrame({
    "Metric": [
        "Number of Clients",
        "Federated Rounds",
        "Parameters Per Model",
        "Upload Bytes",
        "Download Bytes",
        "Total Communication Bytes",
        "Total Communication KB"
    ],
    "Value": [
        number_of_clients, federated_rounds,
        parameters_per_model, upload_bytes, download_bytes,
        total_bytes, round(total_bytes / 1024, 3)
    ]
})
communication_results.to_csv("communication_cost.csv", index=False)

print("\nCommunication Cost Estimate")
print(f"Upload: {upload_bytes} bytes")
print(f"Download: {download_bytes} bytes")
print(f"Total: {total_bytes} bytes")
print(f"Total: {total_bytes / 1024:.3f} KiB")
print("\nPredictions saved!")
print("Model comparison saved!")
print("Communication results saved!")
