
import pandas as pd

# Load model comparison results
comparison = pd.read_csv("model_comparison.csv")

# Load communication cost results
communication = pd.read_csv("communication_cost.csv")

print("\n===== MODEL PERFORMANCE COMPARISON =====")
print(comparison.to_string(index=False))

print("\n===== COMMUNICATION COST RESULTS =====")
print(communication.to_string(index=False))

print("\n===== PROJECT SUMMARY =====")
print("Dataset: METR-LA")
print("Prediction: Next traffic reading using previous 5 readings")
print("Clients: 3")
print("Federated rounds: 10")
print("Communication cost: 2.812 KB (estimated)")

print("\nReport generated successfully!")
