
import pandas as pd
import matplotlib.pyplot as plt

# Load model comparison results
df = pd.read_csv("model_comparison.csv")

# Plot comparison
plt.figure(figsize=(8, 5))

plt.bar(
    df["Metric"],
    df["Baseline"],
    width=0.35,
    label="Baseline",
    align="edge"
)

plt.bar(
    df["Metric"],
    df["Federated"],
    width=-0.35,
    label="Federated Learning",
    align="edge"
)

plt.title("Baseline vs Federated Learning")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Metric Value")
plt.legend()
plt.tight_layout()

# Save graph
plt.savefig("final_comparison_graph.png", dpi=300)
plt.show()

print("Final comparison graph saved!")
