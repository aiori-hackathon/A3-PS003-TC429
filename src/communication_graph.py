import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("client_communication_comparison.csv")
plt.figure(figsize=(8, 5))
plt.plot(df["Clients"], df["Total KB"], marker="o")
plt.xlabel("Number of Clients")
plt.ylabel("Estimated Communication (KB)")
plt.title("Federated Learning Communication Cost")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("communication_graph.png", dpi=200)
plt.close()
print("Saved communication_graph.png")
