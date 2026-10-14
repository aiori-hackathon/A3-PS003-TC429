
import pandas as pd

parameters_per_client = 6
bytes_per_parameter = 8

client_counts = [2, 3, 5, 10]
results = []

for clients in client_counts:
    upload_bytes = (
        clients * parameters_per_client * bytes_per_parameter
    )

    download_bytes = (
        clients * parameters_per_client * bytes_per_parameter
    )

    total_bytes = upload_bytes + download_bytes

    results.append({
        "Clients": clients,
        "Upload Bytes": upload_bytes,
        "Download Bytes": download_bytes,
        "Total Bytes": total_bytes,
        "Total KB": round(total_bytes / 1024, 3)
    })

comparison = pd.DataFrame(results)

comparison.to_csv("client_communication_comparison.csv", index=False)

print(comparison)
print("\nSaved client_communication_comparison.csv")
