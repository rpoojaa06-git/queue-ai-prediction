import pandas as pd
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("data/queue_data.csv")

# Features
X = df[["arrival_rate", "queue_length", "wait_time"]]

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# 🔥 Label clusters based on avg wait time
cluster_means = df.groupby("cluster")["wait_time"].mean()

# Sort clusters (low → high)
sorted_clusters = cluster_means.sort_values().index

labels = {
    sorted_clusters[0]: "Low Traffic",
    sorted_clusters[1]: "Medium Traffic",
    sorted_clusters[2]: "Peak Hour"
}

df["traffic_level"] = df["cluster"].map(labels)

print("\n📊 Clustered Data with Labels:")
print(df[["arrival_rate", "queue_length", "wait_time", "traffic_level"]])
import matplotlib.pyplot as plt

plt.scatter(
    df["queue_length"],
    df["wait_time"],
    c=df["cluster"],
    cmap="viridis"
)

plt.xlabel("Queue Length")
plt.ylabel("Wait Time")
plt.title("Peak Hour Clustering")
plt.colorbar(label="Cluster")
plt.show()