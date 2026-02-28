import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import joblib
import numpy as np

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
df = pd.read_csv("data/queue_data.csv")

# ----------------------------
# 2️⃣ Train regression model (or load saved)
# ----------------------------
try:
    model = joblib.load("models/wait_time_model.pkl")
except:
    X = df[["arrival_rate","service_time","counters","queue_length"]]
    y = df["wait_time"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "models/wait_time_model.pkl")

# ----------------------------
# 3️⃣ Clustering for traffic level
# ----------------------------
X_cluster = df[["arrival_rate", "queue_length", "wait_time"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_cluster)

# Label clusters automatically
cluster_means = df.groupby("cluster")["wait_time"].mean()
sorted_clusters = cluster_means.sort_values().index
labels = {
    sorted_clusters[0]: "Low Traffic",
    sorted_clusters[1]: "Medium Traffic",
    sorted_clusters[2]: "Peak Hour"
}
df["traffic_level"] = df["cluster"].map(labels)

# ----------------------------
# 4️⃣ Streamlit UI
# ----------------------------
st.title("🧠 Queue Digital Twin & Wait Time Predictor")

# Historical Data
st.write("### Historical Data Overview")
st.dataframe(df[["arrival_rate","service_time","counters","queue_length","wait_time","traffic_level"]])

# Graph
st.write("### Queue Length vs Wait Time (Traffic Level)")
fig, ax = plt.subplots()
scatter = ax.scatter(df["queue_length"], df["wait_time"], c=df["cluster"], cmap="viridis")
ax.set_xlabel("Queue Length")
ax.set_ylabel("Wait Time")
ax.set_title("Peak Hour Clustering")
fig.colorbar(scatter, label="Cluster")
st.pyplot(fig)

# ----------------------------
# 5️⃣ User Input Prediction with Sliders
# ----------------------------
st.write("### Predict Wait Time for New Input")

arrival_rate = st.slider("Arrival Rate", min_value=0, max_value=20, value=5)
service_time = st.slider("Service Time", min_value=1, max_value=10, value=2)
counters = st.slider("Number of Counters", min_value=1, max_value=10, value=2)
queue_length = st.slider("Queue Length", min_value=0, max_value=50, value=10)

# Predict dynamically
pred = model.predict([[arrival_rate, service_time, counters, queue_length]])
st.success(f"⏱️ Predicted Wait Time: {pred[0]:.2f} minutes")

# Predict traffic level
cluster_pred = kmeans.predict([[arrival_rate, queue_length, pred[0]]])[0]
traffic_pred = labels[cluster_pred]
st.info(f"🚦 Traffic Level: {traffic_pred}")