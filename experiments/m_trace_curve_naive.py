import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Load CSV
# =====================================================
csv_file = "Naive_core50_curve.csv"
df = pd.read_csv(csv_file)

# If file contains multiple runs, keep the latest one
latest_date = df["date"].max()
df = df[df["date"] == latest_date]

x = df["experience"]
y = df["stream_accuracy"]

# =====================================================
# Plot
# =====================================================
plt.figure(figsize=(8, 6))

plt.plot(x, y, marker='o', linewidth=2, label="Naive")

# Annotate each point
for xi, yi in zip(x, y):
    plt.text(xi, yi + 0.01, f"{yi:.2f}", 
             ha='center', fontsize=8)

# =====================================================
# Professional Formatting
# =====================================================
plt.xlabel("Experience", fontsize=12)
plt.ylabel("Stream Accuracy", fontsize=12)
plt.title("Naive Performance on CORe50\n(Stream Accuracy vs Experience)", fontsize=13)
#plt.title("Naive Performance on CIFAR100\n(Stream Accuracy vs Experience)", fontsize=13)

plt.ylim(0, 1)  # fixed y-axis for fair comparison
plt.xlim(left=0)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()




"""
===== FINAL METRICS naive on cifar100 =====
AA  = 0.0749
AIA = 0.2077
FM  = 0.7163
"""

"""
===== FINAL METRICS naive on core50 =====
AA  = 0.0814
AIA = 0.0800
FM  = 0.0573
"""