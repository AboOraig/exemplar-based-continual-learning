import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# SELECT MEMORY SIZES TO DISPLAY
# =====================================================
memory_sizes_to_plot = [500, 1000, 2000]

# =====================================================
# Load CSV
# =====================================================
csv_file = "ICaRL_cifar100_curve1.csv"
df = pd.read_csv(csv_file)

plt.figure(figsize=(8, 6))

for mem in memory_sizes_to_plot:

    df_mem = df[df["memory_size"] == mem]

    if df_mem.empty:
        print(f"No data found for memory_size = {mem}")
        continue

    df_mem = df_mem.sort_values("experience")

    x = df_mem["experience"]
    y = df_mem["stream_accuracy"]

    # Plot curve
    plt.plot(x, y, marker='o', label=f"Memory = {mem}")

    # Annotate each point
    for xi, yi in zip(x, y):
        plt.text(xi, yi + 0.02, f"{yi:.2f}", ha='center', fontsize=8)

# =====================================================
# Professional Formatting
# =====================================================
plt.xlabel("Experience", fontsize=12)
plt.ylabel("Stream Accuracy", fontsize=12)
#plt.title("ICaRL Performance on CORe50\n(Stream Accuracy vs Experience)", fontsize=13)
plt.title("ICaRL Performance on CIFAR100\n(Stream Accuracy vs Experience)", fontsize=13)

plt.ylim(0, 1)  # Fixed y-axis for fair comparison
plt.xlim(left=0)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()
