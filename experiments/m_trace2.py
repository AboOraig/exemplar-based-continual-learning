import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

try:
    df_naive = pd.read_csv("Naive_core50_curve.csv")
    latest_naive = df_naive["date"].max()
    df_naive = df_naive[df_naive["date"] == latest_naive].sort_values("experience")

    plt.plot(df_naive["experience"],
             df_naive["stream_accuracy"],
             marker='o', linestyle='--', color='gray',
             linewidth=2, label="Naive (Baseline)")
except FileNotFoundError:
    print("Naive log file not found. Skipping.")


# =====================================================
# Plot ICaRL on CIFAR100
# =====================================================
memory_sizes = [5000]
df_icarl_cifar = pd.read_csv("ICaRL_cifar100_curve3.csv")

for mem in memory_sizes:
    df_mem = df_icarl_cifar[df_icarl_cifar["memory_size"] == mem]

    if not df_mem.empty:
        latest_run = df_mem["date"].max()
        df_plot = df_mem[df_mem["date"] == latest_run].sort_values("experience")

        # Remove final point (only for CIFAR100)
        df_plot = df_plot.iloc[:-1]

        plt.plot(df_plot["experience"],
                 df_plot["stream_accuracy"],
                 marker='s',
                 linewidth=2.5,
                 label=f"ICaRL on CIFAR100 (Mem={mem})")

        # Add value labels
        for x, y in zip(df_plot["experience"], df_plot["stream_accuracy"]):
            plt.text(x, y + 0.015, f"{y:.2f}", 
                     ha='center', fontsize=9)


# =====================================================
# Plot ICaRL on CORe50
# =====================================================
memory_sizes = [5000]
df_icarl_core = pd.read_csv("ICaRL_core50_curve.csv")

for mem in memory_sizes:
    df_mem = df_icarl_core[df_icarl_core["memory_size"] == mem]

    if not df_mem.empty:
        latest_run = df_mem["date"].max()
        df_plot = df_mem[df_mem["date"] == latest_run].sort_values("experience")

        plt.plot(df_plot["experience"],
                 df_plot["stream_accuracy"],
                 marker='o',
                 linewidth=2.5,
                 label=f"ICaRL on CORe50 (Mem={mem})")

        # ✅ Add value labels
        for x, y in zip(df_plot["experience"], df_plot["stream_accuracy"]):
            plt.text(x, y + 0.015, f"{y:.2f}", 
                     ha='center', fontsize=9)


# =====================================================
# Professional Formatting
# =====================================================
plt.xlabel("Experience ID", fontsize=13)
plt.ylabel("Avg Seen Accuracy", fontsize=13)
plt.title("iCaRL Performance on CIFAR100 vs CORe50 (Memory = 5000)", fontsize=14)

plt.ylim(0, 1.0)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()