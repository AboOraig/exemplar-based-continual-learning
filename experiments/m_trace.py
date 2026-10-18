import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# =====================================================
# 1. Plot NAIVE
# =====================================================
try:
    df_naive = pd.read_csv("Naive_cifar100_curve1.csv")
    latest_naive = df_naive["date"].max()
    df_naive = df_naive[df_naive["date"] == latest_naive].sort_values("experience")

    plt.plot(df_naive["experience"],
             df_naive["stream_accuracy"],
             marker='o', linestyle='--', color='gray',
             linewidth=2, label="Naive (Baseline)")
except FileNotFoundError:
    print("Naive log file not found. Skipping.")

# =====================================================
# 2. Plot ICaRL (Looping through memory sizes)
# =====================================================
memory_sizes = [500, 1000, 2000, 5000] # Added 5000 as per your previous sweep
df_icarl_all = pd.read_csv("ICaRL_cifar100_curve3.csv") # Note: ensure filename matches

for mem in memory_sizes:
    # Filter by memory size
    df_mem = df_icarl_all[df_icarl_all["memory_size"] == mem]
    
    if not df_mem.empty:
        # Get the latest run for this specific memory size
        latest_run = df_mem["date"].max()
        df_plot = df_mem[df_mem["date"] == latest_run].sort_values("experience")
        
        plt.plot(df_plot["experience"],
                 df_plot["stream_accuracy"],
                 marker='s', # square marker
                 linewidth=2.5,
                 label=f"ICaRL (Mem={mem})")

# =====================================================
# Professional Formatting
# =====================================================
plt.xlabel("Experience ID", fontsize=13)
plt.ylabel("Avg Seen Accuracy", fontsize=13)
plt.title("Impact of Buffer Size on iCaRL Performance (CIFAR-100)", fontsize=14)

plt.ylim(0, 1.0)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)

# Clean up visual style
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()