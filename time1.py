import re

def extract_cumulative_checkpoint_timings(log_file_path):
    time_pattern = re.compile(r'Time:\s*([\d.]+)')
    checkpoint_pattern = re.compile(r'Model saved to .*_(\d+)\.npz')

    checkpoint_cumulative_times = {}
    current_times = []
    total_time = 0.0
    current_checkpoint = None
    checkpoint_order = []

    with open(log_file_path, 'r') as f:
        for line in f:
            # Match 'Time: ...' lines
            time_match = time_pattern.search(line)
            if time_match:
                time_val = float(time_match.group(1))
                total_time += time_val
                current_times.append(time_val)

            # Match 'Model saved to ...' lines
            checkpoint_match = checkpoint_pattern.search(line)
            if checkpoint_match:
                checkpoint_iter = int(checkpoint_match.group(1))
                checkpoint_cumulative_times[checkpoint_iter] = total_time
                checkpoint_order.append(checkpoint_iter)

    # Output
    print("✅ Cumulative time to reach each checkpoint:\n")
    for ckpt in sorted(checkpoint_order):
        time_sec = checkpoint_cumulative_times[ckpt]
        print(f"  Iter {ckpt:>6} : {time_sec:.2f} sec ≈ {time_sec/60:.2f} min")

    final = max(checkpoint_cumulative_times.values())
    print("\n🕒 Total training time:")
    print(f"   {final:.2f} sec ≈ {final/60:.2f} min ≈ {final/3600:.2f} hrs")

# Run
if __name__ == "__main__":
    log_file = "out/Re100wcp.out"  # Change to your actual output log file
    extract_cumulative_checkpoint_timings(log_file)