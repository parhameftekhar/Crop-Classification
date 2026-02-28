
import re
import matplotlib.pyplot as plt
import os

log_file = '/home/parham/Documents/codes/Crop-Classification/logs/fine_tuning/fine_tuning_crop1_20260225_212437.log'
output_plot = '/home/parham/Documents/codes/Crop-Classification/logs/fine_tuning/training_progress.png'

epochs = []
losses = []
f1_scores = []
val_epochs = []

# Regex patterns
loss_pattern = re.compile(r'Epoch \[(\d+)/\d+\], Average Loss: ([\-\d\.]+)')
f1_pattern = re.compile(r'F1 Score: ([\d\.]+)')

current_epoch = None

with open(log_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        # Match Training Loss
        loss_match = loss_pattern.search(line)
        if loss_match:
            current_epoch = int(loss_match.group(1))
            epochs.append(current_epoch)
            losses.append(float(loss_match.group(2)))
        
        # Match F1 Score (usually follows "Overall Results from Confusion Matrix")
        # We look for "F1 Score:" within the next few lines of a validation block
        if "Overall Results from Confusion Matrix:" in line:
            # The F1 score is typically 5 lines down in this log format
            for j in range(1, 10):
                if i + j < len(lines):
                    f1_match = f1_pattern.search(lines[i+j])
                    if f1_match:
                        f1_scores.append(float(f1_match.group(1)))
                        val_epochs.append(current_epoch)
                        break

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot Loss on primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Average Loss (Negative indicates better separation)', color=color)
ax1.plot(epochs, losses, marker='o', linestyle='-', color=color, label='Training Loss', alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.3)

# Create twin axis for F1 Score
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('F1 Score', color=color)
ax2.plot(val_epochs, f1_scores, marker='s', linestyle='--', color=color, label='Validation F1', alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1.0)

plt.title('Training Progress: Loss vs Validation F1 (Crop 1)')
fig.tight_layout()

# Save the plot
plt.savefig(output_plot, dpi=300)
print(f"Plot saved to {output_plot}")

# Optional: Also print summary
if f1_scores:
    print(f"Max F1: {max(f1_scores):.4f} at Epoch {val_epochs[f1_scores.index(max(f1_scores))]}")
    print(f"Min Loss: {min(losses):.6f} at Epoch {epochs[losses.index(min(losses))]}")
