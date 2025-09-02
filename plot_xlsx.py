import matplotlib.pyplot as plt

# --- Data Set 1 ---
x1 = range(1, 6)
y1 = [0.5, 0.5, 0.5, 0.5, 0.9]

# --- Data Set 2 ---
x2 = range(1, 6)
y2 = [0.356014024, 0.2993969508, 0.2286206426, 0.166549457, 0.1167431883]

# --- Create Figure and Subplots ---
# Create a figure to hold the plots.
# 'figsize' is specified to ensure the plots are well-proportioned.
# '1, 2' means 1 row, 2 columns of subplots.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Plotting Data Set 1 ---
ax1.plot(x1, y1, marker="o", linestyle="-", color="b")
ax1.set_title("Sheets Accuracy Plot")
ax1.set_xlabel("Range (1 to 5)")
ax1.set_ylabel("Value")
ax1.grid(True)  # Add a grid for better readability
ax1.set_ylim(0, 1)  # Set y-axis limits for consistency

# --- Plotting Data Set 2 ---
ax2.plot(x2, y2, marker="s", linestyle="--", color="r")
ax2.set_title("Sheets loss plot")
ax2.set_xlabel("Range (1 to 5)")
ax2.set_ylabel("Value")
ax2.grid(True)  # Add a grid for better readability
ax2.set_ylim(0, 1)  # Set y-axis limits for consistency

# --- Final Touches ---
# Adjust the layout to prevent titles and labels from overlapping.
plt.tight_layout()
plt.savefig("sheets_plot.png")
# Display the plots.
plt.show()
