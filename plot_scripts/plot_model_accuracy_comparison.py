import matplotlib.pyplot as plt

# Data for models and their accuracies
models = ['LSTM', 'Transformer', 'RandomForest', 'XGBoost', 'CNN+LSTM', 'CNN+Transformer', 'CNN+LSTM+Attention']
accuracies = [0.58, 0.58, 0.57, 0.59, 0.69, 0.61, 0.76]

# Create a figure and axis with an increased size for better readability
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the bar chart with distinct colors and an edge for clarity
bars = ax.bar(models, accuracies, color='skyblue', edgecolor='black')

# Set title and labels with increased font sizes and padding
ax.set_title('Model Accuracies Comparison', fontsize=20, pad=20)
ax.set_xlabel('Models', fontsize=16, labelpad=10)
ax.set_ylabel('Accuracy', fontsize=16, labelpad=10)
ax.set_ylim(0, 1)  # Set y-axis from 0 to 1

# Annotate each bar with its accuracy value, with a vertical offset to prevent overlap
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 8),  # Increase vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14)

# Adjust x-tick labels to prevent overlapping text
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=14)

# Use tight_layout with extra padding to ensure all elements are properly spaced
plt.tight_layout(pad=3.0)

# Save the plot as an image file
plt.savefig('model_accuracies.png', dpi=300)

# Display the plot
plt.show()
