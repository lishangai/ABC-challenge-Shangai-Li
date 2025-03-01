# 各活动类型的样本数量:
# Activity Type ID
# 2806    35310
# 2807    22537
# 2808    19519
# 2809     1657
# 2810    11257
# 2811    16835
# 2812    11879
# 2813    24789
# 2814    37052
# 2815    34361
import matplotlib.pyplot as plt

# Data: Activity Type IDs and corresponding sample counts
activity_types = [2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815]
sample_counts = [35310, 22537, 19519, 1657, 11257, 16835, 11879, 24789, 37052, 34361]

# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(activity_types, sample_counts, color='lightgray')

# Add title and labels (in English)
plt.title('Acceleration Record Counts by Activity Type', fontsize=16)
plt.xlabel('Activity Type ID', fontsize=12)
plt.ylabel('Number of Acceleration Records', fontsize=12)

# Set x-axis ticks to ensure neat labeling of activity type IDs
plt.xticks(ticks=activity_types, labels=[str(x) for x in activity_types], rotation=0)

# Remove grid lines from the background
plt.grid(False)

# Adjust layout and save the figure as an image
plt.tight_layout()
plt.savefig('acceleration_record_counts_gray.png')

# Display the plot
plt.show()



