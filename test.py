import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters for the 3D Gaussian distributions
mean_class1 = [0, 0, 0]  # Mean for class 1 (Blue)
cov_class1 = [[1, 0.3, 0.2], [0.3, 1, 0.4], [0.2, 0.4, 1]]  # Covariance for class 1

mean_class2 = [3, 3, 3]  # Mean for class 2 (Red)
cov_class2 = [[1, -0.2, 0.1], [-0.2, 1, 0.2], [0.1, 0.2, 1]]  # Covariance for class 2

mean_class3 = [-3, -3, -3]  # Mean for class 3 (Another Blue)
cov_class3 = [[1, 0.4, -0.3], [0.4, 1, -0.2], [-0.3, -0.2, 1]]  # Covariance for class 3

# Generate random samples for each class
samples_class1 = np.random.multivariate_normal(mean_class1, cov_class1, size=50)
samples_class2 = np.random.multivariate_normal(mean_class2, cov_class2, size=50)
samples_class3 = np.random.multivariate_normal(mean_class3, cov_class3, size=50)

# Create DataFrames for each class and round to 2 decimal places
df_class1 = pd.DataFrame({'X': np.round(samples_class1[:, 0], 2), 
                          'Y': np.round(samples_class1[:, 1], 2), 
                          'Z': np.round(samples_class1[:, 2], 2), 
                          'Class': '1'})

df_class2 = pd.DataFrame({'X': np.round(samples_class2[:, 0], 2), 
                          'Y': np.round(samples_class2[:, 1], 2), 
                          'Z': np.round(samples_class2[:, 2], 2), 
                          'Class': '2'})

df_class3 = pd.DataFrame({'X': np.round(samples_class3[:, 0], 2), 
                          'Y': np.round(samples_class3[:, 1], 2), 
                          'Z': np.round(samples_class3[:, 2], 2), 
                          'Class': '3'})

# Merge the DataFrames into a single DataFrame
df_all_classes = pd.concat([df_class1, df_class2, df_class3], ignore_index=True)

# Display all DataFrames
print("Class 1 DataFrame:")
print(df_class1.head())

print("\nClass 2 DataFrame:")
print(df_class2.head())

print("\nClass 3 DataFrame:")
print(df_class3.head())

print("\nMerged DataFrame (All Classes):")
print(df_all_classes.head())

# Save each DataFrame to CSV files
df_class1.to_csv('class1_data.csv', index=False)
df_class2.to_csv('class2_data.csv', index=False)
df_class3.to_csv('class3_data.csv', index=False)
df_all_classes.to_csv('all_classes_data.csv', index=False)

# Visualization (unchanged)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for merged classes
ax.scatter(df_class1['X'], df_class1['Y'], df_class1['Z'], c='blue', label='Class 1 (Blue)', alpha=0.7)
ax.scatter(df_class2['X'], df_class2['Y'], df_class2['Z'], c='red', label='Class 2 (Red)', alpha=0.7)
ax.scatter(df_class3['X'], df_class3['Y'], df_class3['Z'], c='blue', label='Class 3 (Another Blue)', alpha=0.7)

ax.set_title('3D Scatter Plot with Three Classes')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()

plt.show()
