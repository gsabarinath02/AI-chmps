import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Seed for reproducibility
np.random.seed(42)

# Function to generate DataFrame for a class
def generate_class_data(mean, cov, size, class_label):
    samples = np.random.multivariate_normal(mean, cov, size=size)
    df = pd.DataFrame({
        'X': np.round(samples[:, 0], 2),
        'Y': np.round(samples[:, 1], 2),
        'Z': np.round(samples[:, 2], 2),
        'Class': str(class_label)
    })
    return df

# Covariance matrix (same for all classes)
covariance = [[0.5, 0., 0.],
             [0., 0.5, 0.],
             [0., 0., 0.5]]

# Generate data for Class 1 (Blue)
mean_class1_part1 = [0, 0, 0]
df_class1_part1 = generate_class_data(mean_class1_part1, covariance, size=7, class_label='1')

mean_class1_part2 = [6, 6, 6]
df_class1_part2 = generate_class_data(mean_class1_part2, covariance, size=7, class_label='1')

mean_class1_part3 = [-6, 6, 6]
df_class1_part3 = generate_class_data(mean_class1_part3, covariance, size=6, class_label='1')

df_class1 = pd.concat([df_class1_part1, df_class1_part2, df_class1_part3], ignore_index=True)

# Generate data for Class 2 (Red)
mean_class2 = [-3, 3, 3]
df_class2 = generate_class_data(mean_class2, covariance, size=20, class_label='2')

# Generate data for Class 3 (Green)
mean_class3 = [0, 6, 6]
df_class3 = generate_class_data(mean_class3, covariance, size=20, class_label='3')

# Generate data for Class 4 (Orange)
mean_class4 = [0, 6, 0]
df_class4 = generate_class_data(mean_class4, covariance, size=20, class_label='4')

# Merge all classes into a single DataFrame
df_all_classes = pd.concat([df_class1, df_class2, df_class3, df_class4], ignore_index=True)

# Visualization: 3D Scatter Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each class
ax.scatter(df_class1['X'], df_class1['Y'], df_class1['Z'], c='blue', label='Class 1 (Blue)', alpha=0.7)
ax.scatter(df_class2['X'], df_class2['Y'], df_class2['Z'], c='red', label='Class 2 (Red)', alpha=0.7)
ax.scatter(df_class3['X'], df_class3['Y'], df_class3['Z'], c='green', label='Class 3 (Green)', alpha=0.7)
ax.scatter(df_class4['X'], df_class4['Y'], df_class4['Z'], c='orange', label='Class 4 (Orange)', alpha=0.7)

ax.set_title('3D Scatter Plot with Four Classes', fontsize=16, fontweight='bold')
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.set_zlabel('Z-axis', fontsize=12)
ax.legend(fontsize=12)

plt.show()

# Enhanced Function to Display DataFrame in a Separate Window Using Matplotlib Table
def display_dataframe(df, title):
    # Define table colors
    header_color = '#40466e'  # Dark blue
    row_colors = ['#f1f1f2', '#ffffff']  # Alternating light grey and white
    cell_text = df.values.tolist()
    
    # Create a new figure with adequate size
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.4 * len(df)))  # Dynamic height based on number of rows
    ax.axis('off')  # Hide axes

    # Create table
    table = ax.table(cellText=cell_text,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=[header_color]*len(df.columns))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Increased font size for readability
    table.scale(1, 1.5)  # Increased cell height

    # Apply alternating row colors
    for i, key in enumerate(table.get_celld().keys()):
        if key[0] == 0:
            # Header row
            table[key].set_facecolor(header_color)
            table[key].set_text_props(color='w', weight='bold')
        else:
            # Data rows
            table[key].set_facecolor(row_colors[i % 2])
    
    # Add borders
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
    
    # Set title
    plt.title(title, fontsize=16, fontweight='bold', pad=5)
    
    plt.tight_layout()
    plt.show()

# Function to format merged DataFrame with 4 rows per class and dots
def format_merged_dataframe(df, rows_per_class=4):
    merged_sample_rows = []
    classes = sorted(df['Class'].unique(), key=lambda x: int(x))  # Sort classes numerically

    for cls in classes:
        class_df = df[df['Class'] == cls].head(rows_per_class)
        merged_sample_rows.append(class_df)
        # Add a row of dots to indicate continuation, if more rows exist for the class
        if len(df[df['Class'] == cls]) > rows_per_class:
            dots = pd.DataFrame([['...', '...', '...', '..']], columns=df.columns)
            merged_sample_rows.append(dots)

    # Concatenate all sampled rows
    df_merged_formatted = pd.concat(merged_sample_rows, ignore_index=True)
    return df_merged_formatted

# Display individual class DataFrames with enhanced tables
# display_dataframe(df_class1.head(5), "Class 1 DataFrame (First 5 Rows)")
# display_dataframe(df_class2.head(5), "Class 2 DataFrame (First 5 Rows)")
# display_dataframe(df_class3.head(5), "Class 3 DataFrame (First 5 Rows)")
# display_dataframe(df_class4.head(5), "Class 4 DataFrame (First 5 Rows)")

# Format the merged DataFrame
df_merged_formatted = format_merged_dataframe(df_all_classes, rows_per_class=4)

# Display the formatted merged DataFrame
display_dataframe(df_merged_formatted, "Merged DataFrame (Sample Rows)")

# Display the first few rows of each DataFrame in the console (optional)
print("Class 1 DataFrame:")
print(df_class1.head())

print("\nClass 2 DataFrame:")
print(df_class2.head())

print("\nClass 3 DataFrame:")
print(df_class3.head())

print("\nClass 4 DataFrame:")
print(df_class4.head())

print("\nMerged DataFrame (Sample Rows):")
print(df_merged_formatted)

# Save each DataFrame to CSV files
df_class1.to_csv('class1_data.csv', index=False)
df_class2.to_csv('class2_data.csv', index=False)
df_class3.to_csv('class3_data.csv', index=False)
df_class4.to_csv('class4_data.csv', index=False)
df_all_classes.to_csv('all_classes_data.csv', index=False)
df_merged_formatted.to_csv('merged_sample_data.csv', index=False)

print("\nAll DataFrames have been saved to CSV files.")
