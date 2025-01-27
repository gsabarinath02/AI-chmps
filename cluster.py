import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Seed for reproducibility
np.random.seed(42)

# ================== Data Generation Functions ==================
def generate_2d_class_data(mean, cov, size, class_label):
    """Generate 2D data with Age and Weight_Loss features"""
    samples = np.random.multivariate_normal(mean, cov, size=size)
    return pd.DataFrame({
        'Age': np.round(samples[:, 0], 0),
        'Weight_Loss': np.round(samples[:, 1], 2),
        'Class': str(class_label)
    })

def generate_3d_class_data(mean, cov, size, class_label):
    """Generate 3D data with Age, Weight_Loss, and Height features"""
    samples = np.random.multivariate_normal(mean, cov, size=size)
    return pd.DataFrame({
        'Age': np.round(samples[:, 0], 0),
        'Weight_Loss': np.round(samples[:, 1], 2),
        'Height': np.round(samples[:, 2], 2),
        'Class': str(class_label)
    })

# ================== DataFrame Formatting Function ==================
def format_merged_dataframe(df, rows_per_class=3, exclude_columns=None):
    """
    Format dataframe with sample rows from each class and continuation dots.
    Optionally exclude specified columns.
    
    Parameters:
    - df: pandas DataFrame
    - rows_per_class: int, number of sample rows per class
    - exclude_columns: list of column names to exclude
    """
    if exclude_columns is None:
        exclude_columns = []
        
    merged = []
    classes = sorted(df['Class'].unique(), key=lambda x: int(x))
    
    for cls in classes:
        class_data = df[df['Class'] == cls]
        sample = class_data.head(rows_per_class).drop(columns=exclude_columns)
        merged.append(sample)
        
        if len(class_data) > rows_per_class:
            dots = pd.DataFrame(
                [['...' for _ in range(len(sample.columns))]], 
                columns=sample.columns
            )
            merged.append(dots)
    
    return pd.concat(merged, ignore_index=True)

# ================== 2D Data Pipeline ==================
# Generate 2D data
cov_2d = [[20., 0.], [0., 0.5]]
df1_2d = pd.concat([
    generate_2d_class_data([25, 0], cov_2d, 7, '1'),
    generate_2d_class_data([75, 6], cov_2d, 7, '1'),
    generate_2d_class_data([50, 6], cov_2d, 6, '1')
])
df2_2d = generate_2d_class_data([40, 3], cov_2d, 20, '2')
df3_2d = generate_2d_class_data([60, 6], cov_2d, 20, '3')
df4_2d = generate_2d_class_data([80, 0], cov_2d, 20, '4')
df_all_2d = pd.concat([df1_2d, df2_2d, df3_2d, df4_2d])

# 2D Clustering
kmeans_2d = KMeans(n_clusters=4, random_state=42)
df_all_2d['Cluster'] = kmeans_2d.fit_predict(df_all_2d[['Age', 'Weight_Loss']])

# ================== 3D Data Pipeline ==================
# Generate 3D data
cov_3d = [[20., 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]]
df1_3d = pd.concat([
    generate_3d_class_data([25, 0, 0], cov_3d, 7, '1'),
    generate_3d_class_data([75, 6, 6], cov_3d, 7, '1'),
    generate_3d_class_data([50, 6, 6], cov_3d, 6, '1')
])
df2_3d = generate_3d_class_data([40, 3, 3], cov_3d, 20, '2')
df3_3d = generate_3d_class_data([60, 6, 6], cov_3d, 20, '3')
df4_3d = generate_3d_class_data([80, 6, 0], cov_3d, 20, '4')
df_all_3d = pd.concat([df1_3d, df2_3d, df3_3d, df4_3d])

# 3D Clustering
kmeans_3d = KMeans(n_clusters=4, random_state=42)
df_all_3d['Cluster'] = kmeans_3d.fit_predict(df_all_3d[['Age', 'Weight_Loss', 'Height']])

# ================== Visualization Functions ==================
def plot_2d_data_with_clustering(df, centroids, title):
    """Plot 2D data before and after clustering."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # All points in same color before clustering
    axes[0].scatter(df['Age'], df['Weight_Loss'], c='gray', alpha=0.6)
    axes[0].set_title(f'{title} (Before Clustering)')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Weight Loss')

    # After clustering
    scatter = axes[1].scatter(df['Age'], df['Weight_Loss'], c=df['Cluster'], cmap='viridis', alpha=0.6)
    axes[1].scatter(centroids[:, 0], centroids[:, 1],
                    s=200, marker='X', c='red', edgecolor='black', label='Centroids')
    axes[1].set_title(f'{title} (After Clustering)')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Weight Loss')
    cbar = fig.colorbar(scatter, ax=axes[1], label='Cluster')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_3d_data_with_clustering(df, centroids, title):
    """Plot 3D data before and after clustering."""
    fig = plt.figure(figsize=(16, 8))

    # All points in same color before clustering
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(df['Age'], df['Weight_Loss'], df['Height'], c='gray', alpha=0.6)
    ax1.set_title(f'{title} (Before Clustering)')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Weight Loss')
    ax1.set_zlabel('Height')

    # After clustering
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(df['Age'], df['Weight_Loss'], df['Height'],
                          c=df['Cluster'], cmap='viridis', alpha=0.6)
    ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                s=200, marker='X', c='red', edgecolor='black', label='Centroids')
    ax2.set_title(f'{title} (After Clustering)')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Weight Loss')
    ax2.set_zlabel('Height')
    cbar = fig.colorbar(scatter, ax=ax2, shrink=0.5, aspect=10, label='Cluster')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def display_formatted_data_with_clusters(df, title):
    """Display formatted dataframe with sample rows and their clusters, excluding 'Class'."""
    formatted_df = format_merged_dataframe(df, exclude_columns=['Class'])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=formatted_df.values,
        colLabels=formatted_df.columns,
        cellLoc='center',
        loc='center'
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Highlight headers
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')

    plt.title(f'{title} (Samples with Clusters)', pad=20)
    plt.show()

# ================== Visualization ==================
# Plot 2D data
plot_2d_data_with_clustering(df_all_2d, kmeans_2d.cluster_centers_, '2D Weight Loss Analysis')

# Plot 3D data
plot_3d_data_with_clustering(df_all_3d, kmeans_3d.cluster_centers_, '3D Health Metrics Analysis')

# ================== Display and Save Data ==================
# Display samples of 2D data points with clusters
display_formatted_data_with_clusters(df_all_2d, '2D Dataset Samples')

# Display samples of 3D data points with clusters
display_formatted_data_with_clusters(df_all_3d, '3D Dataset Samples')

# ================== Data Export ==================
# Save formatted data samples without 'Class'
format_merged_dataframe(df_all_2d, exclude_columns=['Class']).to_csv('formatted_2d_sample_with_clusters.csv', index=False)
format_merged_dataframe(df_all_3d, exclude_columns=['Class']).to_csv('formatted_3d_sample_with_clusters.csv', index=False)

# Save full datasets with clusters, excluding 'Class'
df_all_2d_with_clusters = df_all_2d.drop(columns=['Class'])
df_all_3d_with_clusters = df_all_3d.drop(columns=['Class'])

df_all_2d_with_clusters.to_csv('full_2d_dataset_with_clusters.csv', index=False)
df_all_3d_with_clusters.to_csv('full_3d_dataset_with_clusters.csv', index=False)

print("Data files with clusters saved:")
print("- formatted_2d_sample_with_clusters.csv")
print("- formatted_3d_sample_with_clusters.csv")
print("- full_2d_dataset_with_clusters.csv")
print("- full_3d_dataset_with_clusters.csv")
