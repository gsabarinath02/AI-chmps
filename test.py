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
def format_merged_dataframe(df, rows_per_class=3):
    """Format dataframe with sample rows from each class and continuation dots"""
    merged = []
    classes = sorted(df['Class'].unique(), key=lambda x: int(x))
    
    for cls in classes:
        class_data = df[df['Class'] == cls]
        sample = class_data.head(rows_per_class)
        merged.append(sample)
        
        if len(class_data) > rows_per_class:
            dots = pd.DataFrame(
                [['...' for _ in range(len(df.columns))]], 
                columns=df.columns
            )
            dots['Class'] = str(cls)  # Maintain class identifier
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

# ================== Visualization ==================
def plot_with_centroids(df, centroids, title, dimensions=2):
    """Generic plotting function with centroids"""
    fig = plt.figure(figsize=(12, 6) if dimensions == 2 else (18, 8))
    
    if dimensions == 2:
        # Original Classes
        plt.subplot(121)
        for class_label, color in [('1', 'blue'), ('2', 'red'), ('3', 'green'), ('4', 'orange')]:
            class_data = df[df['Class'] == class_label]
            plt.scatter(class_data['Age'], class_data['Weight_Loss'], 
                        c=color, label=f'Class {class_label}', alpha=0.6)
        plt.title(f'2D {title}\nOriginal Classes')
        plt.xlabel('Age')
        plt.ylabel('Weight Loss')
        plt.legend()

        # Clustered Data with Centroids
        plt.subplot(122)
        scatter = plt.scatter(df['Age'], df['Weight_Loss'], c=df['Cluster'], cmap='viridis', alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    s=200, marker='X', c='red', edgecolor='black', label='Centroids')
        plt.title(f'2D {title}\nClustered Data with Centroids')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        
    else:  # 3D
        ax1 = fig.add_subplot(121, projection='3d')
        for class_label, color in [('1', 'blue'), ('2', 'red'), ('3', 'green'), ('4', 'orange')]:
            class_data = df[df['Class'] == class_label]
            ax1.scatter(class_data['Age'], class_data['Weight_Loss'], class_data['Height'],
                       c=color, label=f'Class {class_label}', alpha=0.6)
        ax1.set_title(f'3D {title}\nOriginal Classes')
        ax1.legend()

        ax2 = fig.add_subplot(122, projection='3d')
        scatter = ax2.scatter(df['Age'], df['Weight_Loss'], df['Height'],
                             c=df['Cluster'], cmap='viridis', alpha=0.6)
        ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   s=200, marker='X', c='red', edgecolor='black', label='Centroids')
        ax2.set_title(f'3D {title}\nClustered Data with Centroids')
        fig.colorbar(scatter, ax=ax2, label='Cluster')
        ax2.legend()

    plt.tight_layout()
    plt.show()

# Plot 2D data with centroids
plot_with_centroids(df_all_2d, kmeans_2d.cluster_centers_, 'Weight Loss Analysis', dimensions=2)

# Plot 3D data with centroids
plot_with_centroids(df_all_3d, kmeans_3d.cluster_centers_, 'Health Metrics Analysis', dimensions=3)

# ================== Formatted Data Display ==================
def display_formatted_data(df, title):
    """Display formatted dataframe with continuation dots"""
    formatted_df = format_merged_dataframe(df)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
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
    
    plt.title(f'Formatted Data View: {title}', pad=20)
    plt.show()

# Display formatted data
display_formatted_data(df_all_2d, '2D Dataset Sample')
display_formatted_data(df_all_3d, '3D Dataset Sample')

# ================== Data Export ==================
# Save formatted data samples
format_merged_dataframe(df_all_2d).to_csv('formatted_2d_sample.csv', index=False)
format_merged_dataframe(df_all_3d).to_csv('formatted_3d_sample.csv', index=False)

# Save full datasets
df_all_2d.to_csv('full_2d_dataset.csv', index=False)
df_all_3d.to_csv('full_3d_dataset.csv', index=False)

print("Data files saved:")
print("- formatted_2d_sample.csv\n- formatted_3d_sample.csv")
print("- full_2d_dataset.csv\n- full_3d_dataset.csv")
