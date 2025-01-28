import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Seed for reproducibility
np.random.seed(42)

# Function to generate DataFrame for a class with renamed columns
def generate_class_data(mean, cov, size, class_label):
    samples = np.random.multivariate_normal(mean, cov, size=size)
    df = pd.DataFrame({
        'Age': np.round(samples[:, 0], 0),
        'Weight_Loss': np.round(samples[:, 1], 2),
        'Height': np.round(samples[:, 2], 2),
        'Class': str(class_label)
    })
    return df

# Covariance matrix (same for all classes)
covariance = [
    [20., 0., 0.],
    [0., 0.5, 0.],
    [0., 0., 0.5]
]

# Generate data for Class 1 (Blue)
mean_class1_part1 = [25, 0, 0]
df_class1_part1 = generate_class_data(mean_class1_part1, covariance, size=7, class_label='1')

# mean_class1_part2 = [75, 6, 6]
# df_class1_part2 = generate_class_data(mean_class1_part2, covariance, size=7, class_label='1')

# mean_class1_part3 = [50, 6, 6]
# df_class1_part3 = generate_class_data(mean_class1_part3, covariance, size=6, class_label='1')

df_class1 = pd.concat([df_class1_part1], ignore_index=True)

# Generate data for Class 2 (Red)
mean_class2 = [40, 3, 3]
df_class2 = generate_class_data(mean_class2, covariance, size=20, class_label='2')

# Generate data for Class 3 (Green)
mean_class3 = [60, 6, 6]
df_class3 = generate_class_data(mean_class3, covariance, size=20, class_label='3')

# Generate data for Class 4 (Orange)
mean_class4 = [80, 6, 0]
df_class4 = generate_class_data(mean_class4, covariance, size=20, class_label='4')

# Merge all classes into a single DataFrame
df_all_classes = pd.concat([df_class1, df_class2, df_class3, df_class4], ignore_index=True)

# Prepare feature matrix and class labels
X = df_all_classes[['Age', 'Weight_Loss', 'Height']].values
y = df_all_classes['Class'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# List of unique classes
classes = sorted(df_all_classes['Class'].unique(), key=lambda x: int(x))

# Define color mapping for classes
class_color_map = {
    '1': 'blue',
    '2': 'red',
    '3': 'green',
    '4': 'orange'
}

# Function to plot separating plane for a given class
def plot_separating_plane(ax, svm, class_label, scaler, color):
    """
    Plots the separating plane for a given class on the provided Axes3D object.
    """
    # Extract coefficients and intercept from the trained SVM
    coef = svm.coef_[0]
    intercept = svm.intercept_[0]

    # Create mesh grid in original feature space
    age_grid, weight_loss_grid = np.meshgrid(
        np.linspace(df_all_classes['Age'].min()-10, df_all_classes['Age'].max()+10, 50),
        np.linspace(df_all_classes['Weight_Loss'].min()-2, df_all_classes['Weight_Loss'].max()+2, 50)
    )

    # Scale the grid points using the same scaler
    scaled_age = (age_grid - scaler.mean_[0]) / scaler.scale_[0]
    scaled_weight_loss = (weight_loss_grid - scaler.mean_[1]) / scaler.scale_[1]

    # Calculate corresponding Height values using the decision function
    scaled_height = (-(coef[0] * scaled_age + coef[1] * scaled_weight_loss + intercept)) / coef[2]
    
    # Unscale the height to original feature space
    original_height = scaled_height * scaler.scale_[2] + scaler.mean_[2]

    # Plot the surface
    ax.plot_surface(age_grid, weight_loss_grid, original_height, 
                    alpha=0.4, color=color, edgecolor='none', antialiased=True)

# Function to plot individual class planes
def plot_individual_planes(df, X_scaled, y, classes, class_color_map, scaler):
    """
    Plots separate 3D scatter plots for each class with their corresponding separating planes.
    """
    for cls in classes:
        # Create binary labels
        binary_y = np.where(y == cls, 1, 0)

        # Train SVM on scaled data
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X_scaled, binary_y)

        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all points
        for c in classes:
            ax.scatter(
                df[df['Class'] == c]['Age'],
                df[df['Class'] == c]['Weight_Loss'],
                df[df['Class'] == c]['Height'],
                c=class_color_map[c],
                label=f'Class {c}',
                alpha=0.7,
                s=40
            )

        # Plot separating plane
        plot_separating_plane(ax, svm, cls, scaler, class_color_map[cls])

        # Formatting
        ax.set_title(f'Separating Plane for Class {cls}', fontsize=16, pad=20)
        ax.set_xlabel('Age', fontsize=12, labelpad=10)
        ax.set_ylabel('Weight Loss', fontsize=12, labelpad=10)
        ax.set_zlabel('Height', fontsize=12, labelpad=10)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        ax.zaxis.set_tick_params(labelsize=10)
        ax.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()

# Function to plot combined separating planes
def plot_combined_planes(df, X_scaled, y, classes, class_color_map, scaler):
    """
    Plots all classes and their separating planes in one 3D plot.
    """
    # Train all SVMs
    svm_models = {}
    for cls in classes:
        binary_y = np.where(y == cls, 1, 0)
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X_scaled, binary_y)
        svm_models[cls] = svm

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    for c in classes:
        ax.scatter(
            df[df['Class'] == c]['Age'],
            df[df['Class'] == c]['Weight_Loss'],
            df[df['Class'] == c]['Height'],
            c=class_color_map[c],
            label=f'Class {c}',
            alpha=0.7,
            s=40
        )

    # Plot all planes
    for cls in classes:
        plot_separating_plane(ax, svm_models[cls], cls, scaler, class_color_map[cls])

    # Formatting
    ax.set_title('Combined View of All Separating Planes', fontsize=16, pad=20)
    ax.set_xlabel('Age', fontsize=12, labelpad=10)
    ax.set_ylabel('Weight Loss', fontsize=12, labelpad=10)
    ax.set_zlabel('Height', fontsize=12, labelpad=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.zaxis.set_tick_params(labelsize=10)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()

# Function to display DataFrame sample
def display_dataframe(df, title):
    df_sample = pd.concat([df[df['Class'] == c].head(3) for c in classes] + 
                         [pd.DataFrame([['...']*4], columns=df.columns)] * (len(classes)-1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table = ax.table(cellText=df_sample.values, 
                    colLabels=df_sample.columns, 
                    cellLoc='center',
                    loc='center',
                    colColours=['#40466e']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    for k, cell in table._cells.items():
        if k[0] == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f1f1f2' if k[0]%2 else 'white')
    
    plt.title(title, fontsize=14, pad=20)
    plt.show()

# Generate and display visualizations
print("Generating visualizations...\n")

# Display sample data
display_dataframe(df_all_classes, "Sample Dataset Structure")

# Plot individual class planes
plot_individual_planes(df_all_classes, X_scaled, y, classes, class_color_map, scaler)

# Plot combined view
plot_combined_planes(df_all_classes, X_scaled, y, classes, class_color_map, scaler)

# Save data to CSV
df_all_classes.to_csv('all_classes_data.csv', index=False)
print("\nData saved to 'all_classes_data.csv'")