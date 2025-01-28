import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import pandas as pd

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
covariance = [[20., 0., 0.],
              [0., 0.5, 0.],
              [0., 0., 0.5]]

# Generate data for Class 1 (Blue)
mean_class1_part1 = [25, 0, 0]
df_class1_part1 = generate_class_data(mean_class1_part1, covariance, size=7, class_label='1')

mean_class1_part2 = [75, 6, 6]
df_class1_part2 = generate_class_data(mean_class1_part2, covariance, size=7, class_label='1')

mean_class1_part3 = [50, 6, 6]
df_class1_part3 = generate_class_data(mean_class1_part3, covariance, size=6, class_label='1')

df_class1 = pd.concat([df_class1_part1, df_class1_part2, df_class1_part3], ignore_index=True)

# Generate data for other classes
mean_class2 = [40, 3, 3]
df_class2 = generate_class_data(mean_class2, covariance, size=20, class_label='2')

mean_class3 = [60, 6, 6]
df_class3 = generate_class_data(mean_class3, covariance, size=20, class_label='3')

mean_class4 = [80, 6, 0]
df_class4 = generate_class_data(mean_class4, covariance, size=20, class_label='4')

# Merge all classes into a single DataFrame
df_all_classes = pd.concat([df_class1, df_class2, df_class3, df_class4], ignore_index=True)

# Function to plot an ellipsoid
def plot_ellipsoid(ax, points, color, alpha=0.2):
    mean = points.mean(axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = np.sqrt(chi2.ppf(0.95, df=3)) * np.sqrt(eigvals)  # 95% confidence
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = radii[0] * np.cos(u) * np.sin(v)
    y = radii[1] * np.sin(u) * np.sin(v)
    z = radii[2] * np.cos(v)
    ellipsoid = np.dot(np.array([x.ravel(), y.ravel(), z.ravel()]).T, eigvecs.T) + mean

    x_ellipsoid = ellipsoid[:, 0].reshape(x.shape)
    y_ellipsoid = ellipsoid[:, 1].reshape(y.shape)
    z_ellipsoid = ellipsoid[:, 2].reshape(z.shape)

    ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, color=color, alpha=alpha, rstride=4, cstride=4, linewidth=0)

# Visualization: 3D Scatter Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Class 1 parts with separate ellipsoids but single legend entry
first_class1_part = True
for df, color in zip(
    [df_class1_part1, df_class1_part2, df_class1_part3],
    ['blue', 'blue', 'blue']
):
    points = df[['Age', 'Weight_Loss', 'Height']].values
    ax.scatter(
        df['Age'], df['Weight_Loss'], df['Height'],
        c=color, label='Class 1' if first_class1_part else None,  # Label only once
        alpha=0.7
    )
    plot_ellipsoid(ax, points, color, alpha=0.2)
    first_class1_part = False

# Scatter plot and ellipsoid for other classes
for df, color, label in zip(
    [df_class2, df_class3, df_class4],
    ['red', 'green', 'orange'],
    ['Class 2', 'Class 3', 'Class 4']
):
    points = df[['Age', 'Weight_Loss', 'Height']].values
    ax.scatter(df['Age'], df['Weight_Loss'], df['Height'], c=color, label=label, alpha=0.7)
    plot_ellipsoid(ax, points, color, alpha=0.2)

# Set plot titles and labels
ax.set_title('3D Scatter Plot with Ellipsoids for Four Classes', fontsize=16, fontweight='bold')
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Weight_Loss', fontsize=12)
ax.set_zlabel('Height', fontsize=12)
ax.legend(fontsize=12)

plt.show()
