import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. Create Enhanced Realistic Dataset
# ---------------------------------------------------------
np.random.seed(42)

# Base salary parameters (in INR)
base_salary = {
    'High School': 300000,
    'Bachelor': 600000,
    'Master': 900000,
    'PhD': 1200000
}

# Gender bias factor (for realistic pattern)
gender_bias = {'M': 1.15, 'F': 1.0}

data = {
    'Name': [f'Person_{i}' for i in range(50)],
    'Age': np.random.randint(22, 60, 50),
    'Gender': np.random.choice(['M', 'F'], 50, p=[0.6, 0.4]),
    'Height_cm': np.random.normal(168, 10, 50).astype(int),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                50, p=[0.1, 0.4, 0.4, 0.1]),
    'Experience': np.random.randint(0, 35, 50),
    'Certifications': np.random.poisson(3, 50),
    'Department': np.random.choice(['Engineering', 'Sales', 'HR', 'Finance'], 50),
    'Location': np.random.choice(['Metro', 'Non-Metro'], 50, p=[0.7, 0.3])
}

# Salary calculation with realistic patterns
salaries = []
for i in range(50):
    base = base_salary[data['Education'][i]]
    experience_bonus = data['Experience'][i] * 25000
    cert_bonus = data['Certifications'][i] * 30000
    location_multiplier = 1.2 if data['Location'][i] == 'Metro' else 1.0
    gender_multiplier = gender_bias[data['Gender'][i]]
    
    salary = (base + experience_bonus + cert_bonus) * location_multiplier * gender_multiplier
    salaries.append(salary * np.random.uniform(0.9, 1.1))  # Add random variation

# Add one obvious outlier
salaries[25] = 5_500_000  # Extreme outlier
data['Salary'] = np.round(salaries, -3)  # Round to nearest 1000

df = pd.DataFrame(data)

# ---------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------
# Encode education levels
edu_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['Education_Level'] = df['Education'].map(edu_mapping)

# Remove outlier
df_clean = df[df['Salary'] < 5_000_000].copy()

print(f"Original data: {len(df)} records")
print(f"Cleaned data: {len(df_clean)} records\n")

# ---------------------------------------------------------
# 3. Visualization: Before/After Outlier Removal
# ---------------------------------------------------------
plt.figure(figsize=(15, 10))

# Before outlier removal
plt.subplot(2, 2, 1)
sns.scatterplot(x='Experience', y='Salary', data=df, color='red')
plt.title("1. Salary vs Experience (With Outlier)\nNo Clear Pattern")

# After outlier removal
plt.subplot(2, 2, 2)
sns.regplot(x='Experience', y='Salary', data=df_clean, ci=None, 
            line_kws={'color': 'red'}, scatter_kws={'alpha': 0.6})
plt.title("2. After Outlier Removal\nClear Positive Correlation")

# Gender comparison
plt.subplot(2, 2, 3)
sns.boxplot(x='Gender', y='Salary', data=df_clean)
plt.title("3. Gender Salary Comparison\nMale Premium Visible")

# Height vs Salary
plt.subplot(2, 2, 4)
sns.scatterplot(x='Height_cm', y='Salary', data=df_clean)
plt.title("4. Height vs Salary\nNo Clear Relationship")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. Enhanced 2D Relationships
# ---------------------------------------------------------
plt.figure(figsize=(15, 10))

# Education Level
plt.subplot(2, 2, 1)
sns.boxplot(x='Education', y='Salary', data=df_clean, order=edu_mapping.keys())
plt.title("Education Level vs Salary\nClear Progression Visible")

# Certifications
plt.subplot(2, 2, 2)
sns.regplot(x='Certifications', y='Salary', data=df_clean, ci=None,
            line_kws={'color': 'green'}, x_jitter=0.3)
plt.title("Certifications vs Salary\nPositive Correlation")

# Location Impact
plt.subplot(2, 2, 3)
sns.boxplot(x='Location', y='Salary', data=df_clean)
plt.title("Location Impact\nMetro vs Non-Metro Salaries")

# Department Comparison
plt.subplot(2, 2, 4)
sns.boxplot(x='Department', y='Salary', data=df_clean)
plt.title("Department-wise Salary Distribution\nEngineering Leads")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. 3D Relationships
# ---------------------------------------------------------
# 3D Plot 1: Experience, Education, Salary
fig = plt.figure(figsize=(16, 6))

# Plot 1: 3D Scatter with Regression Plane
ax1 = fig.add_subplot(121, projection='3d')
X = df_clean[['Experience', 'Education_Level']]
y = df_clean['Salary']

model = LinearRegression().fit(X, y)

# Create grid for plane
x1_range = np.linspace(X['Experience'].min(), X['Experience'].max(), 10)
x2_range = np.linspace(X['Education_Level'].min(), X['Education_Level'].max(), 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = model.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

ax1.scatter(X['Experience'], X['Education_Level'], y, c='blue', alpha=0.6)
ax1.plot_surface(X1, X2, Z, alpha=0.5, color='orange')
ax1.set_xlabel('Years of Experience')
ax1.set_ylabel('Education Level')
ax1.set_zlabel('Salary (INR)')
ax1.set_title("3D: Experience + Education vs Salary\nCombined Impact Plane")

# 3D Plot 2: Age, Certifications, Salary
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(df_clean['Age'], df_clean['Certifications'], df_clean['Salary'],
           c='green', alpha=0.6)
ax2.set_xlabel('Age')
ax2.set_ylabel('Certifications')
ax2.set_zlabel('Salary (INR)')
ax2.set_title("3D: Age + Certifications vs Salary\nComplex Interaction")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. Advanced Analysis: Gender Gap Across Departments
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='Salary', hue='Gender', data=df_clean)
plt.title("Gender Pay Gap Across Departments\nConsistent Male Premium")
plt.legend(title='Gender')
plt.show()