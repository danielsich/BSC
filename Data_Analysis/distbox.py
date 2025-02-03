import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
file_paths = {
    'outpdserver': '../output/outpdsizeserver.csv',
    'outplserver': '../output/outplserver.csv',
    'outpeserver': '../output/outpeserver.csv',
    'outprpserver': '../output/outprpsize10.csv'
}

# Read the base dataset separately for comparison
base_df = pd.read_csv(file_paths['outpdserver'], na_values='nan')
base_df['distance'] = pd.to_numeric(base_df['distance'], errors='coerce')
base_df.dropna(subset=['distance'], inplace=True)
base_distances = base_df.groupby('customers')['distance'].mean()

# Prepare a DataFrame to hold percentage increases
percentage_increase_data = []

for name, file_path in file_paths.items():
    if name == 'outpdserver':
        continue  # Skip the base model itself

    df = pd.read_csv(file_path, na_values='nan')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df.dropna(subset=['distance'], inplace=True)

    # Apply cutoff for outprpserver at 8 customers
    if name == 'outprpserver':
        df = df[df['customers'] <= 10]

    # Calculate average distances per customer
    average_distances = df.groupby('customers')['distance'].mean()

    # Calculate percentage increase compared to the base model
    common_customers = average_distances.index.intersection(base_distances.index)
    percentage_increase = ((average_distances[common_customers] - base_distances[common_customers]) / base_distances[
        common_customers]) * 100

    # Append data for boxplot
    for customer, increase in percentage_increase.items():
        percentage_increase_data.append((name, customer, increase))

# Create a DataFrame for the boxplot
percentage_increase_df = pd.DataFrame(percentage_increase_data, columns=['Model', 'Customers', 'Percentage Increase'])

# Map the model names to labels
label_mapping = {
    'outplserver': '$F_L$',
    'outpeserver': '$F_E$',
    'outprpserver': '$F_C$'
}

# Replace model names with labels in the DataFrame
percentage_increase_df['Model'] = percentage_increase_df['Model'].map(label_mapping)

# Plotting
plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='Percentage Increase', data=percentage_increase_df, palette="Set3")

plt.xlabel('Modell', fontsize=14)
plt.ylabel('%', fontsize=14)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.grid(True, axis='y')
plt.savefig('../vis/percentage_increase_boxplot.svg', format='svg')

plt.show()