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

# Prepare a DataFrame to hold average speed data
average_speed_data = []

for name, file_path in file_paths.items():
    df = pd.read_csv(file_path, na_values='nan')
    df['averagespeed'] = pd.to_numeric(df['averagespeed'], errors='coerce')
    df.dropna(subset=['averagespeed'], inplace=True)

    # Filter entries with up to 9 customers
    df = df[df['customers'] <= 9]

    # Convert average speed from m/s to km/h
    df['averagespeed_kmh'] = df['averagespeed'] * 3.6

    # Collect average speed data
    for _, row in df.iterrows():
        average_speed_data.append((name, row['averagespeed_kmh']))

# Create a DataFrame for the boxplot
average_speed_df = pd.DataFrame(average_speed_data, columns=['Model', 'Average Speed (km/h)'])

# Map the model names to labels
label_mapping = {
    'outpdserver': '$F_D$',
    'outplserver': '$F_L$',
    'outpeserver': '$F_E$',
    'outprpserver': '$F_C$'
}

# Replace model names with labels in the DataFrame
average_speed_df['Model'] = average_speed_df['Model'].map(label_mapping)

# Output the summary statistics to the terminal
summary_stats = average_speed_df.groupby('Model')['Average Speed (km/h)'].describe(percentiles=[.25, .5, .75])
print(summary_stats[['25%', '50%', '75%']])

# Plotting
plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='Average Speed (km/h)', data=average_speed_df, palette="Set3")

plt.xlabel('Modell', fontsize=14)
plt.ylabel('km/h', fontsize=14)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.grid(True, axis='y')
plt.savefig('../vis/average_speed_boxplot_kmh.svg', format='svg')

plt.show()