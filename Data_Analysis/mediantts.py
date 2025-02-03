import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
file_paths = {
    'outpdserver': '../output/outpdsizeserver.csv',
    'outplserver': '../output/outplserver.csv',
    'outpeserver': '../output/outpeserver.csv',
    'outprpserver': '../output/outprpsize10.csv'
}

quartiles_dict = {}

for name, file_path in file_paths.items():
    df = pd.read_csv(file_path, na_values='nan')

    # Convert tts to numeric
    df['tts'] = pd.to_numeric(df['tts'], errors='coerce')

    # Fill NaN in tts column with 600
    df['tts'].fillna(900, inplace=True)

    # Filter data to include only customers <= 34
    df_filtered = df[df['customers'] <= 34]

    # Calculate quartiles
    quartiles = df.groupby('customers')['tts'].median()
    quartiles_dict[name] = quartiles

# Plotting
plt.figure(figsize=(12, 6))
label_mapping = {
    'outpdserver': '$F_D$ Median',
    'outplserver': '$F_L$ Median',
    'outpeserver': '$F_E$ Median',
    'outprpserver': '$F_C$ Median'
}

for name, quartiles in quartiles_dict.items():
    plt.plot(quartiles.index, quartiles, label=label_mapping[name])

plt.yscale('log')
plt.xlabel('Kunden', fontsize=14)
plt.ylabel('Rechenzeit', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
#plt.title('Median tts per Customersize for Different Datasets', fontsize=16)
plt.savefig('../vis/tts_medians_comparison.svg', format='svg')
plt.show()