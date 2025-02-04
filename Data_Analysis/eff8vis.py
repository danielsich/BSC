import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../eff/eff50.csv'
df = pd.read_csv(file_path)

# Group by the 'eff' column to calculate the average values
eff_averages = df.groupby('eff').mean()

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(eff_averages.index, eff_averages['costs'], color='blue', label='Kosten')

plt.xlabel('ε', fontsize=14)
plt.ylabel('€', fontsize=14)
#plt.title('Effizienz vs Kosten', fontsize=16)
plt.legend()
plt.grid(True)

# Save the plot
#plt.savefig('../vis/eff_vs_costs50.svg', format='svg')

# Show the plot
plt.show()