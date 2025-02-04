import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../servicetime/s20.csv'
df = pd.read_csv(file_path)

# Calculate the service time in minutes
df['service_time'] = df['a'] * 0.5

# Group by the 'a' column to form time windows and calculate the average values
time_window_averages = df.groupby('a').mean()

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(time_window_averages['service_time'], time_window_averages['costs'], color='blue', label='Gesamtkosten')
plt.plot(time_window_averages['service_time'], time_window_averages['driver_pay'], color='green', label='Fahrerkosten')

plt.xlabel('Minuten', fontsize=14)
plt.ylabel('€', fontsize=14)
#plt.title('Service Time vs Costs and Driver Pay (Averaged per Time Window)', fontsize=16)
plt.legend()
plt.grid(True)

# Save the plot
#plt.savefig('service_time_vs_costs_and_driver_pay_averaged.svg', format='svg')

# Show the plot
plt.show()