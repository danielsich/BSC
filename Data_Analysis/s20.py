import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../servicetime/s50.csv'
df = pd.read_csv(file_path)

# Calculate the service time in minutes
df['service_time'] = df['a'] * 0.5

# Group by the 'a' column to form time windows and calculate the average values
time_window_averages = df.groupby('a').mean()

# Calculate the percentage of driver_pay relative to costs
time_window_averages['driver_pay_percentage'] = (time_window_averages['driver_pay'] / time_window_averages['costs']) * 100

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Minuten')
ax1.set_ylabel('€', color=color)
ax1.plot(time_window_averages['service_time'], time_window_averages['costs'], color=color, label='Gesamtkosten')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color = 'tab:green'
ax2.set_ylabel('%', color=color)  # We already handled the x-label with ax1
ax2.plot(time_window_averages['service_time'], time_window_averages['driver_pay_percentage'], color=color, linestyle='dashed', label='Anteil Fahrerkosten')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # Otherwise, the right y-label is slightly clipped
#plt.title('Service Time vs Costs and Driver Pay Percentage (Averaged per Time Window)')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
#plt.savefig('../vis/service_time_vs_costs_and_driver_pay_percentage_averaged50.svg', format='svg')
plt.show()