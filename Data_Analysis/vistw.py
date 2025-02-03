import pandas as pd
import matplotlib.pyplot as plt

# Load csv2df
file_path = '../time/out.csv'
df = pd.read_csv(file_path)

# Filter relevant
filtered_df = df[(df['customers'].notna()) & (df.drop(columns=['customers', 'timewindows']).isna().all(axis=1))]

#in minutes
filtered_df['timewindows'] = filtered_df['timewindows'] / 60

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['customers'], filtered_df['timewindows'], color='blue') # , label='Zeitfenster')

# Add labels and title
plt.xlabel('Kunden')
plt.ylabel('Minuten')
#plt.title('Timewindows where fe')
plt.legend()
plt.savefig('../vis/timeoutstw.svg', format='svg')

# Show plot
plt.show()