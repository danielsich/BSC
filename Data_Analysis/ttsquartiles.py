import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# csv2Dataframe
file_path = '../output/outplserver.csv'
df = pd.read_csv(file_path, na_values='nan')


##tts2num
df['tts'] = pd.to_numeric(df['tts'], errors='coerce')

# nan.tts column with 600
df['tts'].fillna(600, inplace=True)

quartiles = df.groupby('customers')['tts'].quantile([0.25, 0.5, 0.75]).unstack()

quartiles.columns = ['25%', '50%', '75%']

# Print the quartiles
print(quartiles)

#vis
plt.figure(figsize=(12,6))
plt.plot(quartiles.index, quartiles['25%'], label='25% Quartil')
plt.plot(quartiles.index, quartiles['50%'], label='50% Quartil')
plt.plot(quartiles.index, quartiles['75%'], label='75% Quartil')
plt.yscale('log')
plt.xlabel('Kunden', fontsize=14)
plt.ylabel('Rechenzeit',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.title('25%, 50%, und 75% Quartil je Kundenanzahl für das PRP')
plt.legend(fontsize = 14)
plt.grid(True)
plt.savefig('../vis/tts_quartilespl.svg', format='svg')
plt.show()