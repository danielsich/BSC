import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# csv2Dataframe
file_path = '../output/outprpsize.csv'
df = pd.read_csv(file_path, na_values='nan')

# nan.tts column with 180
#df['tts'].replace('nan', np.nan, inplace=True)
df['tts'].fillna(180, inplace=True)

##tts2num
df['tts'] = pd.to_numeric(df['tts'])

quartiles = df.groupby('customers')['tts'].quantile([0.25, 0.5, 0.75]).unstack()

quartiles.columns = ['25%', '50%', '75%']

# Print the quartiles
print(quartiles)

#vis
plt.figure(figsize=(10, 6))
plt.plot(quartiles.index, quartiles['25%'], label='25% Quartil')
plt.plot(quartiles.index, quartiles['50%'], label='50% Quartil')
plt.plot(quartiles.index, quartiles['75%'], label='75% Quartil')
plt.yscale('log')
plt.xlabel('Kunden')
plt.ylabel('Rechenzeit')
plt.title('25%, 50%, und 75% Quartil je Kundenanzahl für das PRP')
plt.legend()
plt.grid(True)
plt.show()