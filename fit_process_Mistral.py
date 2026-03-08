import fitparse
import numpy as np
import pandas as pd

def lire_fichier_fit(fichier_fit):
    fitfile = fitparse.FitFile(fichier_fit)
    records = []
    for record in fitfile.get_messages('record'):
        data = {}
        for field in record:
            if field.name == 'timestamp':
                data['time'] = field.value
            elif field.name == 'speed':
                data['speed'] = field.value
        if 'speed' in data:
            records.append(data)
    df = pd.DataFrame(records)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

# Exemple d'utilisation
df = lire_fichier_fit('fit_files/21920233191_ACTIVITY.fit')

vitesse = df['speed'].values
temps = np.arange(len(vitesse))


from pykalman import KalmanFilter

# Initialisation du filtre de Kalman
kf = KalmanFilter(
    initial_state_mean=vitesse[0],
    initial_state_covariance=1,
    observation_covariance=5,
    transition_covariance=0.001
)

# Estimation de l'état (vitesse filtrée)
vitesse_filtree, _ = kf.filter(vitesse)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(temps, vitesse, label='Vitesse instantanée', alpha=0.5)
plt.plot(temps, vitesse_filtree, label='Vitesse filtrée (Kalman)', color='red')
plt.xlabel('Temps (échantillons)')
plt.ylabel('Vitesse (m/s)')
plt.legend()
plt.title('Comparaison vitesse instantanée vs vitesse filtrée')
plt.show()
