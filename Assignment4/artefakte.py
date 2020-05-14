
import matplotlib.pyplot as plt
import numpy as np
from my_topomap import plot_topomap, multichannel_plot
from pathlib import Path
from scipy import signal
from sklearn.decomposition import FastICA, PCA

"""
Wir laden unsere in der letzten Woche vorverarbeiteten Daten und inspizieren zunächst die einzelnen Kanäle.
Dazu berechnen wir die Standardabweichung des Signals über die Zeit und stellen sie in einem vereinfachten
topographischen Plot dar. Anders als die topographischen Plots der letzten Wochen stellt diese Variante
des Topoplots jeden Kanal einzeln dar, ohne Interpolation.

Aufgabe 1: Überlegen Sie: Warum könnte die Standardabweichung ein geeignetes Maß 
zur Inspektion der Datenqualität sein? Warum nicht der Mittelwert? Welche Kanäle
sind laut Topoplot potentiell problematisch?


"""
fld = Path(__file__).parent.parent / "daten"
filtered_data = np.load(fld / 'filtered_data.npy')[:64,:]
filtered_data *= 1e6
timestamps = np.linspace(0, np.shape(filtered_data)[1]/1000, np.shape(filtered_data)[1])
channel_std = np.std(filtered_data, axis = 1)
plot_topomap(channel_std)



"""
Nachdem wir oben über die Zeit gemittelt haben, um die Kanäle zu vergleichen, gehen
wir jetzt den umgekehrten Weg: wir berechnen das globale Feldpotential, d.h., die
Standardabweichung über die Kanäle zu jedem Zeitpunkt, und stellen es im Zeitverlauf dar.

Aufgabe 2: Berechnen Sie das GFP. Dazu können Sie, wie oben, die Funktion np.std verwenden,
    müssen aber die Achse verändern, über die die Standardabweichung berechnet wird.
    Stellen Sie das GFP graphisch gegen die timestamps dar. Was fällt auf?
"""




#%%
"""
Als nächstes zerlegen wir unser Signal per Komponentenanalysen (ICA und PCA) in
potentielle Quellen. Zunächst veranschaulichen wir uns das Prinzip von und die
Unterschiede zwischen PCA und ICA anhand von Beispieldaten. Das folgende Beispiel
stammt aus der Dokumentation von scikit-learn, einem Python-Modul, das zahlreiche
Funktionen für maschinelles Lernen, u.a. Komponentenanalysen, bietet.
https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html

Aufgabe 3: Versuchen Sie, nachzuvollziehen, was im unten stehenden Code passiert.
    Was ist der auffällige Unterschied zwischen PCA und ICA bei der Rekonstruktion der Quellen?
"""


# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results

plt.figure()

models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()

#%%
"""
Unten wenden wir nun die ICA auf unsere EEG-Daten an und stellen die ersten 10
Komponenten graphisch dar.

Aufgabe 4: Wenden Sie analog die PCA auf unsere Daten an und stellen Sie die 
    ersten 10 Komponenten graphisch dar. Haben Sie zu einer der PCA- oder ICA-
    Komponenten eine Idee, welches neuronale Signal sie repräsentieren könnte?
    
In der Videokonferenz werden wir uns außerdem noch mit der topographischen Darstellung
von Komponenten sowie dem 'Herausrechnen' von Artefaktkomponenten aus dem Signal beschäftigen.
"""


ica = FastICA()
ICAcomps = ica.fit_transform(filtered_data.T)
multichannel_plot(timestamps, ICAcomps[:,:10])

