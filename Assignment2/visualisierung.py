# %%
"""
Übung 2.1
*********

Visualisierung eindimensionaler Arrays

Eindimensionale Arrays sind eine der häufigsten Datentypen, und kommen z.B. als Zeitserien vor. 
Eine typische Zeitserie ist der Spannungsverlauf einer EEG-Elektrode im Verlauf einer Messung, oder der tägliche Aktienkurse.

Geplottet wird in python meist mit matplotlib.
Matplotlib hat sowohl ein funktionales als auch ein objektorientiertes Interface. In der folgenden Zell verwenden wir den objektorienten Ansatz.

Zuerst erstellen wir eine Figure (fig) und darin eine Axes (ax). 
In diese axes kann dann geplottet werden. 
Übergibt man kein zweites Argument, erzeugt matplotlib die x-achse einfach aus der Länge der y-werte. Übergibt man zudem x-werte, wird die Beschriftung automatisch angepasst. Zudem kann man dadurch auch unregelmäßige Verläufe kongruent plotten.

Erkunden Sie andere Funktionen für die Beziehung zwischen x auf y und z

"""
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-10, 10, 0.001)
y = x ** 2
z = x * 10

fig, ax = plt.subplots()
ax.plot(y)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, z)


fig, ax = plt.subplots()
x1 = np.arange(5, 10, 0.001)
y1 = x1 ** 2

x2 = np.arange(-10, 4.5, 0.001)
y2 = x2 ** 2
ax.plot(x1, y1)
ax.plot(x2, y2)
ax.plot(x, z)


# %%
"""
Übung 2.2
*********
Visualisierung multipler eindimensionaler Arrays

Sind die einzelnen Plots nicht in ähnlichen Einheiten oder Größenordnungen, oder wünscht man eine isolierte Darstellung, kann man auch explizit Subplots erzeugen. Im folgenden Beispiel werden 8 solcher Plots erzeugt, und als  4 x 2 Grid ausgelegt. Wir gehen durch alle Grids durch, und plotten das entsprechende Funktionsergebnis.

Versuchen Sie doch, das Grid-Layout in 4 x 2 zu ändern.

Passen Sie die Überschrift für jeden plot dynamisch an, indem Sie die Laufvariable in den Titel einbeziehen
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.001)

idx = 1
"{0}".format(idx)
f"{idx}"


fig, axes = plt.subplots(4, 2)
for idx, ax in enumerate(axes.flatten()):
    print(idx)
    ax.plot(x, x ** idx)
    ax.set_title("Funktion")
# %%
"""
Übung 2.3
*********

Visualisierung von 2-dimensionalen Arrays und Annotation

Man kann mittels der plot-Funktion auch mehrdimensionale Arrays plotten.
Dabei geht matplotlib von selbst durch alle Reihen. 
Oft sind jedoch Reihen und Spalten entgegen der Orientierung, 
in der wir plotten wollen. 
Eine Matrix können wir transponieren, indem wir die .T Methode ausführen.

Sind alle Verläufe eines zwei-dimensionalen Arrays aus der selben Domäne, aber man will diese dennoch separieren, kann man auch einen sogenannte Heatmap verwenden. Dabei werden die Verläufe nicht als Plots übereinander gezeichnet, sondern als Farbverläufe. Dies kann auch sinnvoll bei kategorialen Datensätzen sein, wenn man die verschiedenen Kombination zusammen visualisieren möchte.

Erzeugen Sie auch für die x-Achse Ticks und TickLabels

"""
import numpy as np
import matplotlib.pyplot as plt

xlen = 10
ylen = 9

x = np.reshape(np.arange(0, xlen * ylen), (ylen, xlen))

fig, ax = plt.subplots()
ax.plot(x)

fig, ax = plt.subplots()
ax.plot(x.T)


fig, ax = plt.subplots()
im = ax.imshow(x)
ax.set_xticks(range(xlen))
ax.set_yticks(range(ylen))
ax.set_yticklabels(list("ABCDEFGHI"))
fig.colorbar(im)

# %%
"""
Übung 2.4
*********

Beispielhafte Visualisierung eines Elektrodenlayouts

FÜr EEG-Aufzeichnungen bietet sich natürlich eine topographisch angepasste Heatmap an. 
Dazu benötigt man Information, wo im Raum jede Elektrode sitzt und kann dann 
durch Interpolation einen Oberfläche visualisieren.

Elektrodenposition sind standardisiert, und können z.B. aus mne ausgelesen  und geplottet werden. Passen Sie doch die Auswahl der Elektroden einmal an.

"""
from mne.channels import read_layout

layout = read_layout("EEG1005")
layout.plot()

picks = [layout.names.index(chan) for chan in ["Fpz", "C3"]]

picks = []
for chan in ["Fpz", "C3"]:
    pick.append(layout.names.index(chan))    

layout.plot(picks=picks)
# %%
"""
Übung 2.4
*********

Beispielhafte Visualisierung einer Topographie

Im Folgenden simulieren wir ein Aktivitätsmuster, z.B. Potentiale zu einem bestimmten Zeitpunkt und plotten diese auf dem Skalp.


Dazu habe ich erstmal eine Funktion get_channel_pos geschrieben, welche die x/y Koordinaten bestimmter vorgegebener Kanäle aus dem Layout ausliest und für die Weiterverarbeitung mit plot_topomap anpasst.

Anschließend definieren wir willkürlich eine Anzahl von Kanälen. Die Namen müssen natürlich erlaubt sein. Mit `print(layout.names)` können Sie sich alle erlaubten Kanalnamen anzeigen lassen.


Passen Sie doch die Auswahl einmal nach ihrer Laune an!



"""
from mne.viz import plot_topomap
import numpy as np


def get_channel_pos(channel_labels):
    from mne.channels import read_layout

    layout = read_layout("EEG1005")
    return (
        np.asanyarray([layout.pos[layout.names.index(ch)] for ch in channel_labels])[
            :, 0:2
        ]
        - 0.5
    ) * 2


# -----------------


channel_labels = [
    "Fp1",
    "Fp2",
    "F1",
    "Fz",
    "F2",
    "FC5",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "FC6",
    "TP9",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "TP10",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "P3",
    "Pz",
    "P4",
    "Oz",
]


channel_pos = get_channel_pos(channel_labels)
data = np.arange(0, len(channel_labels), 1)
plot_topomap(data, pos=channel_pos, extrapolate="local", sphere=1)


# %%

"""
Übung 2.5.
**********

Referenzierung

Potentiale werden immer im Bezug zu einer Referenzelektrode gemessen. 
Die klassische EEG-Messung ist dabei unipolar, d.h. in Bezug aller Elektroden auf eine. 
Diese Referenz kann man später digital wechseln, z.B. wie hier im Beispiel auf Oz.

Visualisieren Sie auch das Ergebnis der Rferenzierung mit andere Elektroden. Wie verändert sich die Topographie?

Ein anderes klassisches Verfahren ist die CAR oder Common Average Referenzierung. Dabei wird das Mittel aller Elektroden von allen Elektroden abgezogen. Die Summe aller Potentiale ist daher 0, was nach der Kirchhoffschen Maschenregel erstmal sinnvoll erscheint. Beachten Sie aber, dass wir hier Messen und die Elektroden bestenfalls näherungsweise den Knoten eines Netzwerks entsprechen. 

Überlegen Sie, was passieren würde, wenn eine Elektrode defekt wäre. 
Welchen Einfluss hätte das auf die CAR? Versuchen Sie, das zu simulieren.


"""
import matplotlib.pyplot as plt
from mne.viz import plot_topomap


data = np.arange(0, len(channel_labels), 1)
data[10] = 50

# rereferenzierung gegen z.B. Oz
fig, ax = plt.subplots()
#reref = "Oz"
#ax.set_title(reref)
#ref = data[channel_labels.index(reref)]
#data = data - ref
plot_topomap(data, pos=channel_pos, extrapolate="head", sphere=1, axes=ax)

# common average referenzierung
fig, ax = plt.subplots()
ax.set_title("CAR")
ref = data.mean()
data = data - ref
plot_topomap(data, pos=channel_pos, extrapolate="head", sphere=1, axes=ax)

# %%
"""
Übung 2.6
*********

Beispiel an realen Daten

Zum Abschluss laden wir die realen Daten aus Übung 1.9 und versuchen diese zu plotten.

Plotten Sie doch wie in Übung 2.1 alle Kanale gleichzeitig in eine Graphik (über z.B. den Zeitraum von 0:10000). Verändert sich der Verlauf, wenn man die Daten mittels CAR rereferenziert?  Beachten Sie, dass es sinnvoll sein wird, mit .T zu transponieren, wenn Sie data an plot übergeben!

"""
import matplotlib.pyplot as plt
from mne.viz import plot_topomap

# ------------------------------
channel_labels = [
    "Fp1",
    "Fp2",
    "F1",
    "Fz",
    "F2",
    "FC5",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "FC6",
    "TP9",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "TP10",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "P3",
    "Pz",
    "P4",
    "Oz",
]

raw = load_data()  # aus Übung 1.9
# Wir beschränken unsere Auswahl auf einige wenige Kanäle
raw.pick_channels(channel_labels)
data = raw.get_data()
channel_pos = get_channel_pos(channel_labels)

# wir plotten nur Kanal C3, und zwar nur die ersten 10000 Samples
pick = raw.ch_names.index("C3")
plt.plot(data[pick, 600:660])

# Anschließend plotten wir die Topographie der Potentiale bei Sample 1000
fig, ax = plt.subplots()
plot_topomap(data[0:, 640], pos=channel_pos, extrapolate="head", sphere=1, axes=ax)
# %%
fig, ax = plt.subplots()
im = ax.imshow(data[:, :10000])
fig.colorbar(im)