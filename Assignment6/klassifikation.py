"""
Assignment 6: Klassifikation

Im Folgenden finden Sie Codebeispiele, in denen wir in den Beispiel-EEG Daten
aus den letzten Praktika vorhersagen (klassifizieren) wollen, ob die Augen zu einem bestimmten
Zeitpunkt offen oder geschlossen waren.
Anhand dieses Beispiels sollen die Grundprinzipien des 
maschinellen Lernens verdeutlicht werden.

"""
import matplotlib.pyplot as plt


#%%
'''
Schritt 0: Vorbereitung der Daten

Zunächst werden die Daten geladen, gefiltert, und in Epochen von 3 Sektunden
Länge unterteilt. Diese Schritte haben Sie so oder so ähnlich schon in den letzten
Wochen kennen gelernt.
Für jede Epoche berechnen wir die Energie in vier verschiedenen 
Frequenzbändern (Theta, Alpha, Beta und Gamma) in allen 64 Elektroden.
Jeder Datenpunkt (Zeit-fenster) hat also 4*64 = 256 Feature.

Ziel dieser Übung wird sein, anhand dieser Daten einen Klassifikator zu trainieren
und im Hinblick auf Separierbarkeit, Overfitting, und Generalisierbarkeit zu
evaluieren.

Führen Sie die folgende Zelle aus, um den Datensatz zu erzeugen.

'''
from scipy.signal import welch
from klassifikation_helper_functions import load_data
import numpy as np
from scipy.integrate import simps
from scipy import signal


 # Länge der Segmente in Sekunden
len_epoch_s = 3

# Definition der vier Frequenzbänder, die als Feature verwendet werden
iter_freqs = [
    ('Theta', 4, 7),
    ('Alpha', 8, 12),
    ('Beta', 13, 25),
    ('Gamma', 30, 45)
]

# Lade EEG Daten
raw = load_data()
channel_labels = raw.info['ch_names'][0:64]
raw.pick_channels(channel_labels)
data = raw.get_data()
fs = raw.info['sfreq']  # Samplingfrequenz
num_channels = raw.info['nchan']
timestamps = np.linspace(0, np.shape(data)[1]/fs, np.shape(data)[1])

# Filtern der Daten wie in Übung 3
bandpass = signal.butter(4,(.3,35),btype = 'pass', fs = fs)
bandstop = signal.butter(4,(48,52),btype = 'stop', fs = fs)
filtered_data = signal.lfilter(*bandpass, data)
filtered_data = signal.lfilter(*bandstop, filtered_data)

# Lade den Vektor mit den Informationen über den Zustand der Augen (1 = geöffnet, 0 = geschlossen)
conditions = np.load('Augen_Zustand.npy')

# Aufteilen der Daten in Zeitfenster mit jeweils 3 Sekunden Länge
len_epoch_samp = int(len_epoch_s*fs)    # Entsprechende Anzahl Samples pro Zeitfenster
num_epochs = int(np.floor(len(timestamps)/len_epoch_samp))   # Gesamt-Anzahl der Zeitfenster

# Wir verwenden Welch's Method, um die Energie in den vier Frequenzbändern zu berechnen
# Das Band-Power Dictionary wird für jedes Zeitfenster die relative Energie in jeder Elektrode
# für jedes der vier Frequenzbänder enthalten
band_power = {}
for freq, low, high in iter_freqs:    # Schleife über alle vier Frequenzbänder
    band_power[freq] = np.empty((num_channels, num_epochs))
    for i_channel in range(num_channels):   # Schleife über alle Elektroden
        this_epoch = 0
        for i_epoch in range(num_epochs):    # Schleife über alle Zeitfenser
            # berechne PSD mit Welch's Methode
            freqs, psd = welch(filtered_data[i_channel][this_epoch:this_epoch+len_epoch_samp], fs, nperseg=1000)
            
            # Finde indizes des Frequenzband
            idx_freq = np.logical_and(freqs >= low, freqs <= high)
            freq_res = freqs[1] - freqs[0]
            
            # Gesamt-Energie (Integral über das gesamte PSD Spektrum)
            total_power = simps(psd, dx=freq_res)
            
            # Relative Energie im Frequenzband, das uns interessiert
            band_power[freq][i_channel, i_epoch] = simps(psd[idx_freq], dx=freq_res) / total_power
        
            this_epoch = this_epoch + len_epoch_samp

# Vektor mit allen Anfangszeitpunkten der Zeitfenster
epoch_timestamps = np.linspace(0, num_epochs*len_epoch_s, num_epochs)

# Außerdem extrahieren wir für jedes Zeitfenster den Zustand der Augen
# Der Einfachheit halber wird jedem Zeitfenster der Zustand in der Mitte des Fensters zugeordnet
this_epoch = 0
cond_epoch = np.empty(num_epochs)
for i_epoch in range(num_epochs):
    cond_epoch[i_epoch] = conditions[min(this_epoch + len_epoch_samp // 2, len(conditions)-1)]
    this_epoch = this_epoch + len_epoch_samp

# Nicht mehr benötigte Variablen läschen
del data, raw, bandpass, bandstop

# Anzeige der erzeugten Daten
fig, ax = plt.subplots(5,1)
ax[0].plot(timestamps[range(len(conditions))], conditions)
ax[0].set_xlim((0,305))
ax[0].set_label('Augen auf / zu')
for i_ax in range(1, len(ax)):
    for i_channel in range(num_channels):
        ax[i_ax].plot(epoch_timestamps, band_power[iter_freqs[i_ax-1][0]][i_channel], color = 'tab:blue', alpha = .4)
    ax[i_ax].set_ylabel(iter_freqs[i_ax-1][0])
    ax[i_ax].set_xlim((0,305))

#%%
'''
Schritt 1: Aufteilung in Trainings- und Test-Daten

Zunächst teilen wir unseren kompletten Datensatz in Trainings- und Test-Daten auf.
Die Trainingsdaten werden wir benutzen, um unseren Klassifikator zu trainieren,
mit den Testdaten werden wir evaluieren, wie gut er funktioniert (Validierung).

Wichtig: der Klassifikator darf nur die Trainingsdaten sehen, aber
in keinem Moment Informationen über die Testdaten bekommen!!!
Andernfalls würden wir es ihm 'zu leicht machen' und verzerrte Ergebnisse erhalten.
z.B. bei KOmponentenzerlegung besteht das Risiko, dass man die Komponenten aus
allen Daten bestimmt, und dann erst die Zeitreihen in Trainings- und Testdaten
unterteilt. Das kann zu fälschlich positiven Ergebnissen führen!!

Die Test-Daten werden wir zunächst komplett beiseite lassen und erst wieder 
zur Klassifikator-Validierung anschauen

'''
import random
random.seed(0)

# Wir wählen zufällig 20% der Zeitfenster als Test-Datensatz aus
num_testing = int(np.floor(num_epochs*.2))
num_training = num_epochs - num_testing
select_id = np.ones(num_epochs)
select_id[random.sample(range(num_epochs), num_testing)] = 0

# Bestimmung der entsprechenden Trainigs- und Test-Zeitfenster
test_id = np.where(select_id == 0)[0]
train_id = np.where(select_id == 1)[0]

# Aufgrund der IDs erzeugen wir zwei Dictionaries mit identischer Struktur wie
# band_power, die jeweils nur die Trainings- oder test-Daten beinhalten
train_data, test_data = {}, {}
for freq_band in band_power:
    train_data[freq_band] = band_power[freq_band][:, train_id]
    test_data[freq_band] = band_power[freq_band][:, test_id]

# Auch die Klassenzuordnung in 'Augen auf' und 'Augen zu' wird entsprechend unterteilt
train_labels = cond_epoch[train_id]
test_labels = cond_epoch[test_id]

#%%
'''
Schritt 2.1: Visualisierung der Features

Der folgende Code plottet die Energie in den vier Frequenzbändern für alle 
Zeitfenster im Trainingsdatensatz und alle EEG Kanäle

Vergleichen Sie die Features im Hinblick auf Separierbarkeit.
Welche der Features eignen sich Ihrer Meinuung nach am besten zur Klassifikation? Warum?

'''

channels_to_select = range(num_channels)

def plot_freq_data(data, class_label, channels_to_select):
    fig, ax = plt.subplots(1,2)
    for channel in channels_to_select:
        ax[0].scatter(data['Theta'][channel][class_label == 1].tolist(),
                   data['Alpha'][channel][class_label == 1].tolist(),
                   color='blue', alpha = .3)
        ax[0].scatter(data['Theta'][channel][class_label == 0].tolist(),
                   data['Alpha'][channel][class_label == 0].tolist(),
                   color='red', alpha = .3)
    
    for channel in channels_to_select:
        ax[1].scatter(data['Beta'][channel][class_label == 1].tolist(),
                   data['Gamma'][channel][class_label == 1].tolist(),
                   color='blue', alpha = .3)
        ax[1].scatter(data['Beta'][channel][class_label == 0].tolist(),
                   data['Gamma'][channel][class_label == 0].tolist(),
                   color='red', alpha = .3)
    
    ax[0].set_xlabel('Theta')
    ax[0].set_ylabel('Alpha')
    ax[0].legend(('Augen offen', 'Augen zu'))
    
    ax[1].set_xlabel('Beta')
    ax[1].set_ylabel('Gamma')
    ax[1].legend(('Augen offen', 'Augen zu'))

plot_freq_data(train_data, train_labels, channels_to_select)


#%%
'''
Schritt 2.2: Feature Selektion aufgrund von Vorwissen

Das Öffnen und Schließen der Augen beeinfluss das Signal selbstverständlich
nicht in allen Hirnregionen gleichermaßen.

In wechen Elektroden erwarten Sie aufgrund Ihres Vorwissens zur Neuroanatomie
aus Vorlesung 1 eine Veränderung durch das Öffnen und Schließen der Augen?

Verwenden Sie die folgenen Codezeilen, um die Frequenz-Band Energie für nur
ausgewählte Kanäle anzeigen zu lassen. 

Welche Elektroden würden Sie zur Klassifikation auswählen?

'''
channels_to_select_name = ['Fz', 'Cz']

# diese Funktion gibt die Positionen der ausgewählten Elektroden zurück
def get_channel_id(select, names):
    return [pos for pos, name in enumerate(names) if name in select]

channels_to_select = get_channel_id(channels_to_select_name, channel_labels)

plot_freq_data(train_data, train_labels, channels_to_select)


#%%
'''
Schritt 3.1: Klassifikator-Training

Nun können wir den Trainingsdatensatz verwenden, um einen Klassifikator zu
trainieren.

Zunächst verwenden wir einen linearen Klassifikator basierend auf der Fisher'schen
Diskriminanzfunktion (LinearDiscriminantAnalysis aus sklearn)

Der folgende Code bestimmt ein lineares Modell, um die beiden Klassen 
(1 = Augen offen/ / 0 = Augen zu) optimal zu trennen.

Hierzu werden zunächst Theta- und Alpha-Band-Energie (zur vereinfachten 
zweidimensionalen Darstellung) aller Elektroden verwendet.

Füllen Sie die fehlenden Zeilen zur Berechnung der Performance Scores aus.

Schauen Sie sich dann die Klassifikator Visualisierung, Confusion-Matrix und 
Scores an.
Was sagen diese über die Qualität des Modells aus?

'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import plot_confusion_matrix
from klassifikation_helper_functions import plot_model_fit

# Zunächst verwenden wir alle Elektroden
channels_to_select = range(num_channels)

# vor der Klassifikation müssen wir noch die Features vektorisieren,
# d.h., in eine eindimensionale Form bringen
def vectorize_data(data, labels, channels_to_select):
    data_vect = np.array([np.hstack(data['Theta'][channels_to_select]), 
                  np.hstack(data['Alpha'][channels_to_select]), 
                  np.hstack(data['Beta'][channels_to_select]), 
                  np.hstack(data['Gamma'][channels_to_select])]).T
    label_vect = np.tile(labels, len(channels_to_select))
    return data_vect, label_vect

X_train, y_train = vectorize_data(train_data, train_labels, channels_to_select)

# Modell-Schätzung
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
model = lda.fit(X_train[:,:2], y_train)

# Vorhersage der Klassen-Zuordnung aufgrund der Trainings-Daten und dem Modell
y_pred = model.predict(X_train[:,:2])

# Visualisierung der richtig und falsch klassifizierten Epochen und der
# linearen Trennlinie (Funktion abgeleitet von https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html#sphx-glr-auto-examples-classification-plot-lda-qda-py)
plot_model_fit(model, X_train[:,:2], y_train, y_pred)
plt.title('LDA Trainings-Daten')

# Confusion Matrix anzeigen
conf_mat = plot_confusion_matrix(model, X_train[:,:2], y_train, cmap=plt.cm.Blues) #, normalize = 'all')
plt.title('Performance Trainings-Daten ')

# Scores berechnen
def print_scores(y_true, y_pred):
    TP = sum(np.logical_and(y_true == 1, y_pred == 1)) # True Positives
    FP =  # False Positives
    TN =  # True Negatives
    FN =  # False Negatives
    sens = # Sensitivität
    spec = # Spezifizität
    prec = # Precision
    acc = # Accuracy
    print('Sensitivität:', sens)
    print('Spezifizität:',spec)
    print('Precision:', prec)
    print('Accuracy:', acc)    

print('\nTraining:')
print_scores(y_train, y_pred)

#%%
'''
Schritt 4.1: Validierung an Test-Daten

Um auf die Generalisierbarkeit des Klassifikators zu schließen, muss das Modell
an den Testdaten validiert werden, die nicht zur Bestimmung herangezogen wurden

Vergleichen Sie Training- und Test-Performance. Es bestehen im Wesentlichen zwei
Gefahren: 
    1.Overfitting (Trainingsdaten wurden zu gut oder zu detailliert gefittet, so dass die Verallgemeinerung
                       auf die Testdaten nicht gelingt). Das sollte sich in einem guten Fit
                        für die Trainings- und einem schlechten Fit für die Testdaten zeigen.
    2.Underfitting (Trainingsdaten wurden zu grob / mit den falschen Features gefittet). Das sollte sich
                        in einem schlechten Fit sowohl für Trainings- als auch für Testdaten zeigen.
Was bedeuted das Ergebnis für unseren Klassifikator im Bezug auf Overfitting oder Underfitting?

'''
X_test, y_test = vectorize_data(test_data, test_labels, channels_to_select)

# Vorhersage der Klassen-Zuordnung aufgrund der Trainings-Daten und dem Modell
y_pred = model.predict(X_test[:,:2])

# Visualisierung der richtig und falsch klassifizierten Epochen und der
# linearen Trennlinie (Funktion abgeleitet von https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html#sphx-glr-auto-examples-classification-plot-lda-qda-py)
plot_model_fit(model, X_test[:,:2], y_test, y_pred)
plt.title('LDA Test-Daten')

# Confusion Matrix anzeigen
conf_mat = plot_confusion_matrix(model, X_test[:,:2], y_test, cmap=plt.cm.Blues)
plt.title('Performance Test-Daten ')

# Scores berechnen
print('\nTest:')
print_scores(y_test, y_pred)

#%%
'''
Schritt 3.2: Klassifikator Training mit selektierten Features

In Schritt 2.2 habe Sie schon Features aufgrund Ihres Neuroanatomischen
Vorwissens ausgewählt.

Verwenden Sie den folgenden Code, um einen linearen Klassifikator nur für die
von Ihnen ausgewählten Elektroden 

Vergleichen Sie die Performance mit der Performance des Klassifikators ohne
Feature Selektion. Wie bewerten Sie die beiden Fälle im Bezug auf Overfitting
und Generalisierbarkeit?
'''
########################################
# Auswahl der Elektroden
########################################
channels_to_select_name = ['Cz','Fz']
channels_to_select = get_channel_id(channels_to_select_name, channel_labels)
X_train, y_train = vectorize_data(train_data, train_labels, channels_to_select)
X_test, y_test = vectorize_data(test_data, test_labels, channels_to_select)

########################################
# Modell-Schätzung
########################################
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
model = lda.fit(X_train[:,:2], y_train)

########################################
# Evaluation an Trainings-Daten
########################################
# Vorhersage der Klassen-Zuordnung aufgrund der Trainings-Daten und dem Modell
y_pred = model.predict(X_train[:,:2])

# Visualisierung der richtig und falsch klassifizierten Epochen und der
# linearen Trennlinie
plot_model_fit(model, X_train[:,:2], y_train, y_pred)

# Confusion Matrix anzeigen
conf_mat = plot_confusion_matrix(model, X_train[:,:2], y_train, cmap=plt.cm.Blues)
plt.title('Performance Trainings-Daten ')

# Scores berechnen
print('\nTraining:')
print_scores(y_train, y_pred)


########################################
# Evaluation an Test-Daten
########################################
# Vorhersage der Klassen-Zuordnung aufgrund der Trainings-Daten und dem Modell
y_pred = model.predict(X_test[:,:2])

# Visualisierung der richtig und falsch klassifizierten Epochen und der
# linearen Trennlinie (Funktion abgeleitet von https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html#sphx-glr-auto-examples-classification-plot-lda-qda-py)
plot_model_fit(model, X_test[:,:2], y_test, y_pred)
plt.title('LDA Test-Daten')

# Confusion Matrix anzeigen
conf_mat = plot_confusion_matrix(model, X_test[:,:2], y_test, cmap=plt.cm.Blues)
plt.title('Performance Test-Daten ')

# Scores berechnen
print('\nTest:')
print_scores(y_test, y_pred)



#%%
'''
K-Nächste-Nachbarn Klassifikation

Alternativ können wir den K-Nächste-Nachbarn Algorithmus zur Klassifikation verwenden.
Dieser basiert nicht auf einer Diskriminanzfunktion, sondern sucht für einen Testdatenpunkt X
bei N Features im N-dimensionalen Raum die K nächsten Nachbarn und ordnet X die Eigenschaft
(also Augen auf vs. Augen zu) zu, die die meisten Nachbarn haben. K kann dabei frei gewählt werden.
Der KNN-Algorithmus ist konzeptionell einfacher als die Diskriminanzanalyse, da er keine expliziten Annahmen
über Verteilung oder lineare Separierbarkeit der Daten trifft.

Vergleichen Sie die Ergebnisse für verschiedene Nachbar-Anzahl miteinander
und mit den Ergebnissen des linearen Klassifikators.

Wie ist das Verhalten dieser Klassifikatoren im Bezug auf Overfitting, Generalisierbarkeit und Linearität?
'''
from sklearn.neighbors import KNeighborsClassifier
from klassifikation_helper_functions import plot_knn

# Parameter für Nächste-Nachbarn-Algorithums
n_neighbors = 20

########################################
# Auswahl der Elektroden
########################################
channels_to_select_name = ['Cz','Fz']
channels_to_select = get_channel_id(channels_to_select_name, channel_labels)
X_train, y_train = vectorize_data(train_data, train_labels, channels_to_select)
X_test, y_test = vectorize_data(test_data, test_labels, channels_to_select)

########################################
# Modell-Bestimmung
########################################
knn = KNeighborsClassifier(n_neighbors)
model = knn.fit(X_train[:,:2], y_train)

########################################
# Evaluation an Trainings-Daten
########################################
y_pred = knn.predict(X_train[:,:2])

plot_knn(knn, X_train[:,:2], y_train)
plt.title('K-Nearest-Neighbor Klassifikator Trainings-Daten')

# Confusion Matrix anzeigen
conf_mat = plot_confusion_matrix(model, X_train[:,:2], y_train, cmap=plt.cm.Blues)
plt.title('Performance Trainings-Daten ')

# Scores berechnen
print('\nTraining:')
print_scores(y_train, y_pred)

########################################
# Evaluation an Test-Daten
########################################
# Vorhersage der Klassen-Zuordnung aufgrund der Trainings-Daten und dem Modell
y_pred = model.predict(X_test[:,:2])

plot_knn(knn, X_test[:,:2], y_test)
plt.title('K-Nearest-Neighbor Klassifikator Test-Daten')

# Confusion Matrix anzeigen
conf_mat = plot_confusion_matrix(model, X_test[:,:2], y_test, cmap=plt.cm.Blues)
plt.title('Performance Test-Daten ')

# Scores berechnen
print('\nTest:')
print_scores(y_test, y_pred)



#%%
'''
Naive Bayes Klassifikator

Zuletzt noch der Vergleich zum Naive Bayes Klassifikator. Das besondere an diesem Klassifikator
ist, dass eine 'prior probability'spezifiziert werden kann, d.h. es können Annahmen getroffen werden, 
welche Klasse (Augen auf vs. Augen zu) a priori wie wahrscheinlich ist, während die anderen Klassifikatoren
implizit angenommen haben, dass alle Klassen a priori gleich wahrscheinlich sind, was aber nicht bei jedem
Klassifikationsproblem der Fall ist.

Vergleichen Sie die Ergebnisse des Naive Bayes Klassifikator mit denen der anderen
Klassifikatoren von zuvor.

Wie ist das Verhalten dieser Klassifikatoren im Bezug auf Overfitting, Generalisierbarkeit und Linearität?

'''
from sklearn.naive_bayes import GaussianNB

########################################
# Auswahl der Elektroden
########################################
channels_to_select_name = ['Fz', 'Cz']
channels_to_select = get_channel_id(channels_to_select_name, channel_labels)
X_train, y_train = vectorize_data(train_data, train_labels, channels_to_select)
X_test, y_test = vectorize_data(test_data, test_labels, channels_to_select)

########################################
# Modell-Bestimmung
########################################
gnb = GaussianNB()
model = gnb.fit(X_train[:,:2], y_train)

########################################
# Evaluation an Trainings-Daten
########################################
y_pred = model.predict(X_train[:,:2])

# Confusion Matrix anzeigen
conf_mat = plot_confusion_matrix(model, X_train[:,:2], y_train, cmap=plt.cm.Blues)
plt.title('Performance Trainings-Daten ')

# Scores berechnen
print('\nTraining:')
print_scores(y_train, y_pred)

########################################
# Evaluation an Test-Daten
########################################
# Vorhersage der Klassen-Zuordnung aufgrund der Trainings-Daten und dem Modell
y_pred = model.predict(X_test[:,:2])

# Confusion Matrix anzeigen
conf_mat = plot_confusion_matrix(model, X_test[:,:2], y_test, cmap=plt.cm.Blues)
plt.title('Performance Test-Daten ')

# Scores berechnen
print('\nTest:')
print_scores(y_test, y_pred)



#%%
'''
Als Abschluss noch ein Kommentar zum Thema Kreuz-Validierung.

Wie Sie bemerkt haben, kann die Auswahl des Test-Samples einen Einfluss auf die
Performance der Klassifikator-Eigenschaften haben.

Um ein stabileres Ergebniss zu bekommen, wir deshalb häufig Kreuz-Validierung
(Cross-Validation) verwendet, indem für verschiedene Aufteilungen in Trainings-
und Test-Daten die Klassifikator-Performance bestimmt wird.
Die finale Performance wird dann durch den Durchschnitt und die Standardabweichung
der jeweiligen Scores angegeben.
Hierdurch kann auch bestimmt werden, wie stabil die Performance für Variationen
im Trainings- und Test-Datensatz ist.

Im Folgenden ein Beispiel für die Verwendung von Funktionen aus sklearn zur
Realisierung einer solchen Kreuz-Validierungs-Schleife
'''
from sklearn.model_selection import train_test_split

########################################
# Auswahl der Elektroden
########################################
channels_to_select_name = ['Fz', 'Cz']
channels_to_select = get_channel_id(channels_to_select_name, channel_labels)

# Dieses mal vektorisieren wir ALLE Daten (anfangs in band_power und cond_epoch gespeichert)
X, y = vectorize_data(band_power, cond_epoch, channels_to_select)
X = X[:, :2]

# Aufteilung in Trainings- und Test-Daten mithilfe von train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


########################################
# Kreuz-Validierung mit Funktionen aus sklearn
########################################
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

# Erzeugung von 50 zufälligen Aufteilungen in 20% Test und 80% Trainings-Daten
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

# Bestimmung der Scores für das LDA Modell
scores = cross_val_score(lda, X, y, cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
