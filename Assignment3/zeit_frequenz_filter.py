"""
Assignment 3
************
Sie sollen dieses Skript entsprechend den Aufgaben
vervollständigen. Teilweise handelt es sich nicht um Programmier-, sondern um Freitextaufgaben. In diesem
Fall schreiben Sie Ihre Antwort bitte als Kommentar an die entsprechende Stelle. Laden Sie Ihr vervollständigtes
Skript anschließend per Ilias oder Github hoch.
"""




"""

Wir laden nochmals unsere Rohdaten, mit der Funktion aus dem letzten Assignment.

"""
import mne
from mne.viz import plot_topomap
from pathlib import Path
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal

def load_data():
    fld = Path(__file__).parent.parent / "daten"

    # Import the BrainVision data into an MNE Raw object
    raw = mne.io.read_raw_brainvision(fld / "00_rest_pre.vhdr", preload=True)

    # Read in the event information as MNE annotations
    annot = mne.read_annotations(fld / "00_rest_pre.vmrk")

    # Add the annotations to our raw object so we can use them with the data
    raw.set_annotations(annot)

    # Reconstruct the original events from our Raw object
    events, event_ids = mne.events_from_annotations(raw)
    event_labels = {v: k for k, v in event_ids.items()}
    for event in events:
        print("Sample {0:<10} with {1}".format(event[0], event_labels[event[-1]]))
    data = raw.get_data()
    channel_labels = raw.ch_names
    print(channel_labels)
    print(data.shape)

    return raw

raw = load_data()
data = raw.get_data()
fs = raw.info['sfreq'] #wir extrahieren die Samplingfrequenz - diese werden wir gleich fuer die FFT brauchen.

#%%
"""
Übung 2.1
*********
Wir transformieren zunächst einen Kanal (C3) mittels FFT in den Frequenzraum.
Das Spektrum wird in der Abbildung dargestellt. Sie sehen zwei prominente Peaks -
einen bei 0 Hz und einen bei 50 Hz.

Aufgabe 1: Überlegen Sie - warum ist die Signalstärke bei 0 Hz und 50 Hz am größten?
            Verändern Sie die Achsenbegrenzungen so, dass Sie auch den Bereich zwischen
            0 und 50 Hz gut sehen können. Was fällt hier auf?
Aufgabe 2: Unten berechnen wir die Signalstärke bei 10 und 50 Hz für alle Kanäle.
        Erstellen Sie, wie letzte Woche gelernt, topographische Darstellungen für 
        beide Frequenzen. Vergleichen Sie die Verteilung der Signalstärke auf dem
        Kopf. Was fällt auf?
        Tipp: - Sie werden die Funktionen plot_topomap und get_channel_pos vom
                letzten Mal brauchen. Die erste ist hier bereits importiert, die
                zweite können Sie einfach aus dem Skript von letzter Woche hier
                hinein kopieren (oder - für Fortgeschrittene - aus dem Skript von
                                 letzter Woche importieren)
              - Topographische Darstellungen können durch Ausreißer / Extrem-
                  werte verzerrt sein. Um einen guten Eindruck der Topographie
                  zu bekommen, sollten Sie verschiedene Werte für die obere und
                  untere Grenze der Farbskale ausprobieren. Schauen Sie in die
                  Dokumentation von plot_topomap, dort wird erklärt, wie's geht.

"""
pick = raw.ch_names.index("C3") #wir beginnen, wie letztes Mal, mit Kanal C3.
C3dat = data[pick,:]

fourierspectrum = fft(C3dat)
frx = np.linspace(0, fs/2, num=int(len(fourierspectrum)/2))
plt.figure()
plt.plot(frx, np.abs(fourierspectrum)[:int(len(fourierspectrum)/2)])

#wir führen die FFT nun für jeden Kanal durch und speichern die Signalstärke
#bei 50 Hz und bei 10 Hz in jeweils einer Liste
channel_labels = raw.ch_names[:-4] #die letzten 4 Kanäle sind keine EEG-Kanäle
f10 = np.argmin(np.abs(frx-10))
f50 = np.argmin(np.abs(frx-50))
power_10Hz = []
power_50Hz = []
for chan in range(len(channel_labels)):
    fourierspectrum = fft(data[chan,:])
    powerspectrum = np.abs(fourierspectrum)[:int(len(fourierspectrum)/2)]**2 / len(fourierspectrum)
    power_10Hz.append(powerspectrum[f10])
    power_50Hz.append(powerspectrum[f50])
   

#Ihre Lösung zu Aufgabe 2 hier:


#%%


"""
Übung 2.2
*********
Die FFT basiert auf der Faltung des Signals mit einer komplexen Sinuswelle.
Diese kann durch Eulers Formel e^(i*k) erzeugt werden. In Python werden Imaginär-
zahlen mit 1j (statt i) bezeichnet und die Eulersche Zahl e kann mit der Funktion
exp aus der numpy-Bibliothek exponiert werden. Wenn Sie diesen Abschnitt ausführen, 
werden zwei Figures geöffnet. Die erste stellt eine komplexe Sinuswelle dreidimensional
dar (in den Dimensionen x * y_imaginär * y_real. Wenn Sie sie mit der Maus 
hin- und herbewegen, sehen Sie, dass die Welle geformt ist wie ein Korkenzieher.
Diese komplexe Sinuswelle besteht aus einem realen Kosinus und einem imaginären
Sinus. Da Kosinus und Sinus um pi/2 phasenverschoben sind, entsteht im komplexen
Raum die Korkenzieherform. Diese kann genutzt werden, um nicht nur die Amplitude /
Signalstärke, sondern auch den Phasenversatz der Sinuswellen im Signal zu bestimmen.
In dieser Übung werden wir uns mit allen drei Aspekten des Signals im
Frequenzraum beschäftigen: Frequenz, Amplitude bzw. Signalstärke und Phasenversatz.
Hierfür erzeugen wir drei Sinuswellen und stellen zunächst das Powerspektrum 
(Signalstärke) dar.

    
Aufgabe 1: Demonstrieren Sie, dass die Fouriertransformation informationserhaltend ist,
        indem Sie das Fourierspektrum zurücktransformieren (Funktion ifft, bereits importiert). 
        Prüfen Sie die Gleichwertigkeit des ursprünglichen und des zurücktransformierten Signals 
        im Rahmen der Computertoleranz (Funktion allclose aus numpy)
Aufgabe 2: Variieren Sie die Amplitude und den Phasenversatz (phaseshift) der drei
        Sinuswellen. Welchen Einfluss hat das jeweils auf das Powerspektrum?
"""

x=np.arange(0,6*np.pi,step = .001)
complex_sine = np.exp(1j*x) #Euler'sche Formel
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, np.real(complex_sine), np.imag(complex_sine))


freq_in_Hz = [10, 12, 22]
phaseshift = [0, 0, 0]
amplitude  = [1, 0.7, 0.3]
sfreq = 1000
x = np.linspace(0,2*np.pi,num = fs) #Der Dummy-Vektor mit 1000 Datenpunkten zwischen 0 und 2pi. Wird im nächsten Schritt mit der Frequenz multipliziert.
wave1 = amplitude[0] * np.sin(freq_in_Hz[0] * x + phaseshift[0])
wave2 = amplitude[1] * np.sin(freq_in_Hz[1] * x + phaseshift[1])
wave3 = amplitude[2] * np.sin(freq_in_Hz[2] * x + phaseshift[2])

mix = wave1+wave2+wave3

fourierspectrum = fft(mix)
#Ihre Lösung zu Aufgabe 1 hier



powerspectrum = np.abs(fourierspectrum)[:int(len(fourierspectrum)/2)]**2 / len(fourierspectrum)
frx = np.linspace(0, sfreq/2, num=int(len(fourierspectrum)/2))
plt.figure()
plt.plot(frx, powerspectrum)


"""
Während die Signalstärke aus dem Absolutbetrag der Werte des komplexen Fourier-
spektrums berechnet wird, ergibt sich der Phasenversatz aus dem Winkel relativ
zur realen Achse.
Aufgabe 3: Finden Sie die Indizes, bei denen sich die Maxima (Peaks) des Power-
        spektrums befinden. Sie können dafür zum Beispiel die Funktion scipy.signal.find_peaks
        importieren und benutzen. speichern Sie die Indizes als Variable pks
        
Wenn Sie die Peaks gefunden haben, werden zwei Figures erzeugt, die die 
entsprechenden komplexen Werte aus dem Fourierspektrum auf zwei Weisen darstellt:
die erste in kartesischer Darstellung, mit den realen Werten auf der x-, und den
imaginären Werten auf der y-Achse, die zweite in polarer Darstellung, wobei der
Winkel dem Winkel des komplexen Vektors zur realen Achse, die Entfernung vom 
Mittelpunkt der Länge des Vektors entspricht. Standardmäßig wird der erste Wert blau, 
der zweite  orange und der dritte grün dargestellt.

Aufgabe 4: Verändern Sie die Sinuswellen so, dass wave1 um 90°, wave2 um 120°
        und wave3 um 0° nach links verschoben sind (denken Sie daran, die 
        Gradzahlen in Bogenmaß umzurechnen). Führen Sie die Fouriertransformation
        oben nochmals durch. Welche Werte für den Phasenversatz
        finden Sie für die 3 Wellen in der polaren Darstellung? Warum weichen Sie von den
        gewählten Werten ab?
"""

#Ihre Lösung zu Aufgabe 3 hier:
    

peaks_of_spectrum = fourierspectrum[pks]
fig,ax = plt.subplots()
ax.set_xlim(-5*max(powerspectrum), 5*max(powerspectrum))
ax.set_ylim(-5*max(powerspectrum), 5*max(powerspectrum))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

for pk in peaks_of_spectrum:
    ax.plot([0,np.real(pk)], [0,np.imag(pk)], linewidth = 2)
    
    
plt.figure()
for pk in peaks_of_spectrum:
    plt.polar(np.angle(pk),np.abs(pk), '.')    
    

#%%


"""
Übung 2.3
*********
Wir wenden uns wieder den gemessenen EEG-Daten zu und filtern sie. Dafür verwenden
wir ein Butterworth-IIR-Filter 4. Ordnung (das ist ein Standardfilter für EEG-Daten).
Dazu erzeugen wir zunächst den Filterkernel mit der Funktion signal.butter. Wir erstellen
zunächst ein Bandpass-Filter von 0.3 bis 35 Hz und dann ein 50 Hz-Bandstop-Filter, um das Stromleitungsartefakt zu entfernen.
Anschließend müssen wir die generierten Filterkernel noch auf die Daten anwenden. Wir benutzen dafür
zunächst die Funktion signal.lfilter, mit der wir die Daten von links nach rechts filtern (sog. kausales Filtern).

Aufgabe 1: Greifen Sie, wie oben, Kanal C3 heraus. Stellen Sie gefilterte und ungefilterte Daten
        im Vergleich bildlich dar. Plotten Sie gefilterte und ungefilterte Daten übereinander, jeweils
        im Zeit- und Frequenzraum.

Aufgabe 2: Filtern Sie die Daten nochmals, diesmal mit der Funktion signal.filtfilt anstelle von signal.lfilter (Erklärung unten). Damit wir nächste Woche mit den
        gefilterten Daten weiterarbeiten können, speichern Sie sie als binäre Datei (Funktion save aus numpy).
"""
bandpass = signal.butter(4,(.3,35),btype = 'pass', fs = fs)
bandstop = signal.butter(4,(48,52),btype = 'stop', fs = fs)


filtered_data = signal.lfilter(*bandpass, data)
filtered_data = signal.lfilter(*bandstop, filtered_data)

#Ihre Lösung zu Aufgabe 1 hier




"""
Eine unvorteilhafte Eigenschaft des Butterworth-Filters ist, dass er die Phase der
gefilterten Daten ändert. Um dies zu verhindern, werden Daten meist zweimal gefiltert,
einmal von links nach rechts (kausal), einmal in Gegenrichtung. Das ist in der Funktion
signal.filtfilt implementiert. Da dieser Phasenversatz bei gemessenen Signalen oft
etwas schwer zu sehen ist, veranschaulichen wir ihn uns mit einem Sinussignal.

Aufgabe 2: Erzeugen Sie ein 1s langes Sinussignal der Frequenz 20 Hz mit
        Samplingfrequenz 1000 Hz. Sie können die Sinussignale, die oben erzeugt
        werden, als Vorlage benutzen. Filtern Sie dieses Signal mit dem oben erzeugten
        Bandpass-Filter. Zeigen Sie graphisch, dass ein Phasenversatz entsteht, wenn 
        Sie den Filter mit der Funktion signal.lfilter anwenden, und dass dieser Phasen-
        versatz wieder korrigiert wird, wenn Sie stattdessen die Funktion signal.filtfilt
        verwenden.
    
"""




