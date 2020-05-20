# %%
"""
Übung 1.1
*********
Einfache Abschätzung des Powerspektrums mittels Periodogram

Das Periodogram ist die die *empirische* Fouriertransformierte. Es schätzt also die  spektrale Leistungsdichte eines Signals, bzw. die Energie des Signals für jede der gemessenen Frequenzen. Sie entspricht somit der Fouriertransformation der Autokorrelation des Signals oder dem Quadrat der Amplitude der Signalkomponenten.

Wir wollen uns das im folgenden genauer anschauen. Führen Sie dazu die Zelle aus. Wir generieren zuerst ein sinusoidales Signal einer bestimmten Frequenz und Amplitude, fügen weisses Rauschen hinzu und berechnen dann das Periodogramm des Signals. Anschließend lassen wir uns die Energie der simulierten Frequenz ausgeben und versuchen, dessen Amplitude zu schätzen.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram


fs = 1000
duration = 1
frequency = 20
amplitude = 1

t = np.linspace(0, duration, int(duration * fs))
x = amplitude * np.sin(2 * np.pi * t * frequency)
x += np.random.randn(len(t))

fix, ax = plt.subplots(2, 1)
ax[0].plot(t, x)
f, Pxx = periodogram(x, fs)
ax[1].semilogy(f, Pxx)
ax[1].set_xlabel("frequency [Hz]")
ax[1].set_ylabel("PSD [2/Hz]")
plt.show()
fidx = np.argmin(np.abs(f - frequency))
print(
    "Energie beträgt:",
    Pxx[fidx],
    "die Amplitude betrug vermutlich",
    2 * np.sqrt(Pxx[fidx] / duration / 2),
)
# %%
"""
Übung 1.2
*********

Im Verlauf der nächsten Übungen werden wir immer wieder ein (Mess-)Signal generieren müssen. Wie sie oben sehen, hängt dieses Signal von mehreren Parametern ab. Daher würde es Sinn machen, eine entsprechende Funktion zu definieren und zu vderwenden.

Überlegen Sie, welche Parameter die folgende Funktion benötigt und tragen Sie das fehlende Argument in die Parameterliste der Funktion an Stelle des Fragezeichens ein. Beachten Sie, dass beim Aufruf einer Funktion die Reihenfolge relevant ist!

In der Zeihle mit Assert testen wir, ob ein 10 Hz Signal erfolgreich generiert wurde.

"""


def generate_signal(amplitude, frequency, duration, fs):
    t = np.linspace(0, duration, int(duration * fs))
    x = amplitude * np.sin(2 * np.pi * t * frequency)
    x += np.random.randn(len(t))
    return x


assert len(generate_signal(1, 10, 1, 1000)) == 1000

# %%
"""
Übung 2.1
*********

Nachdem wir die Funktion erzeugt haben, wollen wir das Verhalten des Periodograms etwas genauer erkunden. 

Erinnern Sie sich, dass die Auflösung der Fouriertransformation von der Länge des Signals abhängt. Ist das Signal 1 Sekunde lang, kann man 1 Hz auflösen, bei 2s 1/2 Hz und bei 0.5s auf 2 Hz genau. Wie ist die Auflösung bei einer Dauer von 3 Sekunden?
"""

fs = 1000
duration = 1
frequency = 10
amplitude = 1

x = generate_signal(amplitude, frequency, duration, fs)
f, Pxx = periodogram(x, fs)
print(
    f"Die Frequenzauflösung bei einer Signaldauer von {duration} Sekunde ist {np.diff(f).mean()} Hz"
)

# %%
"""
Übung 2.2
*********

Was passiert nun, wenn die Frequenz sich nicht exakt auflösen lässt?

"""

fs = 1000
duration = 0.5
frequency = 15
amplitude = 1

x = generate_signal(amplitude, frequency, duration, fs)
f, Pxx = periodogram(x, fs)
plt.semilogy(f, Pxx)
fidx = np.argmin(np.abs(f - frequency))
print(
    "Energie beträgt:",
    Pxx[fidx],
    "die Amplitude betrug vermutlich",
    2 * np.sqrt(Pxx[fidx] / duration / 2),
)
# %%
"""
Übung 2.3
*********

Dasselbe würde natúrlich auch passieren, wenn wir ein zu kurzes Signal in das Periodogram geben (also ein Signal von nur 500ms  Dauer mit x[:len(x)//2])

Bonusfragen: 
Was macht '//' im Gegensatz zu '/' ? 
Warum sollte man bei der Indizierung '//' verwenden?
"""
fs = 1000
duration = 1
frequency = 15
amplitude = 1

x = generate_signal(amplitude, frequency, duration, fs)
f, Pxx = periodogram(x[: len(x) // 2], fs)
plt.semilogy(f, Pxx)
fidx = np.argmin(np.abs(f - frequency))
print(
    "Energie beträgt:",
    Pxx[fidx],
    "die Amplitude betrug vermutlich",
    2 * np.sqrt(Pxx[fidx] / duration / 2),
)
# %%
"""
Übung 2.4
*********

Diese Halbierung des Signals hat zwar den Nachteil der redizierten Frequenzauflösung, aber auch einen Vorteil: Wir können hergehen, das Signal in zwei Teile trennen und den Mittelwert der Ergebnisse der beiden Periodogramme verwenden. Dadurch verringert sich der Beitrag des weissen Rauschens im Spektrum. 

Beachten Sie, dass wir die Frequenz wieder auf 10 Hz gesetzt haben, damit die Frequenzauflösung wieder in Ordnung wäre.

Bonusfrage:

Warum wird die Amplitude im folgenden Beispiel jetzt trotzdem noch unterschätzt?
Inwiefern hat  dies mit der Dauer des Signals zu tun?
Wie könnte man dafür korrigieren?

"""
fs = 1000
duration = 1
frequency = 10
amplitude = 1

x = generate_signal(amplitude, frequency, duration, fs)
f, Pxx0 = periodogram(x[: len(x) // 2], fs)
f, Pxx1 = periodogram(x[len(x) // 2 :], fs)
Pxx = (Pxx0 + Pxx1) / 2
plt.semilogy(f, Pxx)
fidx = np.argmin(np.abs(f - frequency))
print(
    "Energie beträgt:",
    Pxx[fidx],
    "die Amplitude betrug vermutlich",
    2 * np.sqrt(Pxx[fidx] / duration / 2),
)

# %%
"""
Übung 3.1
*********

Und damit betreten wir den Bereich von "Welch's Methode": Die Mittelung der Ergebnisse von Periodogramen von (überlappenden) Fenstern des Signals.

Welch's Methode tauscht dabei Frequenzauflösung gegen Rauschreduktion.

Die oben beschrieben Methode, d.h. die Mittelung nicht-überlappende Rechteckfenster finden Sie unten mit scipy's welch-Funktion implementiert. 

Was passiert, wenn man eine Anzahl an Segmenten wählt, die zu einer Segmentdauer führt, bei der die Frequenzauflösung dann wieder nicht passt?
Probieren Sie das Verhalten bei verschiedenen "num_segments" und "duration" aus.

"""
from scipy.signal import welch

fs = 1000
duration = 1
frequency = 10
amplitude = 1
num_segments = 2

x = generate_signal(amplitude, frequency, duration, fs)


segment_duration = duration / num_segments
nperseg = fs * segment_duration
f, Pxx = welch(x, fs, window="boxcar", nperseg=nperseg, noverlap=0)
plt.semilogy(f, Pxx)

# %%
"""
Übung 3.2
*********

Üblich ist Welch"s Methode mit sogenannten Hanning-Fenstern und einem 50%igen Überlappen der Segmente. Das ist auch der Default in scipy, daher können wir die Argumente einfach weglassen, um die Berechnung entsprechend anzupassen. 

Erkunden Sie auch hier das Verhalten bei verschiedenen "num_segments" und "duration". Erweitern Sie dazu zuerst die Zelle um den dafür notwendigen Code.

"""

f, Pxx = welch(x, fs, nperseg=nperseg)
plt.semilogy(f, Pxx)

# %%
"""
Übung 3.3
*********

Wie sieht so ein Hanning-Fenster aus? Lesen Sie die Dokumentation von 
scipy.signal.welch online z.B. unter https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html

Dort wird auf get_window verwiesen. Lassen Sie uns die beiden bisher bekannten zwei Fenster mal genauer anschauen.

Warum entspricht das Boxcar-Fenster (auf Deutsch oft Rechteckfenster genannt) dem Segmentieren durch Indizierung eines zusammenhängenden Blocks? 

Welchen Vorteil könnte das Hanning-Window diesem Rechteckfenster gegenüber besitzen?

Lesen Sie auch die Dokumentation von get_window und visualisieren sie ein paar andere Fenster. 

Probieren Sie einmal, das Leistungsspektrum von Fenstern ihrer Wahl zu bestimmen.

"""

from scipy.signal import get_window

nperseg = 100
fig, ax = plt.subplots(1, 1)
ax.plot(get_window("hanning", nperseg), label="Hanning")
ax.plot(get_window("boxcar", nperseg), label="Boxcar")
ax.legend()
