# %%
"""
Übung 1.1
*********

Führe die Zelle aus, um eine Datei im Schreib-Modus ("w") zu öffnen, und einen inhalt hineinzuschreiben

Schau die Datei in einem Editor an (z.B. Spyder, gedit oder notepad)

Was passiert mit der Datei, wenn man die variable `inhalt` verändert und die Zelle erneut ausführt? 

\n ist laut Industriestandard  das Zeichen für einen Zeilenumbruch (ausser in Windows, das möchte gerne \r\n)
\ wird laut Industristandard (z.b. ASCII) als sogenanntes "Escape" Zeichen 
verwendet, um klarzumachen, dass das folgende Zeichen eine besondere 
Bedeutung hat. Das ist einer der Gründe, warum es problematisch ist, dass Microsoft Verzeichnisstrukturen mit `\` darstellt (statt mit `/` wie Linux und Mac)

"""
fname = "test.txt"
inhalt = "Hello World\n"
with open(file=fname, mode="w") as f:
    f.write(inhalt)

# %%
"""
Übung 1.2
*********

Man kann Dateien auch mit Python lesen. Führe die folgende Zelle aus, um den Inhalt der Datei auszugeben.

Beachte, dass die Datei `test.txt`auf der Festplatte, die Variable `inhalt` aber im Arbeitsspeicher lebt. Man kann beide verändern, ohne die jeweilige andere zu beeinflussen.

"""
fname = "test.txt"
with open(file=fname, mode="r") as f:
    inhalt = f.read()
print(inhalt)

# %%
"""
Übung 1.3
*********

Der Zeiger in der Datei kann mit f.seek gesetzt werden, z.B. 6 bytes nach dem Anfang der Datei mit f.seek(6,0). Was passiert, wenn man die Datei nicht von Anfang an liest?

Die aktuelle Position des Zeigers kann man mit print(f.tell()) erfragen.

Wo befindet sich der Zeiger, wenn die Datei vollständig mit f.read gelesen wurde? 

"""
fname = "test.txt"
with open(file=fname, mode="r") as f:
    f.seek(6, 0)
    inhalt = f.read()
print(inhalt)


# %%
"""
Übung 1.4
*********

Diese Zelle offnet eine Datei im Schreib und Lese -Modus ("r+") und überschreibt Text.

Erzeuge die Datei neu, z.B. mit Übung 1.1. Überschreibe dann aber nicht die ersten 5 Zeichen, sondern die letzen 5 mit 'X'

"""
fname = "test.txt"
with open(file=fname, mode="r+") as f:
    for i in range(5):
        f.write("X")
    f.seek(0, 0)
    print(f.read())


# %%
"""
Übung 1.5
*********

Was passiert, wenn man statt einem String, Zahlen in eine Textdatei schreiben will? 

Eine Lösung wäre es, Zahlen in einen str konvertieren. Passen Sie die Zelle dementsprechend an.

"""
fname = "test.txt"
inhalt = 1001
with open(file=fname, mode="w") as f:
    f.write(inhalt)


# %%
"""
Übung 1.6
*********

Zwar lassen sich vielen Datentypen mittels `str` konvertieren, jedoch nicht mit Sicherheit und Garantie aus einem String rekonstruieren. Zur sicheren und umkehrbaren Konvertierung von Datentypen in Strings gibt es viele Standards und Möglichkeiten. Diese sind teilweise sogar bereits in der Python-Standardbibliothek beinhaltet, z.b. im json-Modul.

Testen Sie json mit anderen Datentypen, z.b. float oder dict

"""
import json

original = ["String", 1, 1.2345, {"key": "value"}]
encoded = json.dumps(original)
decoded = json.loads(encoded)
print("Rekonstruktion", decoded, ":", original, "erfolgreich?", decoded == original)

# %%
"""


Ein derartiges Vorgehen ist aber sehr verschwenderisch und wird daher für Zahlentypen und elektrophysiologische Daten meist nicht verwendet. Warum?

Ein Buchstabe benötigt i.d.R. ein Byte Speicherbedarf pro Zeichen (z.B. nach dem ASCII oder UTF-8-Standard (https://de.wikipedia.org/wiki/UTF-8).

Verwendet man 8-Bit-Integer, kann man Zahlen im Bereich von -128 bis 127 in 1 Byte speichern. Die Zahl -128 als String benötigt aber 4 Zeichen und damit 4 Byte. Das ist viermal soviel Speicherplatz! Für Gleitkommazahlen ist dieser Effekt sogar noch drastischer. Daher verwendet man für reine Zahlendaten häufig ein binäres Speicherformat.

Wie man Buchstaben speichert, ist international ziemlich stark standardisiert und führt daher in der Regel nur zu geringen Problemen (z.B. bei einigen seltenen Sonderzeichen). Das ist auch der Grund, warum Textdateien so simpel und transportabel wirken. In "Wahrheit" sind alle Dateien binäre Dateien, nur der Standard für Text ist sehr einheitlich. Es ist auch der Grund, warum wir Quellcode mit einem Texteditor (anstatt mit Word oder LibreOffice) schreiben. 

Für die meisten binären Datenformate ist das aber leider anders, und je nach Anwendungsfall gibt es sehr viele verschiedene Standards. Die genaue Spezifikation einer binären Datei ist also vor allem bedingt durch die Anforderungen an die Datei.  Welche Datentypen sollen gespeichert werden? Sollen auch Kommentare und Metadaten gespeichert werden? Kann man die Aufnahme unterbrechen und später weiterführen? Zudem verwenden die Hersteller von Biosignalverstärkern gerne eigene Datenformate, um Anwender in ihrem System zu halten. Es gibt mehrere offene Standards für elektrophysiologische Daten und einige der wichtigsten (https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/03-electroencephalography.html) sind vermutlich:

- European data format `.edf`
- BrainVision Core Data Format (.vhdr, .vmrk, .eeg)
- EEGLAB (.set and .fdt files)
- Biosemi data format (.bdf)

Diese Formate können aber nur unimodale Daten, z.B. EEG von einem Verstärkersystem sammeln. Verwendet man multimodale Daten ist ein bedeutender Standard: 

- Extensible data format `.xdf` (https://github.com/sccn/xdf/wiki/Specifications) 

Zum Glück muss man nur selten binäre Dateien direkt lesen. Dafür gibt es häufig spezialisierte Speicher- und Lesefunktionen die in Python von Erweiterungen zur Verfügung gestellt werden. Matlabs `.mat` Dateien kann man z.B. mit scipy.io.loadmat laden (https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html#scipy-io-loadmat) und für Numpy gibt es eigene save und load Routinen.


Übung 1.7
*********

Im Folgenden speichern wir einen Array mittels numpy binär. Danach öffnen wir die Datei explizit im binären Modus mit Lesezugriff durch das Argument "rb".

Wir sehen dann eine Mischung aus Bytes, die als ASCII-Zeichen interpretierbar sind und aus nicht als Zeichen interpretierbaren Bytes. Erkennbar sind, dass numpy auch Metainformation abspeichert. 

Was passiert, wenn wir die Datei als Textdatei öffnen, d.h. mit "r" statt "rb"?

"""
import numpy as np

arr = np.array([[0, 1, 2], [4, 5, 6]])
np.save("test.npy", arr)

with open("test.npy", "rb") as f:
    print(f.read())

back = np.load("test.npy")
print(arr, back)
# %%
"""
Übung 1.8
*********

Natürlich kann man auch binär direkt schreiben und eine Datei manipulieren, indem wir eins der Bytes überschreiben. Hier gibt z.B. das \ for dem x an, dass die folgenden Zeichen als Hexzahl interpretiert werden sollen. 

Erkunden Sie was passiert, wenn sie verschiedenen Stellen mittels f.seek wählen und und überschreiben.
"""

import numpy as np

arr = np.array([[0, 1, 2], [4, 5, 6]])
np.save("test.npy", arr)

with open("test.npy", "r+b") as f:
    f.seek(128, 0)
    f.write(b"\x0f")

back = np.load("test.npy")
print(arr)
print(back)
# %%
"""
Übung 1.9
*********

Nach Installation von MNE laden wir einen EEG Datensatz im BrainVision Core Data Format (.vhdr, .vmrk, .eeg)

Installieren Sie zuerst die `mne` Toolbox in ihrem Python. Unter Anaconda geht das durch Öffnen des Anaconda-Prompts. https://datatofish.com/how-to-install-python-package-in-anaconda/

Je nachdem, wie sie die Pfaderkennung organisiert haben, kann auch ein normales Terminalfenster ausreichen. Dort führen Sie dann `pip install mne` aus.

Als nächstes laden wir die Daten von 

Wenn Sie das nächste Mal Spyder starten, sollte `import mne` funktionieren.


"""
import mne
