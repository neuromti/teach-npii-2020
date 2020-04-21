# Assingment 1: Installation und erste Schritte

## 1. Installieren Sie Anaconda auf Ihrem Computer
- [Windows](https://www.datacamp.com/community/tutorials/installing-anaconda-windows)
- [Mac](https://www.datacamp.com/community/tutorials/installing-anaconda-mac-os-x)
- [Ubuntu](https://wiki.ubuntuusers.de/Anaconda/)

## 2. Installieren Sie git und erstellen Sie einen GitHub Account (falls Sie noch keinen haben)
- Anleitung für git Installation auf [allen Betriebssystemen](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- Erstellen Sie einen [github](https://github.com/) account.

## 3. Download der Kursmaterialien und Erstellung Ihres eigenen "Branchs"
- Öffnen Sie ein Terminal (in Mac oder Linux) oder die Windows Eingabeaufforderung und nutzen Sie `cd /pfad/zum/Ordner` um in den Ordner zu navigieren, in den Sie die Kursunterlagen herunterladen wollen. Dann downloaden Sie die Kursmaterialien mit
```
git clone https://github.com/translationalneurosurgery/teach-npii-2020.git
```
- Navigieren Sie into den neuen Orgner mit `cd teach-npii-2020`
- Erzeugen Sie Ihren eigenen [Branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) and navigieren Sie git in diesen Branch mit den folgenden Befehlen:
```
git branch <your_branch_name>
git checkout <your_branch_name>
```
- Jetzt können Sie die Dateien im heruntergeladenen Ordner nach Belieben bearbeiten. Im Folgenden einige nützliche Befehle zur Versionsverwaltung mit git:
  - Git über neu erstellte Dateien informieren:
    ```
    git add <name_of_new_file> 
    ```
  - Veränderungen in git einbinden:
    ```
    git commit -a -m 'A commit message' 
    ```
  - Lokale Änderungen online hochladen (z.B., um Ihre Ergebnisse zu den Assignments abzugeben):
    ```
    git push -u origin <your_branch_name>
    ```
  - Online Updates herunterladen (z.B., um wöchentlich neue Assignments herunterzuladen):
    ```
    git pull
    ```

## 4. Öffnen Sie Spyder mit einer der folgenden Optionen
- [Anaconda-Navigator](https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-nav-win) öffnen und auf Spyder Button klicken.
- Anaconda Prompt (in Windows) oder Terminal (in Mac/Ubuntu) öffnen und den Befehl `spyder` eingeben.

## 5. Ausführen von "hello_world.py"
Öffnen Sie die Datei "hello_world.py" in Spyder.

Führen Sie die Datei aus, indem Sie auf den "Run file"-Button ![Run](run.png) klicken.

Sie sollten folgende Ausgabe in der Spyder Konsole sehen:
```
Hello, world!
```

## 6. Bitte machen Sie sich mit Datentypen und Kontrollflow in Python vertraut
- Nützliche Ressourcen für eine Einführung in die Frundlagen von Python:
  - https://www.codecademy.com/learn/learn-python-3
  - https://learn.datacamp.com/courses/intro-to-python-for-data-science
  - http://greenteapress.com/thinkpython/thinkpython.pdf
