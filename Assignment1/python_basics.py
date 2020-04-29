#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% --> das ist eine Zelle
# Prakische Shortcuts:
# Ausführung von markiertem Code: F9
# Ausführung einer ganzen Zelle: Ctrl+Enter
# Ganzes Skript: F5
# Block Kommentieren: Crtl + 1


#####################
# 1) Variablen
#####################
a = 1  # integer
b = 2.0  # float = Fließkommazahl
c = 'some string'  # string = text

# Wörter kombinieren: mit +
d = 'another string'
sentence = c + d
print(sentence)


# Die print Funktion
print(a + b)
print(c + b) # --> gibt Fehler
print(c + str(b))
print(c, b)


# Indizierung (von Text)
# Indizierung startet bei 0!
d[0]
d[:3]
d[-1]
d[-2:]


#####################
# 2) Zusammengesetzte Datentypen
#####################
## 2.a) Listen

# Listen können kombinierte Datentypen enthalten
liste1 = ['banane', 'apfel', 10, 1.5]

# Leere Liste
leere_liste = []

# Indizierung analog zu strings:
liste1[0]
liste1[:2]
liste1[-1]
liste1[-2:]

# Listenelement verändern
liste1[2] = 'birne'

# Ein Element hinten anhängen
liste1.append(300)

# Länge der Liste
len(liste1)

# Letztes Element entfernen
liste2.pop()
print(liste2)

# Irgendein Element entfernen
del liste1[1]

# Verschachtelte Listen
liste2 = [liste1, 'anderes Element']
# Indizierung
liste2[0][2]



## 2.b) Dictionaries

# Liste, die durch "keys" indizierbar ist
dict1 = {'bananen': 20, 'äpfel': 10, 'hallo': 'hi', 'eine liste': liste1}

# leeres Dictionary
d_leer = {}

# Indizierung mit KEYS
dict1[0]   # --> gibt einen KeyError, weil 0 kein Key ist
dict1['bananen'] # funktioniert

# Elemente in dictionaries sind nicht geordnet!

dict1[0] = 'hallo' # 0 kann auch als integer definiert werden

dict1.keys() # gibt eine Liste mit allen Keys in dict1 zurück

# Deshalb braucht pop() den Key, der Entfernt werden soll als Argument
dict1.pop(0)

# oder:
del dict1['bananen']


# Verschachtelte Dictionaries gibt es auch 
dict2 = {'ein_dictionary': dict1, 'irgendwas anderes': 10}

# Indizierung funktioniert analog zu Listen
dict2['ein_dictionary']['eine liste'][3]


#####################
# 3) Ablaufsteuerung
#####################
# Tab Syntax in Python!

# 3.a) If - Else Statements
# if, elif, else Anweisung
a = 10
b = 8

if a > b:
    print('a größer als b')


if a > b:
    print('a größer als b')
    print('Zeile 2 innerhalb if statement')
print('Außerhalb if statement')
    
    
if a > b:
    print('a größer als b')
else:
    print('a NICHT größer als b')


if a > b:
    print('a größer als b')
elif a < b:
    print('a kleiner als b')
elif a == b:
    print('a gleich b')


#%% and, or
a = 12
b = 10
c = 8

if a > b:
    print('a larger than b')
    
if b > c:
    print('b larger than c')
    
if a > b and b > c:
    print('a and b larger than c')

if a > b or b > c:
    print('a larger than b or b larger than c')
    
    
#%% in, not in
if 'banane' in liste1:
    print('banane ist in Liste 1')
else:
    print('banane ist NICHT in Liste 1')

if 10 not in liste1:
    print('10 ist NICHT in Liste 1')    

if not 10 in liste1:
    print('10 ist NICHT in Liste 1')   

if 'äpfel' in dict1:
    print('äpfel ist in Dict 1')
else:
    print('äpfel ist NICHT in Dict 1')
    
if 10 in dict1:
    print('10 ist in Dict 1')  
else:
    print('10 ist NICHT in Dict 1')
    
if 10 in dict1.values():
    print('10 ist value in Dict 1')  
else:
    print('10 ist NICHT value in Dict 1')
    
if 10 in dict1.keys():
    print('10 ist key in Dict 1')  
else:
    print('10 ist NICHT key in Dict 1')

if ('äpfel', 10) in dict1.items():
    print('(äpfel, 10) ist item in Dict 1')  
else:
    print('(äpfel, 10) ist NICHT item in Dict 1')


# funktioniert nicht mit einfachen Variablen
if a in b:
    print('a ist in b')
    


## 3.b) while Anweisung
a=0
while a < 10:
    print(a)
    a += 1 # addiere 1
    

# kann auch mit Listen verwendet werden, z.B. um Elemente zu entfernen:
liste1 = ['banane', 'apfel', 'birne', 10, 1.5]
while 'birne' in liste1:
    liste1.pop()


## 3.c) for Anweisung
# iteriert i durch alle Elemente der Liste durch, und führt die folgenden 
# Befehle für jede Instanz von i aus:
for i in [0,1,2,3]: 
    print(i)

# Liste kann auch als Variable definiert werden
num_liste = [0,1,2,3]
for i in num_liste:
    print(i)

# die range Funktion
for i in range(1,4):
    print(i)
    
# For Loops können auch verwendet werden, um Variablen zu verändern, 
# z.B. um die Summe aus allen Elementen einer Liste zu berechnen:
summe = 0
for i in range(4):
    print(i)
    summe = summe + i
print('Summe:', summe)


# For Loops funktionieren natürlich für alle Arten von Listen (nicht nur für Zahlen)
for element in liste1:
    print(element) 

# und auch für Dictionaries
for key in dict1:
    print(key)

for key, value in dict1.items():
    print('key:', key)
    print('value:', value)




#####################
# 4) Funktionen
#####################
# Funktionen definieren
# Funktion mit mehreren Kommandos
def function1():
    print('Hallo')
    print('World')

function1()


# Funktion mit Argumenten
def function2(a):
    print('Argument:', a)

function2(2)
function2(30)

# Funktion die Wert zurückgibt
def multiply_2(a):
    return a*2

ergebnis = multiply_2(8)
print(ergebnis)

# Funktion mit mehreren Argumenten
def add_function(a, b):
    return a + b

summe = add_function(5, 2)
print(summe)


# Bespiel von oben:
def check_if_inside(element, liste):
    if element in liste:
        print(element, 'ist in Liste')
    else:
        print(element, 'ist NICHT in Liste')

check_if_inside('äpfel', liste1)
check_if_inside('banane', liste1)
check_if_inside(10, liste1)
check_if_inside(100, liste1)
check_if_inside('äpfel', dict1)


# Funktion für Liste UND Dictionary:
def check_if_inside2(element, liste):
    if type(liste) == list:
        if element in liste:
            print(element, 'ist in Liste')
        else:
            print(element, 'ist NICHT in Liste')
    elif type(liste) == dict:
        if element in liste:
            print(element, 'ist in Dictonary')
        else:
            print(element, 'ist NICHT in Dictonary')
    else:
        print('Ungültiger Variablentyp!')

# Test:
check_if_inside2('äpfel', liste1)
check_if_inside2('banane', liste1)
check_if_inside2(10, liste1)
check_if_inside2(100, liste1)
check_if_inside2('äpfel', dict1)
check_if_inside2('äpfel', 100)


#####################
# 5) Ausnahmen
#####################
dict1[0] # --> gibt einen KeyError und bricht ab

# Ausnahme behandeln ohne Programmabbruch
try:
    print('Wert in ', 0, ':', dict1[0])
except KeyError:
    print(0, 'ist NICHT in Dictionary')


# als Funktion
def test_key(dictionary, key):
    try:
        print('Wert in ', key, ':', dictionary[key])
    except KeyError:
        print(key, 'ist NICHT in Dictionary')

      
# Test:
test_key(dict1, 0)
test_key(dict1, 'äpfel')


#####################
# 6) Numpy arrays
#####################
import numpy as np
# Vektor- oder Matrix-Operationen sind schwierig und unpraktisch mit Listen
# z.B. 1 zu allen Elementen der Liste addieren
liste1 += 1 # --> funktioniert nicht
liste1 = liste1 + 1 # --> funktioniert auch nicht
liste1 = liste1 + [1] # --> funktioniert, macht aber nicht was wir wollen


# Mit Numpy array:
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array3 = array1 + 1

# Arrays habe immer nur EINEN Datentyp
array2 = np.array([[1, 2, 3], [4, 5, 'banana']]) # --> definiert ein Arrays mit Strings


# Numpy beinhaltet viele Praktische Vektor- und Matrix-Operationen
print(np.sqrt(array1)) # Wurzel für jedes Element
print(np.sum(array1)) # Summne aus allen Elementen

# Addieren/Multiplizieren/Dividieren von Elementen in zwei Matritzen
print(array1 + array1)
print(array1 * array1)
print(array1 - array1)

# Matrix Transposition
array3 = array1.T

# Matrix Multiplikation
array1.dot(array3)





