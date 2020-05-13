#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:37:55 2020

@author: mariuskeute
"""

import mne
from mne.viz import plot_topomap
from pathlib import Path
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from my_topomap import plot_topomap


"""
Wir laden unsere in der letzten Woche vorverarbeiteten Daten.
"""
fld = Path(__file__).parent.parent / "daten"
filtered_data = np.load(fld / 'filtered_data.npy')
filtered_data *= 1e6
channel_std = np.std(filtered_data, axis = 1)
plot_topomap(channel_variance)


















"""
Wir schneiden unsere Rohdaten zunaechst in Segmente von zwei Sekunden Laenge. 
"""

epochs = np.reshape(filtered_data[:,:308000], (2000,68,int(308000/2000)))


