# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:31:06 2022

@author: bylj
"""

import numpy, tables, h5py, os
import matplotlib.pyplot as plt
import numpy as np

os.chdir('C:/Users/bylj/Desktop/Synchrotron_SOLEIL/')

pathfile=('./dataNxs/')

premier_scan=3897 
dernier_scan=4126 

nom0=pathfile+'scan_{}_0001.nxs'.format(premier_scan)
essai0=h5py.File(nom0, 'r')
F130=essai0[u'scan_{}/scan_data/data_26'.format(premier_scan)][0]
temps0=essai0[u'scan_{}/scan_data/sensors_timestamps'.format(premier_scan)][0]

plt.figure()

for i in range(premier_scan,dernier_scan+1):
    
    nom=pathfile+'scan_{}_0001.nxs'.format(i)
    essai=h5py.File(nom, 'r')
    F13=essai[u'scan_{}/scan_data/data_22'.format(i)][0]
    F24=essai[u'scan_{}/scan_data/data_25'.format(i)][0]
    temps=essai[u'scan_{}/scan_data/sensors_timestamps'.format(i)][0]
    
    plt.scatter(temps-temps0, F13, marker='x') #move outside for loop to make it quicker
    plt.scatter(temps-temps0, F24, marker='o')
    # plt.pause(0.01) to make movie!
    
plt.title('loads for #13_150_A')
plt.grid()
plt.show()
