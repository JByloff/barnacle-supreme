# -*- coding: utf-8 -*-
"""
Created on Fri Jul 03 23:03:42 2020

@author: diffabs
"""
import numpy as np

from sympy.solvers import nsolve

import math

import os

import scipy.optimize as op
import matplotlib.pyplot as plt

from numpy import genfromtxt

def f(x):
    """Fonction dont on cherche une racine."""
    return np.exp(-np.pi*RABt2/x)+np.exp(-np.pi*RBCt2/x)-1.

os.chdir('C:/Users/bylj/Desktop/Synchrotron_SOLEIL/Resistivity//')
nom='#12_150_c_a.txt'
#nom='keithley_2m_ar_200611c_point.txt'
#nom='keithley_2m_ar_200612b_partie1.txt'
#nom='keithley_2m_ar_200610b.txt'

#valeurs=numpy.loadtxt(nom, skiprows=1)
valeurs=np.loadtxt(nom, skiprows=12)

#valeurs2=np.genfromtxt(nom, skiprows=15, filling_values='')

nombre_valeurs=np.shape(valeurs)[0]
resultats=np.array([])




debut=0
fin=2606 #big, this is the number of values that he will read, change after
# the iteration has failed for the first time 

nombre_valeurs=fin
# this code takes the resistance values AB, BC, from the Keithley and 
# solves the VanderPauw equation with them. Why does it multiply with -3?
# RABt and RABt2 are just different averages, only RABt2 is used
# uses the averaged Pauw formula (see wikipedia)

for i in range(debut,nombre_valeurs):
    RAB=abs(valeurs[i,15]*1**-3)
    RABt=(-valeurs[i,15]*1**-3+valeurs[i,18]*1**-3-valeurs[i,21]*1**-3+valeurs[i,24]*1**-3)/4
    RABt2=(abs(valeurs[i,15]*1**-3)+abs(valeurs[i,21]*1**-3))/2 #averages AB and CD
    RBC=abs(valeurs[i,16]*1**-3)
    RBCt=(-valeurs[i,16]*1**-3+valeurs[i,19]*1**-3-valeurs[i,22]*1**-3+valeurs[i,25]*1**-3)/4
    RBCt2=(abs(valeurs[i,16]*1**-3)+abs(valeurs[i,22]*1**-3))/2 #averages BC and DA
    solution= op.fsolve(f,0.0)
    resultats=np.hstack((resultats, solution))
'''
plt.figure(2)
plt.plot(valeurs[debut:nombre_valeurs,0]-valeurs[debut,0],resultats, '-ko', label='Rsheet')
plt.plot(valeurs[debut:nombre_valeurs,0]-valeurs[debut,0],valeurs[debut:nombre_valeurs,15], 'r+', linestyle='-', label='RAB')
plt.plot(valeurs[debut:nombre_valeurs,0]-valeurs[debut,0],valeurs[debut:nombre_valeurs,16], 'g-', linestyle='-', label='RBC')
plt.legend()
plt.grid()
plt.title('210716b')
plt.show()
'''
fig, ax1 = plt.subplots()

color = 'red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Resistivity [x10^-3] (Ohm.cm)', color=color)
ax1.plot(valeurs[debut:nombre_valeurs,0]-valeurs[debut,0],resultats*70e-7*1000, color=color, label='resistivity')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid()
#ax1.set_ylim(0.02, 0.08)
#ax1.set_xlim(0, 7800)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



#color = 'blue'
ax2.set_ylabel('Rmeasured (Ohm)', color='black')  # we already handled the x-label with ax1
#ax2.plot(valeurs[debut:nombre_valeurs,0]-valeurs[debut,0],valeurs[debut:nombre_valeurs,15], color='steelblue', marker='.', markersize=8, linewidth=1, label='RBC')
#ax2.plot(valeurs[debut:nombre_valeurs,0]-valeurs[debut,0],valeurs[debut:nombre_valeurs,16], color='green', marker='.', markersize=8, linewidth=1, label='RAB')
ax2.tick_params(axis='y', labelcolor='black')

ax2.set_ylim(-5000, 20)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

#plt.title('210716b')
plt.legend()
plt.show()
#plt.savefig('200611c_graph', dpi= 500)
np.savetxt('#12_150_c_a_res.txt', resultats)

############
#
#
#Traitements numériques... Fourier
#
#L'analyse spectrale d'un signal numérique, ou analyse fréquentielle, se fait par la transformée de Fourier discrète (TFD)
#
#Nous utilisons pour cela la fonction numpy.fft.fft, qui calcule la TFD 
#avec l'algorithme de transformée de Fourier rapide. 
#Pour construire l'échelle de fréquence, il faut attribuer la fréquence d'échantillonnage
# à la plus grande fréquence du spectre. La résolution fréquentielle du spectre est 
#donc égale à l'inverse de la durée totale du signal, soit 1/(NeTe), 
#où Ne est le nombre d'échantillons.
#
##############


'''
import numpy.fft
#Ne = len(u0)
#Te = t0[1]-t0[0]

#fin=10000

Ne=len(valeurs[debut:fin, 0])
Te=valeurs[fin, 0]-valeurs[debut, 0]

uAB=valeurs[debut:fin,18]

uBC=valeurs[debut:fin,19]

spectreAB = np.absolute(numpy.fft.fft(uAB))/Ne
frequencesAB = np.arange(Ne,)*1.0/(Te*Ne)

spectreBC = np.absolute(numpy.fft.fft(uBC))/Ne
frequencesBC = np.arange(Ne,)*1.0/(Te*Ne)

plt.figure()
plt.plot(frequencesAB[1:],spectreAB[1:], label='RAB',marker='o')
plt.figure()
plt.plot(frequencesBC[1:],spectreBC[1:], label='RBC', marker='d')
plt.legend()
plt.xlabel("f (Hz)")
plt.ylabel("RAB ou BC")#si colonne 18
#plt.ylabel("RBC")#si colonne 19
plt.ylim(0,0.005)
plt.grid()
'''