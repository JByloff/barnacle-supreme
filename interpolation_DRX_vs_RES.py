# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

temps_DRX=np.loadtxt('G:/Mechanik/PERSONAL/bylj/04_SOLEIL_2210/03_Samples/#17_50_Cu/DRX_temps.txt', skiprows=1)
temps_RES=np.loadtxt('G:/Mechanik/PERSONAL/bylj/04_SOLEIL_2210/03_Samples/#17_50_Cu/epsilon.txt', skiprows=1)

###interval DRX environ 140 sec
###interval DIC environ ... sec

## Il y a donc moins de points DRX, je m'en sers donc de référence

 

temps_DRX_interpole_tableau=np.zeros(2)

nombre_ligne=np.shape(temps_DRX)[0] #size of array

for i in range(1,nombre_ligne,1):
    essai=np.where(temps_RES[:,0]<temps_DRX[i,0]) # only condition, counts up -> looks until when the DRX time value is bigger than the RES value
    index_fin=essai[0][-1]#derniere ligne dans temps_DIC pour laquelle la valeur temps_DIC[0] est plus petite que temps_DRX
    #interpolation lineaire y=a*x+b
    a_eps1=(temps_RES[index_fin+1,1]-temps_RES[index_fin,1])/(temps_RES[index_fin+1,0]-temps_RES[index_fin,0]) # delta y / delta x
    b_eps1=temps_RES[index_fin,1]-a_eps1*temps_RES[index_fin,0]
    
    #a_eps2=(temps_DIC[index_fin+1,2]-temps_DIC[index_fin,2])/(temps_DIC[index_fin+1,0]-temps_DIC[index_fin,0])
    #b_eps2=temps_DIC[index_fin,2]-a_eps2*temps_DIC[index_fin,0]
    
    DIC_epsequiv_interpole=a_eps1*temps_DRX[i,0]+b_eps1#celui qui a le plus de points
   # RBC_interpole=a_eps2*temps_CIN[i,0]+b_eps2
     
    temps_DRX_interpole_tableau=np.vstack((temps_DRX_interpole_tableau, [temps_DRX[i,0],DIC_epsequiv_interpole]))#,RBC_interpole]))
    
np.savetxt('#17_50_Cu_interpol_eps.txt',temps_DRX_interpole_tableau)

"""
Vérification en tracant les points interpolés par rapport aux données brutes
"""

plt.figure()
plt.plot(temps_DRX_interpole_tableau[:,0],temps_DRX_interpole_tableau[:,1], 'ko', label='DIC_epsequiv_interpol')
#plt.plot(temps_DRX_interpole_tableau[:,0],temps_DRX_interpole_tableau[:,2], 'v', label='RBC_interpol')#
plt.plot(temps_RES[:,0],temps_RES[:,1], '-s', label='eps_brute')
#plt.plot(temps_DIC[:,0],temps_DIC[:,2], '-d', label='RBC_brute')
plt.legend()
plt.grid()
plt.show()
