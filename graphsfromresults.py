# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:52:27 2023

@author: bylj
"""
### plotting of scientific graphs - XRD curves ###


## prerequisites ##

# Import required packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import pandas

# import some nice fonts

import matplotlib.font_manager as fm# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
# print(font_names) # lets use Adobe Devanagari


# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'Adobe Devanagari'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.direction'] = 'in' # set ticks on the inside on both y and x axis
plt.rcParams['ytick.direction'] = 'in'
mpl.rcParams["mathtext.default"] = 'regular' # set math text to regular font (see above)
# mpl.rc('axes',edgecolor='black') # set all axes to one color (black)
# Generate 3 colors from colormap inferno
colors = cm.get_cmap('inferno', 4)

## importing of columns ##

excel_data_df = pandas.read_excel('G:/Mechanik/PERSONAL/bylj/04_SOLEIL_2210/03_Samples/#16_50_CuNP/#16_50_CuNP_results.xlsx', sheet_name='results', skiprows = [0,1,2], usecols=[12,26,38,52])
# print(excel_data_df)

first_row_with_all_NaN = excel_data_df[excel_data_df.isnull().all(axis=1) == True].index.tolist()[0]
# gives me the first row number of the row that has all the values to be NaN. 

excel_data_4col = excel_data_df.loc[0:first_row_with_all_NaN-1] # excel data without the NaN values: epsilon, sigma, FWHM, res

# print(excel_data_df.loc[0:first_row_with_all_NaN-1])
# print all rows that have data (NaN values excluded)

# rolling average -> this makes the data look different, so maybe we should exclude or explain outliers
# excel_data_4col.resistivity = excel_data_4col.resistivity.rolling(6).mean()


## creating the plots ##

# create figure and axis objects with subplots() - fist axis (left) is film stress
fig,ax=plt.subplots()
ax.scatter(excel_data_4col['fit xray time 22 all loading'], excel_data_4col["Cu σ [GPa]"], color = colors(0), marker = ".") # use facecolor to exclude the inside
# set x axis label
ax.set_xlabel(r'$\epsilon$' " [-]")
ax.set_xlim(0, 0.1) # limits for x axis
# set y axis label
ax.set_ylabel("Film Stress Al (111) [GPa]", color=colors(0))
ax.set_ylim(-0.45, 0.4) # limits for y axis
ax.xaxis.set_tick_params(which='major', size=5, width=0.5, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=5, width=0.5, direction='in', right='off', color=colors(0))
ax.spines['right'].set_color(colors(0))
ax.tick_params(axis='y', colors= colors(0))

# twin object for two different y-axes on same plot for FWHM
ax2 = ax.twinx()
# make plot with different y axis using second axis object 
ax2.scatter(excel_data_4col['fit xray time 22 all loading'], excel_data_4col["FWHM_chi90°"], color = colors(1), marker = "+")
ax2.set_ylabel("FWHM – Al (111) – Chi 90 [°]", color=colors(1), loc = "center", labelpad = -40)
ax2.set_ylim(0.25, 0.4) # limits for y axis
ax2.yaxis.set_tick_params(which='major', size=5, width=0.5, direction='in', right='off', pad = -27, color = colors(1))
ax2.spines['right'].set_color(colors(1))
ax2.tick_params(axis='y', colors= colors(1))
plt.setp(ax2.get_yticklabels()[0], visible=False)
plt.setp(ax2.get_yticklabels()[-1], visible=False)

# same for third axis for resistivity
ax3 = ax.twinx()
# make plot with different y axis using third axis object
ax3.scatter(excel_data_4col['fit xray time 22 all loading'], excel_data_4col.resistivity, color = colors(2), marker = "1")
ax3.set_ylabel(r'$\rho$'" [" r'$\Omega$' "cm]", color=colors(2), loc = "center", rotation = 270, labelpad = 15)
ax3.set_ylim(0.50E-5, 2.0E-5) # limits for y axis
ax3.yaxis.set_tick_params(which='major', size=5, width=0.5, direction='out', right='off', color = colors(2))
ax3.spines['right'].set_color(colors(2))
ax3.tick_params(axis='y', colors= colors(2))
ax3.ticklabel_format(style='sci', useOffset = False) # doesn't change much

# plot title
plt.title("50 nm Cu with W Nanoparticles", fontsize = 16, pad = 10)

## show and save the plot ##
plt.show() 
plt.savefig('G:/Mechanik/PERSONAL/bylj/08_Figures/#16_50_CuNP', ext='png', dpi=3000, transparent=False, bbox_inches='tight')
plt.savefig('G:/Mechanik/PERSONAL/bylj/08_Figures/#16_50_CuNP_vector.svg', transparent=False, bbox_inches='tight')