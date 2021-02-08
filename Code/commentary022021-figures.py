import pandas as pd
import os
import re
import numpy as np
from datetime import datetime

# Plotting Packages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"

colors=['#033C5A','#AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F',
        '#033C5A','#AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F']

#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------Import Data--------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Import monthly data
monthlyIndex=pd.read_csv(r'C:\Users\Zoey\Box Sync\GWRSC\Regulatory Sentiment and Uncertainty\Data\TDM Studio\Analysis of Reg News\Data\RegRelevant_MonthlySentimentIndex_Jan2021.csv')
print(monthlyIndex.info())

monthlyIndex['Year-Month']=monthlyIndex['Year'].map(str)+'-'+monthlyIndex['Month'].map(str)
monthlyIndex['date']=monthlyIndex['Year-Month'].astype('datetime64[ns]').dt.date

for dict in ['GI','LM','LSD']:
    monthlyIndex[dict+'index_standardized']=(monthlyIndex[dict+'index']-np.mean(monthlyIndex[dict+'index']))/np.std(monthlyIndex[dict+'index'])
monthlyIndex['UncertaintyIndex_standardized']=(monthlyIndex['UncertaintyIndex']-np.mean(monthlyIndex['UncertaintyIndex']))/np.std(monthlyIndex['UncertaintyIndex'])

# Import weekly data
weeklyIndex=pd.read_csv(r'C:\Users\Zoey\Box Sync\GWRSC\Regulatory Sentiment and Uncertainty\Data\TDM Studio\Analysis of Reg News\Data\RegRelevant_WeeklySentimentIndex_Jan2021.csv')
print(weeklyIndex.info())

weeklyIndex['date']=weeklyIndex['StartDate'].astype('datetime64[ns]').dt.date

for dict in ['GI','LM','LSD']:
    weeklyIndex[dict+'index_standardized']=(weeklyIndex[dict+'index']-np.mean(weeklyIndex[dict+'index']))/np.std(weeklyIndex[dict+'index'])
weeklyIndex['UncertaintyIndex_standardized']=(weeklyIndex['UncertaintyIndex']-np.mean(weeklyIndex['UncertaintyIndex']))/np.std(weeklyIndex['UncertaintyIndex'])

# PCA of monthly sentiment indexes
from sklearn.decomposition import PCA

#features = ['GIindex_standardized', 'LMindex_standardized', 'LSDindex_standardized']
features = ['GIindex', 'LMindex', 'LSDindex']
x = monthlyIndex.loc[:, features].values
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalComponents_neg=principalComponents*(-1)
principalDf = pd.DataFrame(data = principalComponents_neg, columns = ['SentimentPC1', 'SentimentPC2'])
print("Variance explained by PC:", pca.explained_variance_ratio_)

monthlyIndex = pd.concat([monthlyIndex, principalDf], axis = 1)

monthlyIndex['SentimentMax']=monthlyIndex[['GIindex','LMindex','LSDindex']].max(axis=1)
monthlyIndex['SentimentMin']=monthlyIndex[['GIindex','LMindex','LSDindex']].min(axis=1)

# PCA of weekly sentiment indexes
x = weeklyIndex.loc[:, features].values
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalComponents_neg=principalComponents*(-1)
principalDf = pd.DataFrame(data = principalComponents_neg, columns = ['SentimentPC1', 'SentimentPC2'])
print("Variance explained by PC:", pca.explained_variance_ratio_)

weeklyIndex = pd.concat([weeklyIndex, principalDf], axis = 1)


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------Plot Monthly Sentiment & Uncertainty Indexes--------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Plot monthly uncertainty index under Trump with weekly inset
x=monthlyIndex['date'][-49:]
y=monthlyIndex['UncertaintyIndex'][-49:]

fig, ax = plt.subplots(1, figsize=(15,8))
ax.plot(x,y,color=colors[0],marker='D',markersize=8)

# Events
ax.text(datetime(2016,12,1), 0.73, 'Transition\nof power', fontsize=13, color=colors[4],horizontalalignment='center')
ax.text(datetime(2020,4,1), 0.8, 'Coronavirus\noutbreak', fontsize=13, color=colors[4],horizontalalignment='center')
ax.text(datetime(2020,11,1), 0.77, '2020 presidential\nelection', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')
#
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(years_fmt)
# ax.xaxis.set_minor_locator(months)
#
# # round to nearest years.
# datemin = np.datetime64(min(x), 'Y')
# datemax = np.datetime64(max(x), 'Y') + np.timedelta64(1, 'Y')
# ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Monthly Uncertainty Index',fontsize=16)
ax.set_yticks(np.arange(round(min(y),1)-0.1,round(max(y),1)+0.2,0.1))
#ax.set_ylim(bottom=round(min(y),1))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

# Title
fig.suptitle('Figure 1: Uncertainty about Regulatory Policy',
                x=0.72, y=0.95,fontsize=20)
ax.set_title('(January 2017 - January 2021)',fontsize=18,position=(0.85,1.1))

# Inset plot
xins=weeklyIndex['date'][-52:]
yins=weeklyIndex['UncertaintyIndex'][-52:]

axins=inset_axes(ax, width=5, height=2.5, bbox_to_anchor=(.05, .69, .5, .5),
                    bbox_transform=ax.transAxes,loc=2)

axins.plot(xins,yins,color='#033C5A',linewidth=2,marker='D',markersize=5)
axins.format_xdata = mdates.DateFormatter('%Y-%m')
axins.set_yticks(np.arange(round(min(yins),1)-0.1, round(max(yins),1)+0.2, 0.1))
axins.grid(color='gray', which='major', axis='y', linestyle='dotted')
axins.tick_params(axis='both',which='major',labelsize=10)
axins.set_facecolor('#d3d3d3')
axins.set_alpha(0.2)
axins.set_title('Weekly Index over the Past 12 Months',fontsize=14,position=(0.5,0.85))

# Adjust plot position
plt.subplots_adjust(top=0.81, bottom=0.15)

#Notes
fig.text(0.12, 0.02,'Notes: The uncertainty index was estimated using a dictionary-based sentiment analysis'
                    ' approach applied to newspaper text and fixed effects\nregressions. '
                    'For details on the methodology, refer to the latest draft of the Sinclair and Xie paper'
                    ' on "Sentiment and Uncertainty about Regulation".',
         fontsize=14,style='italic')

plt.savefig('Figures/UncertaintyIndex under Trump.jpg', bbox_inches='tight')
plt.savefig('Figures/Figure1.jpg', bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Plot monthly uncertainty index with events by presidential year
x=monthlyIndex['date']
y=monthlyIndex['UncertaintyIndex']

fig, ax = plt.subplots(1, figsize=(15,9))
ax.plot(x,y,color='black')

# Presidential year
ax.axvspan(datetime(1985,1,1),datetime(1989,2,1),alpha=0.1, color=colors[7])
ax.text(datetime(1987,1,1), 0.91, 'Ronald\nReagan', fontsize=13, color=colors[7],horizontalalignment='center')

ax.axvspan(datetime(1989,2,1),datetime(1993,2,1),alpha=0.1, color=colors[7])
ax.text(datetime(1991,1,1), 0.91, 'George H. W.\nBush', fontsize=13, color=colors[7],horizontalalignment='center')

ax.axvspan(datetime(1993,2,1),datetime(2001,2,1),alpha=0.1, color=colors[0])
ax.text(datetime(1997,1,1), 0.91, 'Bill\nClinton', fontsize=13, color=colors[0],horizontalalignment='center')

ax.axvspan(datetime(2001,2,1),datetime(2009,2,1),alpha=0.1, color=colors[7])
ax.text(datetime(2005,1,1), 0.91, 'George W.\nBush', fontsize=13, color=colors[7],horizontalalignment='center')

ax.axvspan(datetime(2009,2,1),datetime(2017,2,1),alpha=0.1, color=colors[0])
ax.text(datetime(2013,1,1), 0.91, 'Barack\nObama', fontsize=13, color=colors[0],horizontalalignment='center')

ax.axvspan(datetime(2017,2,1),datetime(2021,2,1),alpha=0.1, color=colors[7])
ax.text(datetime(2019,1,1),0.91, 'Donald\nTrump', fontsize=13, color=colors[7],horizontalalignment='center')

# events
ax.text(datetime(2008,9,1), 0.8, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2010,3,1), 0.855, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2010,10,1), 0.87, 'Deepwater Horizon\noil spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2010,7,1), 0.84, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='left')

ax.text(datetime(2016,11,1),0.83 , '2016 presidential\nelection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2020,1,1), 0.79, 'Coronavirus\noutbreak', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(monthlyIndex['date'].iloc[0], 'Y')
datemax = np.datetime64(monthlyIndex['date'].iloc[-1], 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Monthly Uncertainty Index',fontsize=16)
ax.set_yticks(np.arange(round(min(y),1),round(max(y),1)+0.1,0.1))
ax.set_ylim(bottom=round(min(y),1))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

# Title
fig.suptitle('Figure 3: Uncertainty about Regulation by Presidential Year',
                y=0.95,fontsize=20)
ax.set_title('(January 1985 - January 2021)',fontsize=18,position=(0.5,1.12))

#Notes
fig.text(0.12, 0.03,'Notes: The uncertainty index was estimated using a dictionary-based sentiment analysis'
                    ' approach applied to newspaper text and fixed effects\nregressions. '
                    'For details on the methodology, refer to the latest draft of the Sinclair and Xie paper'
                    ' on "Sentiment and Uncertainty about Regulation".',
         fontsize=14,style='italic')

# Adjust plot position
plt.subplots_adjust(top=0.81, bottom=0.15)

plt.savefig('Figures/UncertaintyIndex with Events by Presidential Year.jpg', bbox_inches='tight')
plt.savefig('Figures/Figure3.jpg', bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Plot PC1 under Trump with weekly inset
x = monthlyIndex['date'][-49:]
y = monthlyIndex['SentimentPC1'][-49:]

fig, ax = plt.subplots(1, figsize=(15, 8))
ax.plot(x,y,color=colors[0],marker='D',markersize=8)

# Events
#ax.text(datetime(2016,12,1), 0.73, 'Transition\nof Power', fontsize=13, color=colors[4],horizontalalignment='center')
ax.text(datetime(2018,12,1), -0.45, 'Trump midterm\nelection', fontsize=13, color=colors[4],horizontalalignment='center')
#ax.text(datetime(2020,3,1), -0.15, 'Coronavirus\noutbreak', fontsize=13, color=colors[4],horizontalalignment='center')
#ax.text(datetime(2020,12,1), 0.77, '2020 Presidential Election', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')
#
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(years_fmt)
# ax.xaxis.set_minor_locator(months)
#
# # round to nearest years.
# datemin = np.datetime64(min(x), 'Y')
# datemax = np.datetime64(max(x), 'Y') + np.timedelta64(1, 'Y')
# ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Monthly Sentiment Index',fontsize=16)
ax.set_yticks(np.arange(-0.8,1.4,0.4))
#ax.set_ylim(bottom=round(min(y),1))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

# Title
fig.suptitle('Figure 2: Sentiment about Regulatory Policy',
                x=0.26, y=0.95,fontsize=20)
ax.set_title('(January 2017 - January 2021)',fontsize=18,position=(0.1,1.13))

# Inset plot
xins=weeklyIndex['date'][-52:]
yins=weeklyIndex['SentimentPC1'][-52:]

axins=inset_axes(ax, width=5, height=2.5, bbox_to_anchor=(.52, .75, .5, .5),
                    bbox_transform=ax.transAxes,loc=2)

axins.plot(xins,yins,color='#033C5A',linewidth=2,marker='D',markersize=5)
axins.format_xdata = mdates.DateFormatter('%Y-%m')
axins.set_yticks(np.arange(-2, 3, 1))
axins.grid(color='gray', which='major', axis='y', linestyle='dotted')
axins.tick_params(axis='both',which='major',labelsize=10)
axins.set_facecolor('#d3d3d3')
axins.set_alpha(0.1)
axins.set_title('Weekly Index over the Past 12 Months',fontsize=14,position=(0.5,0.85))

# Adjust plot position
plt.subplots_adjust(top=0.79, bottom=0.15)

#Notes
fig.text(0.12, 0.02,'Notes: The sentiment index was estimated using a dictionary-based sentiment analysis'
                    ' approach applied to newspaper text and fixed effects\nregressions. '
                    'For details on the methodology, refer to the latest draft of the Sinclair and Xie paper'
                    ' on "Sentiment and Uncertainty about Regulation".',
         fontsize=14,style='italic')

plt.savefig("Figures/SentimentPC1 under Trump with Weekly Inset.jpg", bbox_inches='tight')
plt.savefig("Figures/Figure2.jpg", bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Plot PC1 with events by presidential year
x = monthlyIndex['date']
y = monthlyIndex['SentimentPC1']

fig, ax = plt.subplots(1, figsize=(15, 9))
ax.plot(x, y, color='black')

# Presidential year
ax.axvspan(datetime(1985,1,1),datetime(1989,2,1),alpha=0.1, color=colors[7])
ax.text(datetime(1987,1,1), 1.6, 'Ronald\nReagan', fontsize=13, color=colors[7],horizontalalignment='center')

ax.axvspan(datetime(1989,2,1),datetime(1993,2,1),alpha=0.1, color=colors[7])
ax.text(datetime(1991,1,1), 1.6, 'George H. W.\nBush', fontsize=13, color=colors[7],horizontalalignment='center')

ax.axvspan(datetime(1993,2,1),datetime(2001,2,1),alpha=0.1, color=colors[0])
ax.text(datetime(1997,1,1), 1.6, 'Bill\nClinton', fontsize=13, color=colors[0],horizontalalignment='center')

ax.axvspan(datetime(2001,2,1),datetime(2009,2,1),alpha=0.1, color=colors[7])
ax.text(datetime(2005,1,1), 1.6, 'George W.\nBush', fontsize=13, color=colors[7],horizontalalignment='center')

ax.axvspan(datetime(2009,2,1),datetime(2017,2,1),alpha=0.1, color=colors[0])
ax.text(datetime(2013,1,1), 1.6, 'Barack\nObama', fontsize=13, color=colors[0],horizontalalignment='center')

ax.axvspan(datetime(2017,2,1),datetime(2021,2,1),alpha=0.1, color=colors[7])
ax.text(datetime(2019,1,1),1.6, 'Donald\nTrump', fontsize=13, color=colors[7],horizontalalignment='center')

# events
ax.text(datetime(1993,9,1), 0.75, 'Clinton\nhealth care plan', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2001,9,1), -0.75, '9/11', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2006,11,1), 0.73, 'Bush midterm\nelection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2008,9,1), -0.6, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2010,3,1), -1, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2010,10,1),-1.25, 'Deepwater Horizon\noil spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2010,12,1), -1.4, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='center')

ax.text(datetime(2012,6,1), -1, 'Libor\nscandal', fontsize=13, color=colors[4],horizontalalignment='left')

ax.text(datetime(2016,11,1), 0.8 , '2016 presidential\nelection', fontsize=13, color=colors[4],horizontalalignment='center')

#ax.text(datetime(2020,1,1), -0.5, 'Coronavirus\noutbreak', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(x.iloc[0], 'Y')
datemax = np.datetime64(x.iloc[-1], 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Monthly Sentiment Index', fontsize=16)
ax.set_yticks(np.arange(round(min(y), 0) - 0.5, round(max(y), 0) + 1, 0.5))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

# Title
fig.suptitle("Figure 4: Sentiment about Regulation by Presidential Year",
             y=0.95, fontsize=20)
ax.set_title('(January 1985 - January 2021)', fontsize=18,position=(0.5,1.12))

# Notes
fig.text(0.12, 0.03, 'Notes: The sentiment index was estimated using a dictionary-based sentiment analysis'
                    ' approach applied to newspaper text and fixed effects\nregressions. '
                    'For details on the methodology, refer to the latest draft of the Sinclair and Xie paper'
                    ' on "Sentiment and Uncertainty about Regulation".',
         fontsize=14, style='italic')

# Adjust plot position
plt.subplots_adjust(top=0.81, bottom=0.15)

plt.savefig("Figures/SentimentPC1 with events by Presidential Year.jpg", bbox_inches='tight')
plt.savefig("Figures/Figure4.jpg", bbox_inches='tight')
plt.show()
