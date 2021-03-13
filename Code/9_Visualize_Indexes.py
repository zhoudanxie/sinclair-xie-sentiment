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

from sklearn.decomposition import PCA

colors=['#033C5A','#AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F',
        '#033C5A','#AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F']

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------Compute Monthly Reg Relevance Index-------------------------------------------
# Total article counts
totalMonthlyCount=pd.read_csv('Data/totalMonthlyArticleCount.csv')
print(totalMonthlyCount['Newspaper'].value_counts())

# Merge total article count and reg relevant article count
monthlyCount=pd.read_csv('Data/RegRelevant_MonthlyArticleCount.csv')
totalMonthlyCount.loc[totalMonthlyCount['Newspaper']=='LA Times','Newspaper']='Los Angeles Times'
totalMonthlyCount.loc[totalMonthlyCount['Newspaper']=='Washington Post','Newspaper']='The Washington Post'
mergedMonthlyCount=monthlyCount.merge(totalMonthlyCount,on=['Newspaper','Year','Month'],how='left')
mergedMonthlyCount['year-month']=mergedMonthlyCount['Year'].map(str)+'-'+mergedMonthlyCount['Month'].map(str)
print(mergedMonthlyCount.info())

# Function to calculate index
newspapers=mergedMonthlyCount['Newspaper'].unique()
index=mergedMonthlyCount[['year-month']].drop_duplicates().reset_index(drop=True)

T1_start="1985-1"
T1_end="2009-12"
T2_start="1985-1"
T2_end="2009-12"
def calulate_index(var):
    mergedMonthlyCount['X']=mergedMonthlyCount[var]/mergedMonthlyCount['Total article count']
    # Standardization over T1
    for newspaper in newspapers:
        mergedMonthlyCount.loc[mergedMonthlyCount['Newspaper']==newspaper,'variance']=\
            np.var(mergedMonthlyCount[(mergedMonthlyCount['Newspaper']==newspaper) &
            (T1_start <= mergedMonthlyCount['year-month']) & (mergedMonthlyCount['year-month'] <= T1_end)]['X'])
    mergedMonthlyCount['Y']=mergedMonthlyCount['X']/np.sqrt(mergedMonthlyCount['variance'])
    # Multi-paper index
    for month in index['year-month']:
        index.loc[index['year-month'] == month, 'Z'] = np.mean(
            mergedMonthlyCount[mergedMonthlyCount['year-month'] == month]['Y'])
    # Normalization over T2
    M=np.mean(index.loc[(T2_start<=index['year-month']) & (index['year-month']<=T2_end),'Z'])
    new_var=index['Z']*(100/M)
    return new_var

# Compute indexes
index['Reg Relevance']=calulate_index('RegRelevance')

index.drop('Z',axis=1).to_csv('Data/RegRelevanceIndex.csv',index=False)

#-----------------------------------------------------------------------------------------------------------------------
# Plot reg relevance index
index=pd.read_csv('Data/RegRelevanceIndex.csv')
index['date']=index['year-month'].astype('datetime64[ns]').dt.date
print(index.info())

x=index['date']
y=index['Reg Relevance']

fig, ax = plt.subplots(1, figsize=(15,10))
ax.plot(x,y,color=colors[0])

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(index['date'].iloc[0], 'Y')
datemax = np.datetime64(index['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
ax.grid(False)
# ax.set_title('Figure 1: Index of News Coverage on Regulation, January 1985-August 2020',fontsize=22)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14, color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')

ax.set_ylabel('Index of News Attention to Regulation',fontsize=16)
ax.set_yticks(np.arange(60,max(y)+20,20))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig('Figures/Manuscript Figures/Figure1.jpg', bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# Plot reg relevance index with event shading
x=index['date']
y=index['Reg Relevance']

fig, ax = plt.subplots(1, figsize=(15,10))
ax.plot(x,y,color=colors[0])

# events
ax.axvspan(datetime(1997,6,1),datetime(1997,7,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1997,6,1), 120, 'Tobacco settlement', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2001,9,1),datetime(2001,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2001,9,1), 70, '9/11', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2002,7,1),datetime(2002,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2003,1,1), 145, 'Sarbanes-Oxley', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2008,9,1),datetime(2008,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2009,11,1), 150, 'Lehman Brothers', fontsize=13, color=colors[4],horizontalalignment='right')

ax.axvspan(datetime(2010,3,1),datetime(2010,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,3,1), 182, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,4,1),datetime(2010,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2011,3,1), 190, 'Deepwater Horizon\noil spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,7,1),datetime(2010,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,7,1), 170, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='left')

ax.axvspan(datetime(2016,11,1),datetime(2017,3,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2016,11,1), 195, '2016 Trump Election', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(index['date'].iloc[0], 'Y')
datemax = np.datetime64(index['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
ax.grid(False)
# ax.set_title('Figure 1: Index of News Coverage on Regulation, January 1985-August 2020',fontsize=22)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14, color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')

ax.set_ylabel('Index of News Attention to Regulation',fontsize=16)
ax.set_yticks(np.arange(60,max(y)+20,20))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig('Figures/Reg Relevance Index with Events (Jan 1985-Aug 2020).jpg', bbox_inches='tight')
plt.savefig('Figures/Manuscript Figures/Figure1_2.jpg', bbox_inches='tight')
#plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------Plot Monthly Sentiment & Uncertainty Indexes--------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Import the sentiment indexes
monthlyIndex=pd.read_csv('Data/RegRelevant_MonthlySentimentIndex.csv')
print(monthlyIndex.info())

monthlyIndex['Year-Month']=monthlyIndex['Year'].map(str)+'-'+monthlyIndex['Month'].map(str)
monthlyIndex['date']=monthlyIndex['Year-Month'].astype('datetime64[ns]').dt.date

for dict in ['GI','LM','LSD']:
    monthlyIndex[dict+'index_standardized']=(monthlyIndex[dict+'index']-np.mean(monthlyIndex[dict+'index']))/np.std(monthlyIndex[dict+'index'])
monthlyIndex['UncertaintyIndex_standardized']=(monthlyIndex['UncertaintyIndex']-np.mean(monthlyIndex['UncertaintyIndex']))/np.std(monthlyIndex['UncertaintyIndex'])

# PCA of monthly sentiment indexes
features = ['GIindex', 'LMindex', 'LSDindex']
x = monthlyIndex.loc[:, features].values
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print("Variance explained by PC1 and PC2:", pca.explained_variance_ratio_)
print("PC1 feature weights [GI, LM, LSD]:", pca.components_[0])

principalComponents_neg=principalComponents*(-1)
principalDf = pd.DataFrame(data = principalComponents_neg, columns = ['SentimentPC1', 'SentimentPC2'])
monthlyIndex = pd.concat([monthlyIndex, principalDf], axis = 1)

#-----------------------------------------------------------------------------------------------------------------------
# Stationarity tests
# Augmented Dickey-Fuller test (H0: non-stationary)
from arch.unitroot import ADF
def adf_test(var):
    x=monthlyIndex[var]
    adf = ADF(x)
    print('Results of Augmented Dickey-Fuller Test for '+var)
    print("Test statistic:",'{0:0.6f}'.format(adf.stat))
    print('p-value:','{0:0.6f}'.format(adf.pvalue))
    print('Lags:',adf.lags)

adf_test('UncertaintyIndex')
adf_test('LMindex')
adf_test('GIindex')
adf_test('LSDindex')
adf_test('SentimentPC1')

# Phillips-Perron test (H0: non-stationary)
from arch.unitroot import PhillipsPerron
def pp_test(var):
    x=monthlyIndex[var]
    pp = PhillipsPerron(x)
    print('Results of Phillips-Perron Test for '+var)
    print("Test statistic:",'{0:0.6f}'.format(pp.stat))
    print('p-value:','{0:0.6f}'.format(pp.pvalue))
    print('Lags:',pp.lags)

pp_test('UncertaintyIndex')
pp_test('LMindex')
pp_test('GIindex')
pp_test('LSDindex')
pp_test('SentimentPC1')

# KPSS test (H0: stationary)
from arch.unitroot import KPSS
def kpss_test(var):
    x=monthlyIndex[var]
    kpss = KPSS(x)
    print('Results of KPSS Test for '+var)
    print("Test statistic:",'{0:0.6f}'.format(kpss.stat))
    print('p-value:','{0:0.6f}'.format(kpss.pvalue))
    print('Lags:',kpss.lags)

kpss_test('UncertaintyIndex')
kpss_test('LMindex')
kpss_test('GIindex')
kpss_test('LSDindex')
kpss_test('SentimentPC1')

#-----------------------------------------------------------------------------------------------------------------------
# Correlations between sentiment indexes
import scipy.stats

print('LM & GI:',scipy.stats.pearsonr(monthlyIndex['LMindex'], monthlyIndex['GIindex']))
print('LM & LSD:',scipy.stats.pearsonr(monthlyIndex['LMindex'], monthlyIndex['LSDindex']))
print('LSD & GI',scipy.stats.pearsonr(monthlyIndex['LSDindex'], monthlyIndex['GIindex']))

print(np.corrcoef(monthlyIndex['LMindex_standardized'], monthlyIndex['GIindex_standardized']))
print(np.corrcoef(monthlyIndex['LMindex_standardized'], monthlyIndex['LSDindex_standardized']))
print(np.corrcoef(monthlyIndex['GIindex_standardized'], monthlyIndex['LSDindex_standardized']))

#-----------------------------------------------------------------------------------------------------------------------
# Plot monthly uncertainty index
x=monthlyIndex['date']
y=monthlyIndex['UncertaintyIndex']

fig, ax = plt.subplots(1, figsize=(15,10))
ax.plot(x,y,color=colors[0])

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(monthlyIndex['date'].iloc[0], 'Y')
datemax = np.datetime64(monthlyIndex['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Uncertainty Index',fontsize=16)
ax.set_yticks(np.arange(round(min(y),1),round(max(y),1)+0.1,0.1))
ax.set_ylim(bottom=round(min(y),1))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig('Figures/Manuscript Figures/Figure3.jpg', bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Function to plot monthly sentiment index
def plotSentimentIndex(dict):
    dict_name=dict.split('index')[0]
    x=monthlyIndex['date']
    y=monthlyIndex[dict]

    fig, ax = plt.subplots(1, figsize=(15,10))
    ax.plot(x,y,color=colors[0])

    # format the ticks
    years = mdates.YearLocator(2)   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # round to nearest years.
    datemin = np.datetime64(x.iloc[0], 'Y')
    datemax = np.datetime64(x.iloc[-1], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)

    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = lambda x: '$%1.2f' % x
    fig.autofmt_xdate()

    # Set tick and label format
    ax.tick_params(axis='both',which='major',labelsize=14)
    ax.set_ylabel('Sentiment Index',fontsize=16)
    ax.set_yticks(np.arange(round(min(y),1)-0.1,round(max(y),1)+0.4,0.1))
    ax.grid(color='gray', which='major', axis='y', linestyle='dashed')
    # ax.spines['top'].set_color('#d3d3d3')
    # ax.spines['right'].set_color('#ffffff')
    # ax.spines['left'].set_color('#ffffff')
    # ax.spines['bottom'].set_color('#d3d3d3')

    # Title
    fig.suptitle("Figure 6: Monthly Sentiment Index Using " + dict_name + " Dictionary",
                    y=0.95,fontsize=20)
    ax.set_title('(January 1985 - August 2020)',fontsize=18)

    # Inset plot
    xins=x.iloc[-8:]
    yins=y.iloc[-8:]

    axins=inset_axes(ax, width=4, height=2, bbox_to_anchor=(.04, .48, .6, .5),
                        bbox_transform=ax.transAxes,loc=2)

    axins.plot(xins,yins,color='#033C5A',linewidth=2,marker='D',markersize=8)
    axins.format_xdata = mdates.DateFormatter('%Y-%m')
    axins.set_yticks(np.arange(round(min(yins),1)-0.1, round(max(yins),1)+0.2, 0.1))
    axins.grid(color='gray', which='major', axis='y', linestyle='dotted')
    axins.tick_params(axis='both',which='major',labelsize=10)
    axins.set_facecolor('#d3d3d3')
    axins.set_alpha(0.5)
    axins.set_title('Index over the Past Eight Months',fontsize=14,position=(0.5,0.85))

    # Notes
    fig.text(0.12, 0.07,"Notes: The sentiment index is based on the sentiment analysis of expanded regulatory sentences"
                "using the " + dict_name + " dictionary",
             fontsize=14,style='italic')

    plt.savefig("Data/TDM Studio/Analysis of Reg News/Figures/Monthly " + dict_name + " Sentiment Index (Jan 1985-Aug 2020).jpg")
    plt.show()

# Plot sentiment indexes
plotSentimentIndex('GIindex')
plotSentimentIndex('LMindex')
plotSentimentIndex('LSDindex')

#-----------------------------------------------------------------------------------------------------------------------
# Plot three sentiment indexes in one graph
x=monthlyIndex['date']
y1=monthlyIndex['GIindex']
y2=monthlyIndex['LSDindex']
y3=monthlyIndex['LMindex']

fig, ax = plt.subplots(1, figsize=(15,10))
ax.plot(x,y1,color=colors[0],label='GI sentiment index')
ax.plot(x,y2,color=colors[1],label='LSD sentiment index')
ax.plot(x,y3,color=colors[2],label='LM sentiment index')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(x.iloc[0], 'Y')
datemax = np.datetime64(x.iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14)
ax.set_ylabel('Sentiment Index',fontsize=16)
#ax.set_yticks(np.arange(round(min(min(y1),min(y2),min(y3)),1)-0.1,round(max(max(y1),max(y2),max(y3)),1)+0.4,0.1))
ax.grid(color='gray', which='major', axis='y', linestyle='dashed')
# ax.spines['top'].set_color('#d3d3d3')
# ax.spines['right'].set_color('#ffffff')
# ax.spines['left'].set_color('#ffffff')
# ax.spines['bottom'].set_color('#d3d3d3')

# Title
fig.suptitle("Figure 7: Monthly Sentiment Indexes",
                y=0.95,fontsize=20)
ax.set_title('(January 1985 - August 2020)',fontsize=18)
fig.legend(loc='lower left', bbox_to_anchor=(.3, .1, .9, .9), ncol=3, fontsize=14)

# Notes
fig.text(0.12, 0.07,"Notes: The sentiment indexes were estimated using sentiment analyses of expanded regulatory sentences"
            "using the GI, LSD or LM dictionary.",
         fontsize=14,style='italic')

plt.savefig("Figures/Monthly Sentiments Indexes (Jan 1985-Aug 2020).jpg", bbox_inches='tight')

#-----------------------------------------------------------------------------------------------------------------------
# Plot standardized sentiment indexes in one graph
x=monthlyIndex['date']
y1=monthlyIndex['GIindex_standardized']
y2=monthlyIndex['LSDindex_standardized']
y3=monthlyIndex['LMindex_standardized']

fig, ax = plt.subplots(1, figsize=(15,10))
ax.plot(x,y1,color=colors[1],linewidth=1.1,label='GI sentiment index')
ax.plot(x,y2,color='#FF4500',linewidth=1.1,label='LSD sentiment index')
ax.plot(x,y3,color=colors[0],linewidth=1.1,label='LM sentiment index')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(x.iloc[0], 'Y')
datemax = np.datetime64(x.iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Standardized Sentiment Index',fontsize=16)
ax.set_yticks(np.arange(-4,5,1))
ax.set_ylim(bottom=-4)
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

# Title
fig.legend(loc='lower left', bbox_to_anchor=(0.15, 0.01), ncol=3, fontsize=14)
fig.subplots_adjust(bottom=0.15)

plt.savefig('Figures/Manuscript Figures/Figure2.jpg', bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Plot sentiment PC1
x=monthlyIndex['date']
y=monthlyIndex['SentimentPC1']

fig, ax = plt.subplots(1, figsize=(15,10))
ax.plot(x,y,color=colors[0])

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(monthlyIndex['date'].iloc[0], 'Y')
datemax = np.datetime64(monthlyIndex['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('First Principal Component of Sentiment Indexes',fontsize=16)
ymin=-1.2; ymax=1.4
ax.set_yticks(np.arange(ymin,ymax,0.4))
ax.set_ylim(bottom=ymin)
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig('Figures/Monthly Sentiment PC1 (Jan 1985-Aug 2020).jpg', bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Plot monthly uncertainty index with event shading
x=monthlyIndex['date']
y=monthlyIndex['UncertaintyIndex']

fig, ax = plt.subplots(1, figsize=(15,10))
ax.plot(x,y,color=colors[0])

# events
ax.axvspan(datetime(2008,9,1),datetime(2008,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2008,9,1), 0.8, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,3,1),datetime(2010,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,3,1), 0.865, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,4,1),datetime(2010,5,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,10,1), 0.88, 'Deepwater Horizon\noil spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,7,1),datetime(2010,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,7,1), 0.85, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='left')

ax.axvspan(datetime(2016,11,1),datetime(2017,3,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2016,11,1),0.85 , '2016 Trump\nElection', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,3,1),datetime(2020,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,1,1), 0.8, 'Coronavirus\noutbreak', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(monthlyIndex['date'].iloc[0], 'Y')
datemax = np.datetime64(monthlyIndex['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Uncertainty Index',fontsize=16)
ax.set_yticks(np.arange(round(min(y),1),round(max(y),1)+0.1,0.1))
ax.set_ylim(bottom=round(min(y),1))
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

plt.savefig('Figures/Monthly Reg Uncertainty Index with Events (Jan 1985-Aug 2020).jpg', bbox_inches='tight')
#plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Plot standardized sentiment indexes in one graph with events
x=monthlyIndex['date']
y1=monthlyIndex['GIindex_standardized']
y2=monthlyIndex['LSDindex_standardized']
y3=monthlyIndex['LMindex_standardized']

fig, ax = plt.subplots(1, figsize=(18,10))
ax.plot(x,y1,color=colors[1],linewidth=1.1,label='GI sentiment index')
ax.plot(x,y2,color='#FF4500',linewidth=1.1,label='LSD sentiment index')
ax.plot(x,y3,color=colors[0],linewidth=1.1,label='LM sentiment index')

# events
ax.axvspan(datetime(1993,9,1),datetime(1993,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(1993,9,1), 3.5, 'Clinton\nhealth care plan', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2001,9,1),datetime(2001,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2001,9,1), -3, '9/11', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2006,11,1),datetime(2006,12,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2006,11,1), 2.5, 'Bush\nmidterm election', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,3,1),datetime(2010,4,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,3,1), -3.1, 'Obamacare', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,4,1),datetime(2010,5,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,10,1), -3.6, 'Deepwater Horizon\noil spill', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2010,7,1),datetime(2010,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2010,12,1), -3.9, 'Dodd-Frank', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2008,9,1),datetime(2008,10,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2008,9,1), -2.5, 'Lehman\nBrothers', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2012,7,1),datetime(2012,8,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2012,7,1), 2.2, 'Libor\nscandal', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2016,11,1),datetime(2017,3,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2016,11,1),3.7 , '2016 Trump Election', fontsize=13, color=colors[4],horizontalalignment='center')

ax.axvspan(datetime(2020,3,1),datetime(2020,5,1),alpha=0.5, color='#d3d3d3')
ax.text(datetime(2020,1,1), -2.5, 'Coronavirus\noutbreak', fontsize=13, color=colors[4],horizontalalignment='center')

# format the ticks
years = mdates.YearLocator(2)   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(x.iloc[0], 'Y')
datemax = np.datetime64(x.iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14,color='#d3d3d3')
ax.tick_params(axis='both',which='minor',color='#d3d3d3')
ax.set_ylabel('Standardized Sentiment Index',fontsize=18)
ax.set_yticks(np.arange(-4,5,1))
ax.set_ylim(bottom=-4)
ax.grid(color='#d3d3d3', which='major', axis='y')

# Borders
# ax.spines['right'].set_color('#d3d3d3')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('#d3d3d3')
ax.spines['bottom'].set_color('#d3d3d3')

# Title
fig.legend(loc='lower left', bbox_to_anchor=(0.2, 0.01), ncol=3, fontsize=16)
fig.subplots_adjust(bottom=0.15)

plt.savefig('Figures/Monthly Standardized Sentiment Indexes with Events.jpg', bbox_inches='tight')
#plt.show()
