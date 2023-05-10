'''
[To Do]
1. Download CUSIP and County info of callable and noncallable Green munis from bloomberg
2. Using CUSIP, pull (dated date, maturity, coupon) from MSRB
3. Find matching (non-)callable Brown list from MSRB
4. ratings, bond purpose, AMT_ISSUED, callable, federal etc. from fidelity
    
[Update]
    - 'par_traded' added for MSRB DL [March 20, 2021]
    
    - dic_coldwave2019 was mistakenly allocated so revoked in Section 5 [March 6, 2021]
    
    - "countinously callable" added for scraping [March 1, 2021]

    - switched TimeoutException with NoSuchElementException [Feb 28, 2021]
    - nextcalldate scraping => removed "/a" to accompany "--" case [Feb 28, 2021]
    - coupontype, Federally Taxable, Subject to Alternative Minimum Tax	
    
    - Sec2 => # REDUCE: remove itself and green label munis [Feb 27, 2021]
    - df_GBuniverse_GBcallableC, df_BBuniverse_BBcallableC [Feb 27, 2021]
    - fidelity "state" extract [Feb 27, 2021]
    - deleted the mistake .drop_duplicates(keep='first') in Section 2 [Feb 27, 2021]

 --------------------------------------------------------------------------
 
Green muni universe pulled from BBG: Filtering FIELDS are not identical to Export FIELDS
    
    [Filter for bloomberg search]    
    + Dated Date: after June 2013 (LW, p. 10)
    + Call Feature: exclude for exact-matching (DROP: anytime, annual, semi-annual)
    + Coupon Type (Is Floater): fixed only (LW: only fixed-rate vs Baker: exclude floating)
    + xxx: maturity: exclude > 30
    + Federal Tax (Is Federal Taxable) and Alternative Minimum Tax (AMT): exclude both to ensure similar tax treatment (Schwert, 2017)
    + Green Label: Yes

    [Export field: TWO only] => CUSIP + U.S. County Of Issuance (COUTNY: only for green muni only since same issuer)    
    
    via MSRB: + [Cpn, Dated Date, Maturity]
    via fidelity: [Three ratings, bond purpose, AMT_ISSUED, callable, federal]

    - Amount Out (nearly eq. maturit size, but not available in MSRB) => NOT equal to AMT_ISSUED    
    - Rating: what if initially rating match, but later up/downgraded or the other way around?    
    - Although LW allows +/- 1 year maturity date for exact matching, I do NOT.    
    - DATED DATE: the date from which interest on a new issue of municipal securities typically starts to accrue. This date is often used to identify a particular series of bonds of an issuer.

Section 1. l.180 MSRB1
Section 2. l.240 MSRB2
Section 3. l.560 Fidelity
Section 4. l.780 Final matching & merging base + fidelity + county info
Section 5. l.1060 Hazard
Section 6. l.1130 Merging Sec 5 data, MSRB transactions, and statistics
Section 7. l.1410 Kernel univariate
Section 8. l.1600 Kernel bivariate
Section 9. l.1660 Rplot Choropleth
'''
#%% Section 0. ENVIRONMENT

import os
import sys
import pandas as pd
import numpy as np
#import random
import datetime
#from datetime import datetime # ignore yellow warning???
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import wrds
import random
#import re
import time
from datetime import date
from datetime import datetime as dt
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.keys import Keys 
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException

from sklearn.neighbors import KernelDensity
import math
from scipy import stats
#from scipy.stats import wilcoxon # NOT IMPLEMENTABLE => only allows same length x,y
from scipy.stats import mannwhitneyu # FROM: https://stackoverflow.com/questions/33890367/python-wilcoxon-unequal-n/33890615

'''
# MWW vs Wilcoxon signed-rank test: The difference comes from the assumptions. 
#    - In the MWW test you are interested in the difference between two independent populations (null hypothesis: the same, alternative: there is a difference) 
#    - while in Wilcoxon signed-rank test you are interested in testing the same hypothesis but with paired/matched samples.

# The Mann–Whitney U test / Wilcoxon rank-sum test is not the same as the Wilcoxon signed-rank test, although both are nonparametric and involve summation of ranks. The Mann–Whitney U test is applied to independent samples. The Wilcoxon signed-rank test is applied to matched or dependent samples.
# FROM: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test

# The Mann-Whitney U test is a test of equality of distributions for ordinal data. Only if you assume that the distribution is symmetric does it equate to a test of equality of medians (as well, it assumes homogeneity of variance).
# FROM: https://www.researchgate.net/post/What-is-the-difference-between-Mann-Whitney-U-and-Moods-median-test

# Mood’s two-sample test for scale parameters is a non-parametric test for the null hypothesis that two samples are drawn from the same distribution with the same scale parameter.
# FROM: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mood.html

# It is crucial to note, however, that the null hypothesis verified by the Wilcoxon–Mann–Whitney U (and so the Kruskal–Wallis test) is not about medians. The test is sensitive also to differences in scale parameters and symmetry. As a consequence, if the Wilcoxon–Mann–Whitney U test rejects the null hypothesis, one cannot say that the rejection was caused only by the shift in medians. It is easy to prove by simulations, 
# FROM: https://en.wikipedia.org/wiki/Median_test

'''
# location of input and output files
directory_input="/Users/YUMA/Desktop/3rdpaper/Datasets/"
directory_output="/Users/YUMA/Desktop/3rdpaper/Datasets/Output/"
directory_output_fidelity="/Users/YUMA/Desktop/3rdpaper/Datasets/Output/Fidelityscraped/"

# ************ #
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@ For implementing noncallable coupon matching; 1 is stricter (BASELINE), 0 is looser @@@
noncallable_couponmatching = 1; # fidelity scraping is based on 0 (i.e., larger file)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

dic_noncallable_couponmatching = {1 : 'Couponmatching_Yes_strict', 0 : 'Couponmatching_No_loose'} 
dic_noncallable_couponmatching_value = dic_noncallable_couponmatching[noncallable_couponmatching]

# Original Exact Match (1) or LW Within One Year Match (0); either match ALLOWS coupon difference (LW, p.11+35) if noncallable
exactMat_or_withinoneyearMat = 0;

# Moodys, S&P nonexistent => do NOT drop (LW, pp. 33–34)
drop_NANAratingsmatch = 0;

# ************ #

# For importing:
import_dataset = 0

# For exporting: ALWAYS set to ZERO and execute by line
export_dataset = 0

# For checking: ALWAYS set to ZERO and execute by line
check_dataset = 0

# Avoid confusion when Shift + Return
concat_dataset = 0

# filtering data from bloomberg or fidelity; callable data; fixed/float; federally taxable; subject to AMT
bbg_or_fidelity_filter = 0

# remove duplicates rows (Compustat) and only leave the latest: http://kaichen.work/?p=387
drop_duplicate_row = 1

# alarm
alarm_on = 1

# REMOVE DUPLICATE: https://thispointer.com/pandas-find-duplicate-rows-in-a-dataframe-based-on-all-or-selected-columns-using-dataframe-duplicated-in-python/ 
def duplicatefilter(dataframe,identifier,timeseries): # identifier=("GVKEY", "PERMCO"), timeseries="datadate"
    
    global drop_duplicate_row
    
    if drop_duplicate_row == 1:
    
        booleanlist = dataframe.duplicated(subset=[identifier,timeseries], keep='last')
        booleanlist.value_counts()
        
        dataframe["duplicate"] = pd.DataFrame(booleanlist)
        dataframe = dataframe[dataframe['duplicate'] == False] #THIS WILL ONLY PRIVATELY CREATE dataframe!!!
        
    return dataframe.drop(columns='duplicate')


def sexam(series):
    print(series.dtypes); print("\n")
    print("length: "+str(len(series))); print("\n")
    print("unique counts: "+str(series.nunique())); print("\n")
    print("positive counts: "+str(sum(n > 0 for n in series))+", which is "+str(round(sum(n > 0 for n in series)/len(series),3))); print("\n")
    print("zero counts: "+str(series.isin([0]).sum())+", which is "+str(round(series.isin([0]).sum()/len(series),3))); print("\n")
    print("negative counts: "+str(sum(n < 0 for n in series))+", which is "+str(round(sum(n < 0 for n in series)/len(series),3))); print("\n")
    print("npnan counts: "+str(series.isin([np.nan]).sum())+", which is "+str(round(series.isin([np.nan]).sum()/len(series),3))); print("\n")
    print(series.value_counts()); print("\n")
    print(series.describe()); print("\n")
    
    try:
        sns.distplot(series.astype(float)); plt.figure()
        print("No np.nan obs!")
        
        if series.min() > 0:
            sns.distplot(np.log(series.astype(float)))
            print("Log version below")            
        else:
            print("MIN is possibly negative!")
        
    except ValueError:
        sns.distplot(series.astype(float).dropna()); plt.figure()
        print("np.nan obs dropped in display below!")
        
        if series.min() > 0:
            sns.distplot(np.log(series.dropna().astype(float)))
            print("Log version below")
        else:
            print("MIN is possibly negative!")        


# USELESS: dataframe.describe(), dataframe.info
def aexam(dataframe): # array exam
    print("zero counts:")
    print(dataframe.isin([0]).sum()); print("\n"+"np.nan counts:")
    print(dataframe.isin([np.nan]).sum()); print("\n") #SAME: dataframe.isna().sum()
    print(dataframe.dtypes)

# Equivalent to stats.ttest_ind(x, y, equal_var = True)
# TWO SAMPLE MEAN TEST: https://www.jmp.com/en_ch/statistics-knowledge-portal/t-test/two-sample-t-test.html
def twosamplemeantest(p,q):
    mean_diff = p.mean() - q.mean()
    
    std_diff_pre = ( (len(p) - 1) * (p.std())**2 + (len(q) - 1) * (q.std())**2 ) / (len(p) + len(q) - 2)    
    std_diff = math.sqrt(std_diff_pre) * math.sqrt( 1/len(p) + 1/len(q) )
    
    return mean_diff / std_diff

# FROM: https://pythonfordatascienceorg.wordpress.com/welch-t-test-python-pandas/
def welch_ttest(x, y): 
    ## Welch-Satterthwaite Degrees of Freedom ##
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
   
    t, p = stats.ttest_ind(x, y, equal_var = False)
    
    print("\n",
          f"Welch's t-test= {t:.4f}", "\n",
          f"p-value = {p:.4f}", "\n",
          f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}")

#%%    
    
        
#%% Section 1. Import CUSIP of Green Muni downloaded from bloomberg => attach basic info from MSRB
'''
# INPUT: df_GBuniverse['CUSIP']
# OUTPUT: df_GBuniverse['CUSIP','Dated Date','Cpn','Maturity']

For the same CUSIP, (security_description, dated_date, maturity_date) should be identical (FROM msrb_variables_09252018version.pdf)

'''    
# OPEN => CLOSE is INCONVENIENT: db.close()
db = wrds.Connection(); 'ymws21'; 'Apfel3159###';

# LOOP1: EXCEL FILES
for filename in ['GBnoncallable_20210115.xlsx','GBcallable_20210115.xlsx']: #
    
    df_GBuniverse = pd.read_excel(os.path.join(directory_input, filename), sheet_name='Municipals') #, usecols=['CUSIP']    

    df_GBuniverse_extended = pd.DataFrame(); count = 0;
    
    # LOOP2: CUSIP
    for cusip in df_GBuniverse['CUSIP']:
        print(filename + " No." + str(count) + " – CUSIP is: " + cusip + "\n")
        
        # LATER: transaction data e.g., yield, dollar_price, offer_price_takedown_indicator
        appendedbasicinfo = db.raw_sql("SELECT cusip, dated_date, coupon, maturity_date \
                   FROM msrb.msrb \
                   WHERE cusip \
                   IN ('" + cusip + "') \
                   ")
            
        df_GBuniverse_extended = df_GBuniverse_extended.append(appendedbasicinfo)
    
        count += 1
        
    # LOOP1 END: 
    globals()[ 'df_GBuniverse_' + filename.split('_')[0] ] = df_GBuniverse_extended.copy()
    
if alarm_on == 1:
    os.system('say "your first computation is finished"')

# DUPLICATE
df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable.drop_duplicates()    
df_GBuniverse_GBcallable = df_GBuniverse_GBcallable.drop_duplicates()

# RESET INDEX: nesessary for the next indexed step!
df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable.reset_index().drop(columns=['index'])
df_GBuniverse_GBcallable = df_GBuniverse_GBcallable.reset_index().drop(columns=['index'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# EXPORT
if export_dataset == 1:
    df_GBuniverse_GBnoncallable.to_csv(directory_input+"Interim_process_Immuneto_Noncall_CouponMatching/df_GBuniverse_GBnoncallable.csv", index=False)    
    df_GBuniverse_GBcallable.to_csv(directory_input+"Interim_process_Immuneto_Noncall_CouponMatching/df_GBuniverse_GBcallable.csv", index=False)    

# IMPORT
if import_dataset == 1:
    df_GBuniverse_GBnoncallable = pd.read_csv(os.path.join(directory_input, "Interim_process_Immuneto_Noncall_CouponMatching/df_GBuniverse_GBnoncallable.csv"))
    df_GBuniverse_GBcallable = pd.read_csv(os.path.join(directory_input, "Interim_process_Immuneto_Noncall_CouponMatching/df_GBuniverse_GBcallable.csv"))

#%%    
    
    
#%% Section 2. Exact Match or +/-1 Match: Bloomberg Green Muni to Brown Muni via MSRB
'''
# INPUT: GBuniverse_indexlist
# OUTPUT: df_BBuniverse – prepare CUSIP 6-digit from greem muni CUSIP
# ERROR for coupon if using IN => use BETWEEN

    - Witin one year match has 10 times more rows than exact match; yet the DL time is not so different!
    - ???: why sample in exactMat_or_withinoneyearMat = 1 is larger than the MSRB filitering using => AND maturity_date \ #                       IN ('" + globals()[dataframename]['maturity_date'][index] + "') \
    => because BB contains the mirror image of GB

# FAILED: sorting on maturity_year via MSRB is difficult => do it on Spyder
    
'''
# EXPORT: @@@@ exactMat_or_withinoneyearMat = 0 AND right after "your second computation is finished" was announced and duplicatedropped and reduced @@@@
if export_dataset == 1:
    df_BBuniverse_BBnoncallable.to_csv(directory_input+"Interim_process_Immuneto_Noncall_CouponMatching/df_BBuniverse_BBnoncallable.csv", index=False)    
    df_BBuniverse_BBcallable.to_csv(directory_input+"Interim_process_Immuneto_Noncall_CouponMatching/df_BBuniverse_BBcallable.csv", index=False)    

# IMPORT: @@@@ exactMat_or_withinoneyearMat = 0 AND right after "your second computation is finished" was announced and duplicatedropped and reduced @@@@
if import_dataset == 1:
    df_BBuniverse_BBnoncallable = pd.read_csv(os.path.join(directory_input, "Interim_process_Immuneto_Noncall_CouponMatching/df_BBuniverse_BBnoncallable.csv"))
    df_BBuniverse_BBcallable = pd.read_csv(os.path.join(directory_input, "Interim_process_Immuneto_Noncall_CouponMatching/df_BBuniverse_BBcallable.csv"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BELOW SMALL SECTION IS NECESSARY EVEN IF IMPORTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# NEW Column: cusip6, only AFTER DUPLLICATE above!
df_GBuniverse_GBnoncallable['cusip6'] = df_GBuniverse_GBnoncallable['cusip'].str[:6]
df_GBuniverse_GBcallable['cusip6'] = df_GBuniverse_GBcallable['cusip'].str[:6]

# NEW Column: maturity year in int64 type for GB
df_GBuniverse_GBnoncallable['maturity_year'] = df_GBuniverse_GBnoncallable['maturity_date'].astype(str).str[:4].astype(int)
df_GBuniverse_GBcallable['maturity_year'] = df_GBuniverse_GBcallable['maturity_date'].astype(str).str[:4].astype(int)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ABOVE SMALL SECTION IS NECESSARY EVEN IF IMPORTED ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

db = wrds.Connection(); 'ymws21'; 'Apfel3159###';

if exactMat_or_withinoneyearMat == 1: # HERE DONT CHANGE NUMERIC (AVOID CONFUSION)
    
    # LOOP1
    for dataframename in ['df_GBuniverse_GBnoncallable','df_GBuniverse_GBcallable']: #
     
        # xxxDROP DUPLICATE since cannot accomodate (same cusip6, different dated_date) + DELETE: the [ and ] surrounding cusip for SQL
        #GBuniverse_indexlist = globals()[dataframename]['cusip6'].drop_duplicates(keep='first').index.values.astype(int) #https://stackoverflow.com/questions/41217310/get-index-of-a-row-of-a-pandas-dataframe-as-an-integer
        GBuniverse_indexlist = globals()[dataframename][['cusip6','dated_date']].drop_duplicates(subset=['cusip6','dated_date']).index.values.astype(int) #https://stackoverflow.com/questions/41217310/get-index-of-a-row-of-a-pandas-dataframe-as-an-integer
        
        # NEW
        df_BBuniverse = pd.DataFrame(); count = 0;
        
        # LOOP2: matching MSRB
        for index in GBuniverse_indexlist: # Different ways to iterate over rows in Pandas Dataframe: https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
            print( str(dataframename) + " No." + str(count) + ", cusip6dig is: " + globals()[dataframename]['cusip'][index][:6] + "\n")
            
            try:
                # LATER: transaction data e.g., yield, dollar_price, offer_price_takedown_indicator
                df_BBuniverse_current = db.raw_sql("SELECT cusip, dated_date, coupon, maturity_date \
                           FROM msrb.msrb \
                           WHERE cusip \
                           LIKE '" + str( globals()[dataframename]['cusip'][index][:6] ) + "___' \
                           AND dated_date \
                           IN ('" + globals()[dataframename]['dated_date'][index].strftime("%Y/%m/%d").replace("/", "-") + "') \
                           AND maturity_date \
                           IN ('" + globals()[dataframename]['maturity_date'][index].strftime("%Y/%m/%d").replace("/", "-") + "') \
                           ")        
#                           AND coupon \
#                           BETWEEN ('" + str(globals()[dataframename]['coupon'][index]) + "') AND ('" + str(globals()[dataframename]['coupon'][index]) + "') \
        
        
            # IF in Sec 1: imported from excel rather that via MSRB: .strftime("%Y/%m/%d")    
            except AttributeError: 
                df_BBuniverse_current = db.raw_sql("SELECT cusip, dated_date, coupon, maturity_date \
                           FROM msrb.msrb \
                           WHERE cusip \
                           LIKE '" + str( globals()[dataframename]['cusip'][index][:6] ) + "___' \
                           AND dated_date \
                           IN ('" + globals()[dataframename]['dated_date'][index] + "') \
                           AND maturity_date \
                           IN ('" + globals()[dataframename]['maturity_date'][index] + "') \
                           ")
#                           AND coupon \
#                           BETWEEN ('" + str(globals()[dataframename]['coupon'][index]) + "') AND ('" + str(globals()[dataframename]['coupon'][index]) + "') \
                
            df_BBuniverse = df_BBuniverse.append(df_BBuniverse_current)
        
            count += 1
    
        # LOOP1 END
        globals()[ 'df_BBuniverse_' + dataframename.split('_')[2].replace('GB','BB') ] = df_BBuniverse.copy()
    
# +++++++++++ +++++++++++ +++++++++++
        
elif exactMat_or_withinoneyearMat == 0: # HERE DONT CHANGE NUMERIC (AVOID CONFUSION)
    
    # LOOP1
    for dataframename in ['df_GBuniverse_GBnoncallable','df_GBuniverse_GBcallable']: #
     
        # xxxDROP DUPLICATE since cannot accomodate (same cusip6, different dated_date) + DELETE: the [ and ] surrounding cusip for SQL
        #GBuniverse_indexlist = globals()[dataframename]['cusip6'].drop_duplicates(keep='first').index.values.astype(int) #https://stackoverflow.com/questions/41217310/get-index-of-a-row-of-a-pandas-dataframe-as-an-integer
        GBuniverse_indexlist = globals()[dataframename][['cusip6','dated_date']].drop_duplicates(subset=['cusip6','dated_date']).index.values.astype(int) #https://stackoverflow.com/questions/41217310/get-index-of-a-row-of-a-pandas-dataframe-as-an-integer
        
        # NEW
        df_BBuniverse = pd.DataFrame(); count = 0;
        
        # LOOP2: matching MSRB
        for index in GBuniverse_indexlist: # Different ways to iterate over rows in Pandas Dataframe: https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
            print( str(dataframename) + " No." + str(count) + ", cusip6dig is: " + globals()[dataframename]['cusip'][index][:6] + "\n")
            
            try:
                # LATER: transaction data e.g., yield, dollar_price, offer_price_takedown_indicator
                df_BBuniverse_current = db.raw_sql("SELECT cusip, dated_date, coupon, maturity_date \
                           FROM msrb.msrb \
                           WHERE cusip \
                           LIKE '" + str( globals()[dataframename]['cusip'][index][:6] ) + "___' \
                           AND dated_date \
                           IN ('" + globals()[dataframename]['dated_date'][index].strftime("%Y/%m/%d").replace("/", "-") + "') \
                           ")        
    #                       AND maturity_date \
    #                       LIKE '" + str( globals()[dataframename]['maturity_year'][index] ) + "______' \
#                           AND coupon \
#                           BETWEEN ('" + str(globals()[dataframename]['coupon'][index]) + "') AND ('" + str(globals()[dataframename]['coupon'][index]) + "') \
        
        
            # IF in Sec 1: imported from excel rather that via MSRB: .strftime("%Y/%m/%d")
            except AttributeError: 
                df_BBuniverse_current = db.raw_sql("SELECT cusip, dated_date, coupon, maturity_date \
                           FROM msrb.msrb \
                           WHERE cusip \
                           LIKE '" + str( globals()[dataframename]['cusip'][index][:6] ) + "___' \
                           AND dated_date \
                           IN ('" + globals()[dataframename]['dated_date'][index] + "') \
                           ")
    #                       AND maturity_date \
    #                       LIKE '" + str( globals()[dataframename]['maturity_year'][index] ) + "______' \
#                           AND coupon \
#                           BETWEEN ('" + str(globals()[dataframename]['coupon'][index]) + "') AND ('" + str(globals()[dataframename]['coupon'][index]) + "') \
                
            df_BBuniverse = df_BBuniverse.append(df_BBuniverse_current)
        
            count += 1
    
        # LOOP1 END
        globals()[ 'df_BBuniverse_' + dataframename.split('_')[2].replace('GB','BB') ] = df_BBuniverse.copy()
        
        
if alarm_on == 1:
    os.system('say "your second computation is finished"')

####################

if check_dataset == 1:
    
    # QUESTION: why 64987DNA8 shows but not 64987DNK6 (LW, p. 37)?
    df_GBuniverse_GBcallable[ df_GBuniverse_GBcallable.cusip == '64987DNA8' ]    
    df_BBuniverse_BBnoncallable[ df_BBuniverse_BBnoncallable.cusip6 == '64987D' ]
    df_BBuniverse_BBcallable[ df_BBuniverse_BBcallable.cusip6 == '64987D' ]

    # ANSWER: because of => GBuniverse_indexlist = globals()[dataframename]['cusip6'].drop_duplicates(keep='first').index.values.astype(int)            
    df_BBuniverse_BBnoncallable[ df_BBuniverse_BBnoncallable.cusip == '64987DNK6' ]
    df_BBuniverse_BBcallable[ df_BBuniverse_BBcallable.cusip == '64987DNK6' ]
    
####################

# DUPLICATE
df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable.drop_duplicates()    
df_BBuniverse_BBcallable = df_BBuniverse_BBcallable.drop_duplicates()    

# REDUCE A: remove itself and green label munis
df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable[ ~df_BBuniverse_BBnoncallable['cusip'].isin( df_GBuniverse_GBnoncallable['cusip'] ) ]
df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable[ ~df_BBuniverse_BBnoncallable['cusip'].isin( df_GBuniverse_GBcallable['cusip'] ) ]

# UNNECESSARY: checked "nochange" ok
#df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable[ ~df_GBuniverse_GBnoncallable['cusip'].isin( df_BBuniverse_BBnoncallable['cusip'] ) ]

# REDUCE A: remove itself and green label munis
df_BBuniverse_BBcallable = df_BBuniverse_BBcallable[ ~df_BBuniverse_BBcallable['cusip'].isin( df_GBuniverse_GBcallable['cusip'] ) ]
df_BBuniverse_BBcallable = df_BBuniverse_BBcallable[ ~df_BBuniverse_BBcallable['cusip'].isin( df_GBuniverse_GBnoncallable['cusip'] ) ]

# UNNECESSARY: checked "nochange" ok
#df_GBuniverse_GBcallable = df_GBuniverse_GBcallable[ ~df_GBuniverse_GBcallable['cusip'].isin( df_BBuniverse_BBcallable['cusip'] ) ]

# REDUCE B: # filtering data from bloomberg or fidelity; callable data; fixed/float; federally taxable; subject to AMT
if bbg_or_fidelity_filter == 1:
    sys.exit()

# +++++++ +++++++ +++++++ +++++++ +++ DO NOT START FROM HERE EVEN IF IMPORTED +++ +++++++ +++++++ +++++++ +++++++ #

# NEW Column: cusip6, only AFTER DUCPLICATE above!
df_BBuniverse_BBnoncallable['cusip6'] = df_BBuniverse_BBnoncallable['cusip'].str[:6]
df_BBuniverse_BBcallable['cusip6'] = df_BBuniverse_BBcallable['cusip'].str[:6]

# NEW Column: maturity year in int64 type for BB
df_BBuniverse_BBnoncallable['maturity_year'] = df_BBuniverse_BBnoncallable['maturity_date'].astype(str).str[:4].astype(int)
df_BBuniverse_BBcallable['maturity_year'] = df_BBuniverse_BBcallable['maturity_date'].astype(str).str[:4].astype(int)

# MATCHING: maturity AND coupon
if exactMat_or_withinoneyearMat == 1: # obs

    # CONCATENATE: weak condition i.e. exact match ALLOWS coupon difference for noncallable
    if noncallable_couponmatching == 0:
        df_GBuniverse_GBnoncallable['concatenate'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + df_GBuniverse_GBnoncallable['maturity_date'].astype(str) #+ '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + df_BBuniverse_BBnoncallable['maturity_date'].astype(str) #+ '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)

    # CONCATENATE: strong condition i.e. exact match PROHIBITS coupon difference for noncallable
    elif noncallable_couponmatching == 1:
        df_GBuniverse_GBnoncallable['concatenate'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + df_GBuniverse_GBnoncallable['maturity_date'].astype(str) + '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + df_BBuniverse_BBnoncallable['maturity_date'].astype(str) + '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)

    # callable => strict condition only
    df_GBuniverse_GBcallable['concatenate'] = df_GBuniverse_GBcallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBcallable['dated_date'].astype(str) + '_' + df_GBuniverse_GBcallable['maturity_date'].astype(str) + '_' + df_GBuniverse_GBcallable['coupon'].astype(str)
    df_BBuniverse_BBcallable['concatenate'] = df_BBuniverse_BBcallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBcallable['dated_date'].astype(str) + '_' + df_BBuniverse_BBcallable['maturity_date'].astype(str) + '_' + df_BBuniverse_BBcallable['coupon'].astype(str)
    
    # REDUCE C: exact maturity
    df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable[ df_GBuniverse_GBnoncallable['concatenate'].isin( df_BBuniverse_BBnoncallable['concatenate'] ) ]
    df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable[ df_BBuniverse_BBnoncallable['concatenate'].isin( df_GBuniverse_GBnoncallable['concatenate'] ) ]    

    df_GBuniverse_GBcallable = df_GBuniverse_GBcallable[ df_GBuniverse_GBcallable['concatenate'].isin( df_BBuniverse_BBcallable['concatenate'] ) ]
    df_BBuniverse_BBcallable = df_BBuniverse_BBcallable[ df_BBuniverse_BBcallable['concatenate'].isin( df_GBuniverse_GBcallable['concatenate'] ) ]    
    
# MATCHING: +/- one year maturity difference
elif exactMat_or_withinoneyearMat == 0: # obs

    # CONCATENATE: WEAK condition ALLOWS coupon difference for noncallable
    if noncallable_couponmatching == 0:        
        df_GBuniverse_GBnoncallable['concatenate_plus'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] + 1).astype(str) #+ '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_GBuniverse_GBnoncallable['concatenate_zero'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] + 0).astype(str) #+ '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_GBuniverse_GBnoncallable['concatenate_minu'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] - 1).astype(str) #+ '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
    
        df_BBuniverse_BBnoncallable['concatenate_plus'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] + 1).astype(str) #+ '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate_zero'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] + 0).astype(str) #+ '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate_minu'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] - 1).astype(str) #+ '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)

    # CONCATENATE: STRONG condition PROHIBITS coupon difference for noncallable
    elif noncallable_couponmatching == 1:        
        df_GBuniverse_GBnoncallable['concatenate_plus'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] + 1).astype(str) + '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_GBuniverse_GBnoncallable['concatenate_zero'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] + 0).astype(str) + '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_GBuniverse_GBnoncallable['concatenate_minu'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] - 1).astype(str) + '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
    
        df_BBuniverse_BBnoncallable['concatenate_plus'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] + 1).astype(str) + '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate_zero'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] + 0).astype(str) + '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate_minu'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] - 1).astype(str) + '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)

    # CONCATENATE: STRONG condition PROHIBITS coupon difference for callable
    df_GBuniverse_GBcallable['concatenate_plus'] = df_GBuniverse_GBcallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBcallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBcallable['maturity_year'] + 1).astype(str) + '_' + df_GBuniverse_GBcallable['coupon'].astype(str)
    df_GBuniverse_GBcallable['concatenate_zero'] = df_GBuniverse_GBcallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBcallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBcallable['maturity_year'] + 0).astype(str) + '_' + df_GBuniverse_GBcallable['coupon'].astype(str)
    df_GBuniverse_GBcallable['concatenate_minu'] = df_GBuniverse_GBcallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBcallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBcallable['maturity_year'] - 1).astype(str) + '_' + df_GBuniverse_GBcallable['coupon'].astype(str)

    df_BBuniverse_BBcallable['concatenate_plus'] = df_BBuniverse_BBcallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBcallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBcallable['maturity_year'] + 1).astype(str) + '_' + df_BBuniverse_BBcallable['coupon'].astype(str)
    df_BBuniverse_BBcallable['concatenate_zero'] = df_BBuniverse_BBcallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBcallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBcallable['maturity_year'] + 0).astype(str) + '_' + df_BBuniverse_BBcallable['coupon'].astype(str)
    df_BBuniverse_BBcallable['concatenate_minu'] = df_BBuniverse_BBcallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBcallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBcallable['maturity_year'] - 1).astype(str) + '_' + df_BBuniverse_BBcallable['coupon'].astype(str)
        
    # ORIGINAL: for testing
    df_GBuniverse_GBnoncallable_original = df_GBuniverse_GBnoncallable.copy()
    df_BBuniverse_BBnoncallable_original = df_BBuniverse_BBnoncallable.copy()
    #df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable_original.copy()
    #df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable_original.copy()
    
    # REDUCE C: maturity
    df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable[ df_GBuniverse_GBnoncallable['concatenate_plus'].isin( df_BBuniverse_BBnoncallable['concatenate_zero'] ) |
                                                               df_GBuniverse_GBnoncallable['concatenate_zero'].isin( df_BBuniverse_BBnoncallable['concatenate_zero'] ) |
                                                               df_GBuniverse_GBnoncallable['concatenate_minu'].isin( df_BBuniverse_BBnoncallable['concatenate_zero'] ) ]
    # NECESSARY: the order in interchangeable btw GB and BB
    df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable[ df_BBuniverse_BBnoncallable['concatenate_plus'].isin( df_GBuniverse_GBnoncallable['concatenate_zero'] ) |
                                                               df_BBuniverse_BBnoncallable['concatenate_zero'].isin( df_GBuniverse_GBnoncallable['concatenate_zero'] ) |
                                                               df_BBuniverse_BBnoncallable['concatenate_minu'].isin( df_GBuniverse_GBnoncallable['concatenate_zero'] ) ]

    df_GBuniverse_GBcallable = df_GBuniverse_GBcallable[ df_GBuniverse_GBcallable['concatenate_plus'].isin( df_BBuniverse_BBcallable['concatenate_zero'] ) |
                                                               df_GBuniverse_GBcallable['concatenate_zero'].isin( df_BBuniverse_BBcallable['concatenate_zero'] ) |
                                                               df_GBuniverse_GBcallable['concatenate_minu'].isin( df_BBuniverse_BBcallable['concatenate_zero'] ) ]
    # NECESSARY: the order in interchangeable btw GB and BB
    df_BBuniverse_BBcallable = df_BBuniverse_BBcallable[ df_BBuniverse_BBcallable['concatenate_plus'].isin( df_GBuniverse_GBcallable['concatenate_zero'] ) |
                                                               df_BBuniverse_BBcallable['concatenate_zero'].isin( df_GBuniverse_GBcallable['concatenate_zero'] ) |
                                                               df_BBuniverse_BBcallable['concatenate_minu'].isin( df_GBuniverse_GBcallable['concatenate_zero'] ) ]

####################

# INDEX: not essential
df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable.reset_index().drop(columns=['index'])
df_GBuniverse_GBcallable = df_GBuniverse_GBcallable.reset_index().drop(columns=['index'])

df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable.reset_index().drop(columns=['index'])
df_BBuniverse_BBcallable = df_BBuniverse_BBcallable.reset_index().drop(columns=['index'])

# CHECK: exactly equal in both cases
len(df_GBuniverse_GBnoncallable['cusip6'].unique()) == len(df_BBuniverse_BBnoncallable['cusip6'].unique())
len(df_GBuniverse_GBcallable['cusip6'].unique()) == len(df_BBuniverse_BBcallable['cusip6'].unique())

# PARTITION
df_GBuniverse_GBnoncallableA = df_GBuniverse_GBnoncallable[:300] 
df_GBuniverse_GBnoncallableB = df_GBuniverse_GBnoncallable[300:600]
df_GBuniverse_GBnoncallableC = df_GBuniverse_GBnoncallable[600:]
df_BBuniverse_BBnoncallableA = df_BBuniverse_BBnoncallable[:300] 
df_BBuniverse_BBnoncallableB = df_BBuniverse_BBnoncallable[300:600]
df_BBuniverse_BBnoncallableC = df_BBuniverse_BBnoncallable[600:900]
df_BBuniverse_BBnoncallableD = df_BBuniverse_BBnoncallable[900:]

df_GBuniverse_GBcallableA = df_GBuniverse_GBcallable[:300] 
df_GBuniverse_GBcallableB = df_GBuniverse_GBcallable[300:]
df_BBuniverse_BBcallableA = df_BBuniverse_BBcallable[:300] 
df_BBuniverse_BBcallableB = df_BBuniverse_BBcallable[300:]

if check_dataset == 1:    
    df_GBuniverse_GBnoncallable.to_excel(directory_input+"df_GBuniverse_GBnoncallable.xlsx", index=False)
    df_BBuniverse_BBnoncallable.to_excel(directory_input+"df_BBuniverse_BBnoncallable.xlsx", index=False)    
    df_GBuniverse_GBcallable.to_excel(directory_input+"df_GBuniverse_GBcallable.xlsx", index=False)        
    df_BBuniverse_BBcallable.to_excel(directory_input+"df_BBuniverse_BBcallable.xlsx", index=False)    
    
'''    
df_GBuniverse_GBnoncallable[ df_GBuniverse_GBnoncallable['cusip6'].isin( df_BBuniverse_BBnoncallable['cusip6'] ) ]
df_GBuniverse_GBcallable[ df_GBuniverse_GBcallable['cusip6'].isin( df_BBuniverse_BBcallable['cusip6'] ) ]

df_GBuniverse_GBnoncallable[df_GBuniverse_GBnoncallable.cusip6 == '13032U' ] #13032UMR6
df_BBuniverse_BBnoncallable[df_BBuniverse_BBnoncallable.cusip6 == '13032U' ]

'''
#%%


#%% Section 3. Scraping ratings: Fidelity
'''
If errors are generated frequently, it means access is restrited => use VPN

'''

login = 1; alarm_on = 1; chromedriver = '/usr/local/bin/chromedriver'; driver = webdriver.Chrome(chromedriver);

driver.maximize_window(); driver.implicitly_wait(10);

# Fidelity
driver.get('https://guest.fidelity.com/GA/profile?ccsource=em_Acquisition_GuestAccessWelcome_11_28_2016')

WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, "#Link_1477394234528"))).click()

driver.find_element_by_xpath('//*[@id="member-id-field"]').send_keys('malcom') # ymws21
driver.find_element_by_xpath('//*[@id="password-field"]').send_keys('02213159') # Apfel3159###
                            
WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, "body > div.layout-body-region > table > tbody > tr:nth-child(2) > td > table > tbody > tr:nth-child(1) > td.form-cell > form > div > table > tbody > tr:nth-child(5) > td > a"))).click()

####################################
# TIME
timerstart_scraping = time.time()

# LOOP1: @@@ INPUT HERE @@@
for dataframename2 in ['df_BBuniverse_BBnoncallableD']: # 'df_GBuniverse_GBnoncallableA','df_GBuniverse_GBnoncallableB','df_GBuniverse_GBnoncallableC','df_BBuniverse_BBnoncallableA','df_BBuniverse_BBnoncallableB','df_BBuniverse_BBnoncallableC','df_BBuniverse_BBnoncallableD'
                                                                                        # 'df_GBuniverse_GBcallableA','df_GBuniverse_GBcallableB','df_BBuniverse_BBcallableA','df_BBuniverse_BBcallableB'
    # RENEW: for each LOOP1
    df_fidelity = pd.DataFrame(); count = 0;
    
    # LOOP2: cusip9
    for cusipnum in globals()[dataframename2]['cusip']: # ['262668AE6','59261AG35','59261AG43','59261APZ4','59261AG76','59261AQE0','59261APY7','59261ANE3','59261APQ4','59261AQK6','706643CH4','59261ANG8','59261AXR3','59261APR2','59261APW1','59261ACW5','6460666L9','520066JS1','59261AQG5','59261ADN4']
        
        # PRINT
        print(dataframename2 + " No." + str(count) + " – CUSIP is: " + cusipnum + "\n")
        
        # SITE1
        driver.get('https://fixedincome.fidelity.com/ftgw/fi/FILanding')
    
        # CUSIP NUM
        driver.find_element_by_xpath('//*[@id="bondSearchField"]').send_keys(str(cusipnum))
        
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "#findBondsForm > div.searchBox > button"))).click()
    
        try:
            # BOND INFO    
            # description    
#            description = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[4]/td[2]/a').text
            # coupon
#            coupon = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[4]/td[3]').text
            # maturity
#            maturity = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[4]/td[4]').text
            # next call date: xxxCAUSES ERROR IF --
            #nextcalldate = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[4]/td[5]/a').text            
            nextcalldate = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[4]/td[5]').text
            # moodys
#            moodys = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[4]/td[6]').text    
            # S&P
#            sandp = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[4]/td[7]').text    
            # YTM
#            ytm = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[4]/td[12]').text    

            WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "#FIBondTabtheForm > table > tbody > tr:nth-child(4) > td.description-alignment > a"))).click()
    
        except NoSuchElementException: #TimeoutException: 
            
            try:        
#                description = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[5]/td[2]/a').text
                # coupon
#                coupon = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[5]/td[3]').text
                # maturity
#                maturity = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[5]/td[4]').text
                # next call date: xxxCAUSES ERROR IF --
                #nextcalldate = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[5]/td[5]/a').text
                nextcalldate = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[5]/td[5]').text                
                # moodys
#                moodys = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[5]/td[6]').text    
                # S&P
#                sandp = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[5]/td[7]').text    
                # YTM
#                ytm = driver.find_element_by_xpath('//*[@id="FIBondTabtheForm"]/table/tbody/tr[5]/td[12]').text    

                WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "#FIBondTabtheForm > table > tbody > tr:nth-child(5) > td.description-alignment > a"))).click()
            
            except NoSuchElementException: #TimeoutException: # e.g., 271015NW5
                
                df_fidelity = df_fidelity.append(pd.Series([cusipnum, "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"], #, "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"
                                                    index=["CUSIP", "nextcalldate", "callprotection", "continuouslycallable", "calldefeased", "calledbonds", "makewholecall", "conditionalcall", "putoption", "prerefunded", "prerefundedprice", "originalissueamt", "proceeduse", "state", "fedtaxable", "subjectamt", "moodyscurrent", "moodyscurrentdate", "spcurrent", "spcurrentdate", "coupontype"]), ignore_index=True) #,"description", "coupon", "maturity", "moodys", "sandp", "ytm", "payfrequency", "moodysprior", "spprior", "coupontype", "issuedate", "dateddate"
                # LOOP2 MIDDLE END:
                count += 1                
                continue
    
        # seconds
        time.sleep(random.randint(1, 3))
        
        # SITE2: @@@ [MOVED TO ABOVE SEPARATELY!!! + NoSuchElementException SWITCHED TO TimeoutException] @@@ => AGAIN SWITCHED [Feb 28, 2021]
#        try:
#            WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
#                (By.CSS_SELECTOR, "#FIBondTabtheForm > table > tbody > tr:nth-child(4) > td.description-alignment > a"))).click()
#        except TimeoutException:
#            WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
#                (By.CSS_SELECTOR, "#FIBondTabtheForm > table > tbody > tr:nth-child(5) > td.description-alignment > a"))).click()
    
        # pay frequency
        #payfrequency = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[1]/tbody/tr[2]/td').text
        # CALL PROTECTION
        callprotection = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[2]/tbody/tr[1]/td').text
        # CONT CALLABLE
        continuouslycallable = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[2]/tbody/tr[2]/td').text
        # CALL DEFEASED
        calldefeased = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[2]/tbody/tr[3]/td').text
        # CALLED BONDS
        calledbonds = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[2]/tbody/tr[4]/td').text
        # MAKE WHOLE CALL
        makewholecall = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[2]/tbody/tr[5]/td').text
        # CONDITIONAL CALL
        conditionalcall = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[2]/tbody/tr[6]/td').text

        # PUT OPTION
        putoption = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[2]/tbody/tr[12]/td').text
        # PREREFUNDED
        prerefunded = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[2]/tbody/tr[13]/td').text
        # PREREFUNDED PRICE
        prerefundedprice = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[2]/tbody/tr[14]/td').text


        # ORIGINAL ISSUE AMT
        originalissueamt = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[3]/tbody/tr[7]/td').text
        # PROCEEDS USE
        proceeduse = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[4]/tbody/tr[1]/td').text
        # STATE
        state = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[4]/tbody/tr[2]/td').text
        # FEDERALLY TAXABLE
        fedtaxable = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[4]/tbody/tr[3]/td').text
        # SUBJECT TO AMT
        subjectamt = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[4]/tbody/tr[4]/td').text
        
        # Moodys current
        moodyscurrent = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[1]/tbody/tr[3]/td[2]').text
        # Moodys current date
        moodyscurrentdate = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[1]/tbody/tr[4]/td[2]/span').text    
        # Moodys prior
        #moodysprior = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[1]/tbody/tr[3]/td[3]').text
    
        try:
            # SP current
            spcurrent = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[1]/tbody/tr[12]/td[2]').text
            # SP current date
            spcurrentdate = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[1]/tbody/tr[13]/td[2]/span').text    
            # SP prior
            #spprior = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[1]/tbody/tr[12]/td[3]').text
    
        except NoSuchElementException:
            # SP current
            spcurrent = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[1]/tbody/tr[13]/td[2]').text
            # SP current date
            spcurrentdate = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[1]/tbody/tr[14]/td[2]/span').text    
            # SP prior
            #spprior = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[1]/tbody/tr[13]/td[3]').text
        
        # coupon type
        coupontype = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[2]/table[2]/tbody/tr[1]/td').text
        
        # issue date
        #issuedate = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[3]/tbody/tr[1]/td').text
        # dateddate
        #dateddate = driver.find_element_by_xpath('//*[@id="overview"]/table/tbody/tr/td[1]/table[3]/tbody/tr[2]/td').text
        
        # APPEND
        df_fidelity = df_fidelity.append(pd.Series([cusipnum, nextcalldate, callprotection, continuouslycallable, calldefeased, calledbonds, makewholecall, conditionalcall, putoption, prerefunded, prerefundedprice, originalissueamt, proceeduse, state, fedtaxable, subjectamt, moodyscurrent, moodyscurrentdate, spcurrent, spcurrentdate, coupontype], #, description, coupon, maturity, moodys, sandp, ytm, payfrequency, moodysprior, spprior, coupontype, issuedate, dateddate
                                            index=["CUSIP", "nextcalldate", "callprotection", "continuouslycallable", "calldefeased", "calledbonds", "makewholecall", "conditionalcall", "putoption", "prerefunded", "prerefundedprice", "originalissueamt", "proceeduse", "state", "fedtaxable", "subjectamt", "moodyscurrent", "moodyscurrentdate", "spcurrent", "spcurrentdate", "coupontype"]), ignore_index=True) #,"description", "coupon", "maturity", "moodys", "sandp", "ytm", "payfrequency", "moodysprior", "spprior", "coupontype", "issuedate", "dateddate"
        # seconds
        time.sleep(random.randint(1, 3))
        
        # LOOP2 END:
        count += 1
        
    # LOOP1 END:
    globals()[ 'df_fidelity_' + dataframename2.split('_')[2] ] = df_fidelity.copy()

    #if alarm_on == 1:
    #    os.system('say "Go to the next batch"')

# TIME
timerend_scraping = time.time(); print( str( (timerend_scraping - timerstart_scraping) / 60 ) + " minutes for scraping" )

if alarm_on == 1:
    os.system('say "your scraping is finished"')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# EXPORT
if export_dataset == 1:
    df_fidelity_GBnoncallableA.to_csv(directory_output_fidelity+"df_fidelity_GBnoncallableA.csv", index=False)
    df_fidelity_GBnoncallableB.to_csv(directory_output_fidelity+"df_fidelity_GBnoncallableB.csv", index=False)
    df_fidelity_GBnoncallableC.to_csv(directory_output_fidelity+"df_fidelity_GBnoncallableC.csv", index=False)    

    df_fidelity_BBnoncallableA.to_csv(directory_output_fidelity+"df_fidelity_BBnoncallableA.csv", index=False)
    df_fidelity_BBnoncallableB.to_csv(directory_output_fidelity+"df_fidelity_BBnoncallableB.csv", index=False)
    df_fidelity_BBnoncallableC.to_csv(directory_output_fidelity+"df_fidelity_BBnoncallableC.csv", index=False)    
    df_fidelity_BBnoncallableD.to_csv(directory_output_fidelity+"df_fidelity_BBnoncallableD.csv", index=False)        

    df_fidelity_GBcallableA.to_csv(directory_output_fidelity+"df_fidelity_GBcallableA.csv", index=False)
    df_fidelity_GBcallableB.to_csv(directory_output_fidelity+"df_fidelity_GBcallableB.csv", index=False)

    df_fidelity_BBcallableA.to_csv(directory_output_fidelity+"df_fidelity_BBcallableA.csv", index=False)
    df_fidelity_BBcallableB.to_csv(directory_output_fidelity+"df_fidelity_BBcallableB.csv", index=False)
    
# IMPORT
if import_dataset == 1:
    df_fidelity_GBnoncallableA = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_GBnoncallableA.csv'))
    df_fidelity_GBnoncallableB = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_GBnoncallableB.csv'))
    df_fidelity_GBnoncallableC = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_GBnoncallableC.csv'))    

    df_fidelity_BBnoncallableA = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_BBnoncallableA.csv'))
    df_fidelity_BBnoncallableB = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_BBnoncallableB.csv'))
    df_fidelity_BBnoncallableC = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_BBnoncallableC.csv'))    
    df_fidelity_BBnoncallableD = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_BBnoncallableD.csv'))    
    
    df_fidelity_GBcallableA = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_GBcallableA.csv'))
    df_fidelity_GBcallableB = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_GBcallableB.csv'))

    df_fidelity_BBcallableA = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_BBcallableA.csv'))
    df_fidelity_BBcallableB = pd.read_csv(os.path.join(directory_output_fidelity, 'df_fidelity_BBcallableB.csv'))

    # NEW CONCAT: inside the block => just to avoid confusion when Shift + Return
    df_fidelity_GBnoncallable = pd.concat( [df_fidelity_GBnoncallableA, df_fidelity_GBnoncallableB, df_fidelity_GBnoncallableC], axis=0 )
    df_fidelity_BBnoncallable = pd.concat( [df_fidelity_BBnoncallableA, df_fidelity_BBnoncallableB, df_fidelity_BBnoncallableC, df_fidelity_BBnoncallableD], axis=0 )
    df_fidelity_GBcallable = pd.concat( [df_fidelity_GBcallableA, df_fidelity_GBcallableB], axis=0 )    
    df_fidelity_BBcallable = pd.concat( [df_fidelity_BBcallableA, df_fidelity_BBcallableB], axis=0 )        

#%%
   

#%% Section 4. Final matching & merging base + fidelity + county info
'''
[Conservative approach to determinig callability]

 - Completely relabeling green/brown by fidelity data instead of bloomberg is doable but time-consuming
   => rather, take a conservative approach dropping data that have mismatch between bbg an fidelity

[Callability matching rules]
 - if Next Call Date is "--" or "View" then noncallable except for countinously callable case; if specific date then callable
 - also, there is a specific case 130536RK3 of "View" being callable => but this appears to be very rare 
 - the "no data case" has to be dropped => "Fidelity's current fixed income offerings do not match your search criteria or the offering period has closed for this offering. Please try your search again with new criteria or speak to a representative at 1-800-544-6666 for additional assistance."


[Usefull parameters] 
 - call protection is should NOT be used at all as criterion => majority of GB noncallable bonds are YES
 - calldefeased / conditionalcall does not seem to help since NO is absolutely dominant

 - conditionallcallable is YES => definitely callable bond

 - makewholcall appears to be dominated by nextcalldate in terms of informativeness so does not help (?)
 - putoption appears to be dominated by continuoslycallable in terms of informativeness so does not help (?)
 
'''
if export_dataset == 1: # BEFORE transform, and rating matching
    df_GBuniverse_GBnoncallable.to_csv( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_GBuniverse_GBnoncallable_Merged.csv"), index=False)    
    df_BBuniverse_BBnoncallable.to_csv( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_BBuniverse_BBnoncallable_Merged.csv"), index=False)    
    df_GBuniverse_GBcallable.to_csv( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_GBuniverse_GBcallable_Merged.csv"), index=False)    
    df_BBuniverse_BBcallable.to_csv( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_BBuniverse_BBcallable_Merged.csv"), index=False)    

# @@@@@@@@@@@ FIDELITY IS ALREADY INCORPORATED SO NO NEED TO GO TO SECTION 1–3 @@@@@@@@@@@    
# @@@@@@@@@@@ NOTWITHSTANDING, I WOULD USUALLY START AT LEAST FROM SECTION 5 AND 6, BUT IDEALLY SEC 7 @@@@@@@@@@@    
if import_dataset == 1: # BEFORE transform, and rating matching
    df_GBuniverse_GBnoncallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_GBuniverse_GBnoncallable_Merged.csv'))
    df_BBuniverse_BBnoncallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_BBuniverse_BBnoncallable_Merged.csv'))
    df_GBuniverse_GBcallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_GBuniverse_GBcallable_Merged.csv'))
    df_BBuniverse_BBcallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_BBuniverse_BBcallable_Merged.csv'))

# ==== ==== ==== ==== ==== ==== ==== ==== [SKIP BLOCK BELOW IF IMPORTED] ==== ==== ==== ==== ==== ==== ==== ==== #

# REDUCE B: filtering data from bloomberg or fidelity; callable data; fixed/float; federally taxable; subject to AMT
# DOMINO: reduction here would render BB-GB matching unbalanced again => ulterior matching in Sec 6 necessary

if bbg_or_fidelity_filter == 0:    
    # REDUCE B1: common across noncall and call
    df_fidelity_GBnoncallable = df_fidelity_GBnoncallable[ (df_fidelity_GBnoncallable['coupontype'] == 'FIXED') & (df_fidelity_GBnoncallable['fedtaxable'] == 'NO') & (df_fidelity_GBnoncallable['subjectamt'] == 'NO') ]
    df_fidelity_BBnoncallable = df_fidelity_BBnoncallable[ (df_fidelity_BBnoncallable['coupontype'] == 'FIXED') & (df_fidelity_BBnoncallable['fedtaxable'] == 'NO') & (df_fidelity_BBnoncallable['subjectamt'] == 'NO') ]
    df_fidelity_GBcallable = df_fidelity_GBcallable[ (df_fidelity_GBcallable['coupontype'] == 'FIXED') & (df_fidelity_GBcallable['fedtaxable'] == 'NO') & (df_fidelity_GBcallable['subjectamt'] == 'NO') ]
    df_fidelity_BBcallable = df_fidelity_BBcallable[ (df_fidelity_BBcallable['coupontype'] == 'FIXED') & (df_fidelity_BBcallable['fedtaxable'] == 'NO') & (df_fidelity_BBcallable['subjectamt'] == 'NO') ]

    # REDUCE B2: callability => --(2) is noncallable; VIEW(4) depends (e.g., countinuous callable); MM/DD/YYYY is callable
    df_fidelity_GBnoncallable = df_fidelity_GBnoncallable[ (df_fidelity_GBnoncallable['nextcalldate'].str.len() <= 4) & (df_fidelity_GBnoncallable['continuouslycallable'] != 'YES') ]
    df_fidelity_BBnoncallable = df_fidelity_BBnoncallable[ (df_fidelity_BBnoncallable['nextcalldate'].str.len() <= 4) & (df_fidelity_BBnoncallable['continuouslycallable'] != 'YES') ]

    # REDUCE B3: call date exists OR continuouslycallable 
    df_fidelity_GBcallable = df_fidelity_GBcallable[ (df_fidelity_GBcallable['nextcalldate'].str.len() > 4) | (df_fidelity_GBcallable['continuouslycallable'] == 'YES') ]
    df_fidelity_BBcallable = df_fidelity_BBcallable[ (df_fidelity_BBcallable['nextcalldate'].str.len() > 4) | (df_fidelity_BBcallable['continuouslycallable'] == 'YES') ]    

# *********************

# MERGE: fidelity
df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable.merge( df_fidelity_GBnoncallable.rename(columns={'CUSIP':'cusip'}), on=['cusip'] )
df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable.merge( df_fidelity_BBnoncallable.rename(columns={'CUSIP':'cusip'}), on=['cusip'] )
df_GBuniverse_GBcallable = df_GBuniverse_GBcallable.merge( df_fidelity_GBcallable.rename(columns={'CUSIP':'cusip'}), on=['cusip'] )
df_BBuniverse_BBcallable = df_BBuniverse_BBcallable.merge( df_fidelity_BBcallable.rename(columns={'CUSIP':'cusip'}), on=['cusip'] )

# MERGE: attach County info => no need for how='left' since row num does not change anyway
df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable.merge( pd.read_excel(os.path.join(directory_input, 'GBnoncallable_20210115.xlsx'), sheet_name='Municipals').rename(columns={'CUSIP':'cusip'}), on=['cusip'] ) #
df_GBuniverse_GBcallable = df_GBuniverse_GBcallable.merge( pd.read_excel(os.path.join(directory_input, 'GBcallable_20210115.xlsx'), sheet_name='Municipals').rename(columns={'CUSIP':'cusip'}), on=['cusip'] ) #

# EXPAND: 
df_GBuniverse_GBnoncallable[['County','State']] = df_GBuniverse_GBnoncallable['U.S. County Of Issuance'].str.split(', ',expand=True,)
df_GBuniverse_GBcallable[['County','State']] = df_GBuniverse_GBcallable['U.S. County Of Issuance'].str.split(', ',expand=True,)

# MERGE left: attach County to BB universe WITH how='left' => N/A data approx. 1%? but issuance state available (e.g., fidelity) => indicating that issued at state level
df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable.merge( df_GBuniverse_GBnoncallable[['cusip6','U.S. County Of Issuance','County','State']].drop_duplicates(), on=['cusip6'], how='left' )
df_BBuniverse_BBcallable = df_BBuniverse_BBcallable.merge( df_GBuniverse_GBcallable[['cusip6','U.S. County Of Issuance','County','State']].drop_duplicates(), on=['cusip6'], how='left' )
    
# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ [ START FROM HERE ] ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #

# TRANSFORM: $ remove
df_GBuniverse_GBnoncallable['originalissueamt'] = df_GBuniverse_GBnoncallable['originalissueamt'].str.replace("$","").str.replace(",","").str.replace(",","").replace("--", np.nan).astype(float)
df_BBuniverse_BBnoncallable['originalissueamt'] = df_BBuniverse_BBnoncallable['originalissueamt'].str.replace("$","").str.replace(",","").str.replace(",","").replace("--", np.nan).astype(float)
df_GBuniverse_GBcallable['originalissueamt'] = df_GBuniverse_GBcallable['originalissueamt'].str.replace("$","").str.replace(",","").str.replace(",","").replace("--", np.nan).astype(float)
df_BBuniverse_BBcallable['originalissueamt'] = df_BBuniverse_BBcallable['originalissueamt'].str.replace("$","").str.replace(",","").str.replace(",","").replace("--", np.nan).astype(float)

# TRANSFORM: MM
df_GBuniverse_GBnoncallable['originalissueamt'] = df_GBuniverse_GBnoncallable['originalissueamt'] / 1000000
df_BBuniverse_BBnoncallable['originalissueamt'] = df_BBuniverse_BBnoncallable['originalissueamt'] / 1000000
df_GBuniverse_GBcallable['originalissueamt'] = df_GBuniverse_GBcallable['originalissueamt'] / 1000000
df_BBuniverse_BBcallable['originalissueamt'] = df_BBuniverse_BBcallable['originalissueamt'] / 1000000

# CHECK
#df_GBuniverse_GBnoncallable.groupby(['State'])['State'].count()
#df_GBuniverse_GBnoncallable.groupby(['County'])['County'].count()

#######################################################################################################

# REDUCE D: Moodys AND S&P => BASELINE is AND; 2 is intended to drop when ratings are not available for both
# DOMINO: reduction here would render BB-GB matching unbalanced again => ulterior matching in Sec 6 necessary

################################ [ MOVED TO THE BOTTOM OF THIS SCRIPT] ################################

# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #

# AGG RATING: https://wolfstreet.com/credit-rating-scales-by-moodys-sp-and-fitch/
# NOTE: coupontype REMARKETED, which is dropped due to ['coupontype'] == 'FIXED', has ratings of VMIG1 => https://www.bonddesk.com/moodys.html

# CHECK
if check_dataset == 1:
    df_GBuniverse_GBnoncallable['spcurrent'].unique()
    df_BBuniverse_BBnoncallable['spcurrent'].unique()
    df_GBuniverse_GBcallable['spcurrent'].unique()
    df_GBuniverse_GBcallable['spcurrent'].unique()
    
    df_GBuniverse_GBnoncallable['moodyscurrent'].unique()
    df_BBuniverse_BBnoncallable['moodyscurrent'].unique()
    df_GBuniverse_GBcallable['moodyscurrent'].unique()
    df_GBuniverse_GBcallable['moodyscurrent'].unique()

# DIC: short ver. => dic_sp = {'AAA':1, 'AA+':2, 'AA':3, 'AA-':4, 'A+':5, 'A':6, 'A-':7, 'NR':22}
# NR indicates that a rating has not been assigned or is no longer assigned.

dic_sp = {'AAA':1, 'AA+':2, 'AA':3, 'AA-':4, 'A+':5, 'A':6, 'A-':7,
          'BBB+':8, 'BBB':9, 'BBB-':10, 'BB+':11, 'BB':12, 'BB-':13, 'B+':14, 'B':15, 'B-':16, 
          'CCC+':17, 'CCC':18, 'CCC-':19, 'CC':20, 'C':21, 
          'D':22, 'NR':22, 'WR':22, '--':22}


# DIC: short ver. => dic_moodys = {'AAA':1, 'AA1':2, 'AA2':3, 'AA3':4, 'A1':5, 'A2':6, 'A3':7, 'WR':22}
# NOTE: the treatmen of 'CA':20.5 should NOT matter anyway => majority of bonds are investment grades
# WR stands for “withdrawn rating.” Reasons for withdrawals include: debt maturity; calls, puts, conversions, etc.; business reasons (e.g. change in the size of a debt issue), or the issuer defaults

dic_moodys = {'AAA':1, 'AA1':2, 'AA2':3, 'AA3':4, 'A1':5, 'A2':6, 'A3':7, 
              'BAA1':8, 'BAA2':9, 'BAA3':10, 'BA1':11, 'BA2':12, 'BA3':13, 'B1':14, 'B2':15, 'B3':16, 
              'CAA1':17, 'CAA2':18, 'CAA3':19, 'CA':20.5, 
              'C':22, 'NR':22, 'WR':22, '--':22}


def ratingconvert(dataframe):
    
    dataframe['moodyscurrent_num'] = dataframe['moodyscurrent'].map(dic_moodys)
    dataframe['spcurrent_num'] = dataframe['spcurrent'].map(dic_sp)
    
    # NEW columns
    dataframe['aggrating'] = ( dataframe['moodyscurrent_num'] + dataframe['spcurrent_num'] ) / 2

    return dataframe

# NOTE: there is NO need to df = func(df)
ratingconvert(df_GBuniverse_GBnoncallable); ratingconvert(df_BBuniverse_BBnoncallable); ratingconvert(df_GBuniverse_GBcallable); ratingconvert(df_BBuniverse_BBcallable)

# *************

# Moodys, S&P nonexistent => DO NOT drop (LW, pp. 33–34)
'''
# xxx REDUCE F1: NA = NA match => drop
if drop_NANAratingsmatch == 1: 
    df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable[ (df_GBuniverse_GBnoncallable['moodyscurrent'] != '--') & (df_GBuniverse_GBnoncallable['spcurrent'] != '--') ]
    df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable[ (df_BBuniverse_BBnoncallable['moodyscurrent'] != '--') & (df_BBuniverse_BBnoncallable['spcurrent'] != '--') ]
    df_GBuniverse_GBcallable = df_GBuniverse_GBcallable[ (df_GBuniverse_GBcallable['moodyscurrent'] != '--') & (df_GBuniverse_GBcallable['spcurrent'] != '--') ]
    df_BBuniverse_BBcallable = df_BBuniverse_BBcallable[ (df_BBuniverse_BBcallable['moodyscurrent'] != '--') & (df_BBuniverse_BBcallable['spcurrent'] != '--') ]

'''
# *************
    
# REDUCE E: ULTIMATE MATCHING: repeat code in Sec 2, BUT integrate REDUCE D (only Moodys AND S&P; OR is unnecessary)
# => in retrospect, only a further mild reduction from REDUCE D (moved to the bottom of this script)!

# MATCHING: maturity AND coupon
if exactMat_or_withinoneyearMat == 1: # DO NOT EXACT MATCH!
    sys.exit()

# MATCHING: +/- one year maturity difference
elif exactMat_or_withinoneyearMat == 0: # obs

    # CONCATENATE: WEAK condition ALLOWS coupon difference for noncallable
    if noncallable_couponmatching == 0:        
        df_GBuniverse_GBnoncallable['concatenate_plus'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['aggrating'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] + 1).astype(str) #+ '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_GBuniverse_GBnoncallable['concatenate_zero'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['aggrating'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] + 0).astype(str) #+ '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_GBuniverse_GBnoncallable['concatenate_minu'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['aggrating'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] - 1).astype(str) #+ '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
    
        df_BBuniverse_BBnoncallable['concatenate_plus'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['aggrating'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] + 1).astype(str) #+ '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate_zero'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['aggrating'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] + 0).astype(str) #+ '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate_minu'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['aggrating'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] - 1).astype(str) #+ '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)

    # CONCATENATE: STRONG condition PROHIBITS coupon difference for noncallable
    elif noncallable_couponmatching == 1:        
        df_GBuniverse_GBnoncallable['concatenate_plus'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['aggrating'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] + 1).astype(str) + '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_GBuniverse_GBnoncallable['concatenate_zero'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['aggrating'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] + 0).astype(str) + '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
        df_GBuniverse_GBnoncallable['concatenate_minu'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['aggrating'].astype(str) + '_' + df_GBuniverse_GBnoncallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBnoncallable['maturity_year'] - 1).astype(str) + '_' + df_GBuniverse_GBnoncallable['coupon'].astype(str)
    
        df_BBuniverse_BBnoncallable['concatenate_plus'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['aggrating'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] + 1).astype(str) + '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate_zero'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['aggrating'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] + 0).astype(str) + '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)
        df_BBuniverse_BBnoncallable['concatenate_minu'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['aggrating'].astype(str) + '_' + df_BBuniverse_BBnoncallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBnoncallable['maturity_year'] - 1).astype(str) + '_' + df_BBuniverse_BBnoncallable['coupon'].astype(str)

    # CONCATENATE: STRONG condition PROHIBITS coupon difference for callable
    df_GBuniverse_GBcallable['concatenate_plus'] = df_GBuniverse_GBcallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBcallable['aggrating'].astype(str) + '_' + df_GBuniverse_GBcallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBcallable['maturity_year'] + 1).astype(str) + '_' + df_GBuniverse_GBcallable['coupon'].astype(str)
    df_GBuniverse_GBcallable['concatenate_zero'] = df_GBuniverse_GBcallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBcallable['aggrating'].astype(str) + '_' + df_GBuniverse_GBcallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBcallable['maturity_year'] + 0).astype(str) + '_' + df_GBuniverse_GBcallable['coupon'].astype(str)
    df_GBuniverse_GBcallable['concatenate_minu'] = df_GBuniverse_GBcallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBcallable['aggrating'].astype(str) + '_' + df_GBuniverse_GBcallable['dated_date'].astype(str) + '_' + (df_GBuniverse_GBcallable['maturity_year'] - 1).astype(str) + '_' + df_GBuniverse_GBcallable['coupon'].astype(str)

    df_BBuniverse_BBcallable['concatenate_plus'] = df_BBuniverse_BBcallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBcallable['aggrating'].astype(str) + '_' + df_BBuniverse_BBcallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBcallable['maturity_year'] + 1).astype(str) + '_' + df_BBuniverse_BBcallable['coupon'].astype(str)
    df_BBuniverse_BBcallable['concatenate_zero'] = df_BBuniverse_BBcallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBcallable['aggrating'].astype(str) + '_' + df_BBuniverse_BBcallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBcallable['maturity_year'] + 0).astype(str) + '_' + df_BBuniverse_BBcallable['coupon'].astype(str)
    df_BBuniverse_BBcallable['concatenate_minu'] = df_BBuniverse_BBcallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBcallable['aggrating'].astype(str) + '_' + df_BBuniverse_BBcallable['dated_date'].astype(str) + '_' + (df_BBuniverse_BBcallable['maturity_year'] - 1).astype(str) + '_' + df_BBuniverse_BBcallable['coupon'].astype(str)
            
    # REDUCE:
    df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable[ df_GBuniverse_GBnoncallable['concatenate_plus'].isin( df_BBuniverse_BBnoncallable['concatenate_zero'] ) |
                                                               df_GBuniverse_GBnoncallable['concatenate_zero'].isin( df_BBuniverse_BBnoncallable['concatenate_zero'] ) |
                                                               df_GBuniverse_GBnoncallable['concatenate_minu'].isin( df_BBuniverse_BBnoncallable['concatenate_zero'] ) ]
    # NECESSARY: the order in interchangeable btw GB and BB
    df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable[ df_BBuniverse_BBnoncallable['concatenate_plus'].isin( df_GBuniverse_GBnoncallable['concatenate_zero'] ) |
                                                               df_BBuniverse_BBnoncallable['concatenate_zero'].isin( df_GBuniverse_GBnoncallable['concatenate_zero'] ) |
                                                               df_BBuniverse_BBnoncallable['concatenate_minu'].isin( df_GBuniverse_GBnoncallable['concatenate_zero'] ) ]

    df_GBuniverse_GBcallable = df_GBuniverse_GBcallable[ df_GBuniverse_GBcallable['concatenate_plus'].isin( df_BBuniverse_BBcallable['concatenate_zero'] ) |
                                                               df_GBuniverse_GBcallable['concatenate_zero'].isin( df_BBuniverse_BBcallable['concatenate_zero'] ) |
                                                               df_GBuniverse_GBcallable['concatenate_minu'].isin( df_BBuniverse_BBcallable['concatenate_zero'] ) ]
    # NECESSARY: the order in interchangeable btw GB and BB
    df_BBuniverse_BBcallable = df_BBuniverse_BBcallable[ df_BBuniverse_BBcallable['concatenate_plus'].isin( df_GBuniverse_GBcallable['concatenate_zero'] ) |
                                                               df_BBuniverse_BBcallable['concatenate_zero'].isin( df_GBuniverse_GBcallable['concatenate_zero'] ) |
                                                               df_BBuniverse_BBcallable['concatenate_minu'].isin( df_GBuniverse_GBcallable['concatenate_zero'] ) ]

# ************

# NEW column
df_GBuniverse_GBnoncallable['municalltype'] = 'GBnoncallable'
df_BBuniverse_BBnoncallable['municalltype'] = 'BBnoncallable'
df_GBuniverse_GBcallable['municalltype'] = 'GBcallable'
df_BBuniverse_BBcallable['municalltype'] = 'BBcallable'
    
# CHECK
if export_dataset == 1: # BEFORE transform, and rating matching
    # EXPORT CHECK
    pd.concat( [df_GBuniverse_GBnoncallable, df_BBuniverse_BBnoncallable], axis=0 ).to_excel( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_matchingcheck_beforeSec6_Noncallable.xlsx"), index=False)
    pd.concat( [df_GBuniverse_GBcallable, df_BBuniverse_BBcallable], axis=0 ).to_excel( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_matchingcheck_beforeSec6_Callable.xlsx"), index=False)
        
#%%
    
    
#%% Section 5. Spatiotemporal data: Hazard / climate opinion / social capital
'''
# Cat event study => prominent cases 2018 heat wave, 2019 cold wave, 2019 COVID

# OUTPUT: df_climateopinion2016, df_climateopinion2018, df_States

# OLD: 2019 North American cold wave (Jan-Feb): https://www.google.com/search?q=January%7BFebruary+2019+North+American+cold+wave%2C&rlz=1C5CHFA_enDE843DE843&oq=January%7BFebruary+2019+North+American+cold+wave%2C&aqs=chrome..69i57.245j0j7&sourceid=chrome&ie=UTF-8
dic_coldwave2019 = {'CA':1, 'IL':1, 'IN':1, 'IA':1, 'KY':1, 'MI':1, 'MN':1, 'NY':1, 'ND':1, 'WA':1, 'WI':1}

'''
# State abbreviation: https://www.mapsofworld.com/usa/states/
dic_USCAcohortABC = {'CA':1, 'NY':1, 'WA':1, 'CT':1, 'RI':1, 'MA':1, 'VT':1, 'OR':1, 'HI':1, 'VA':1, 'MN':1, 'DE':1, }

# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #

# 2018 North American heat wave (July-Aug) 18 states heat advisories: https://en.wikipedia.org/wiki/2018_North_American_heat_wave 
dic_heatwave2018 = {'AZ':1, 'CA':1, 'CO':1, 'LA':1, 'NV':1, 'OK':1, 'TX':1, 'UT':1, 'MT':1}

# 2019 cold wave – diff from average, monthly: https://www.climate.gov/maps-data/data-snapshots/averagetempanom-monthly-cmb-2019-02-00?theme=Temperature
dic_coldwave2019 = {'WA':1, 'OR':1, 'CA':1, 'ID':1, 'NV':1, 'MT':1, 'WY':1, 'UT':1, 'AZ':1, 'CO':1, 'NM':1, 'ND':1, 'SD':1, 'NE':1, 'KS':1, 'OK':1, 'MN':1, 'IA':1, 'MO':1, 'WI':1, 'IL':1}

# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #

# [ Climate Opinion 2016 ]

# NEW: climate opinion
df_climateopinion2016 = pd.read_csv('/Users/YUMA/Desktop/3rdpaper/Yale/YCOM_2016_Data.01.csv') # , usecols=, delimiter=",
# EXPAND
df_climateopinion2016[['County','State']] = df_climateopinion2016['GeoName'].str.split(', ',expand=True,)

# NEW child: state level
df_climateopinion2016_state = df_climateopinion2016[ df_climateopinion2016['GeoType'] == 'State' ]

# NEW child: county level => National, State, Core-based statistical area (CBSA), cd113
df_climateopinion2016_county = df_climateopinion2016[ df_climateopinion2016['GeoType'] == 'County' ]
df_climateopinion2016_county.rename(columns={'GEOID':'geoid_int'}, inplace=True)

###############
# [ Climate Opinion 2018 => state level analysis only]

# NEW: climate opinion: https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python
df_climateopinion2018 = pd.read_csv('/Users/YUMA/Desktop/3rdpaper/Yale/YCOM_2018_Data.csv', engine='python') # , usecols=, delimiter=",
# EXPAND
df_climateopinion2018[['County','State']] = df_climateopinion2018['GeoName'].str.split(', ',expand=True,)

# +++++

# NEW child: state level
df_climateopinion2018_State = df_climateopinion2018[ df_climateopinion2018['GeoType'] == 'State' ]

# NEW
df_climateopinion2018_State['happeningUD'] = np.array( df_climateopinion2018_State['happening'] >= df_climateopinion2018_State['happening'].median() ).astype(int)
df_climateopinion2018_State['humanUD'] = np.array( df_climateopinion2018_State['human'] >= df_climateopinion2018_State['human'].median() ).astype(int)
df_climateopinion2018_State['worriedUD'] = np.array( df_climateopinion2018_State['worried'] >= df_climateopinion2018_State['worried'].median() ).astype(int)
df_climateopinion2018_State['harmUSUD'] = np.array( df_climateopinion2018_State['harmUS'] >= df_climateopinion2018_State['harmUS'].median() ).astype(int)
df_climateopinion2018_State['CO2limitsUD'] = np.array( df_climateopinion2018_State['CO2limits'] >= df_climateopinion2018_State['CO2limits'].median() ).astype(int)

# REDUCE: both cont and binary variables => use GeoName for df_climateopinion2018_State and ['County','State'] for df_climateopinion2018_County so that no double naming in Sec 6
df_climateopinion2018_State = df_climateopinion2018_State[['GeoName','happening','happeningUD','human','humanUD','worried','worriedUD','harmUS','harmUSUD','CO2limits','CO2limitsUD']]

# CORR
df_climateopinion2018_State['human'].corr(df_climateopinion2018_State['harmUS'])
df_climateopinion2018_State['human'].corr(df_climateopinion2018_State['worried'])
df_climateopinion2018_State['human'].corr(df_climateopinion2018_State['happening'])
df_climateopinion2018_State['human'].corr(df_climateopinion2018_State['CO2limits'])
df_climateopinion2018_State['CO2limits'].corr(df_climateopinion2018_State['harmUS'])
df_climateopinion2018_State['CO2limits'].corr(df_climateopinion2018_State['worried'])
df_climateopinion2018_State['CO2limits'].corr(df_climateopinion2018_State['happening'])

# +++++

# NEW child: county level
df_climateopinion2018_County = df_climateopinion2018[ df_climateopinion2018['GeoType'] == 'County' ]

# NEW
df_climateopinion2018_County['happeningUD_county'] = np.array( df_climateopinion2018_County['happening'] >= df_climateopinion2018_County['happening'].median() ).astype(int)
df_climateopinion2018_County['humanUD_county'] = np.array( df_climateopinion2018_County['human'] >= df_climateopinion2018_County['human'].median() ).astype(int)
df_climateopinion2018_County['worriedUD_county'] = np.array( df_climateopinion2018_County['worried'] >= df_climateopinion2018_County['worried'].median() ).astype(int)
df_climateopinion2018_County['harmUSUD_county'] = np.array( df_climateopinion2018_County['harmUS'] >= df_climateopinion2018_County['harmUS'].median() ).astype(int)
df_climateopinion2018_County['CO2limitsUD_county'] = np.array( df_climateopinion2018_County['CO2limits'] >= df_climateopinion2018_County['CO2limits'].median() ).astype(int)

# RENAME
df_climateopinion2018_County.rename(columns={'happening':'happening_county', 'human':'human_county', 'worried':'worried_county', 'harmUS':'harmUS_county', 'CO2limits':'CO2limits_county'}, inplace=True)

# REDUCE: both cont and binary variables => use GeoName for df_climateopinion2018_State and ['County','State'] for df_climateopinion2018_County so that no double naming in Sec 6
df_climateopinion2018_County = df_climateopinion2018_County[['County','State','happening_county','happeningUD_county','human_county','humanUD_county','worried_county','worriedUD_county','harmUS_county','harmUSUD_county','CO2limits_county','CO2limitsUD_county']]

# CORR
df_climateopinion2018_County['human_county'].corr(df_climateopinion2018_County['harmUS_county'])
df_climateopinion2018_County['human_county'].corr(df_climateopinion2018_County['worried_county'])
df_climateopinion2018_County['human_county'].corr(df_climateopinion2018_County['happening_county'])
df_climateopinion2018_County['human_county'].corr(df_climateopinion2018_County['CO2limits_county'])
df_climateopinion2018_County['CO2limits_county'].corr(df_climateopinion2018_County['worried_county'])
df_climateopinion2018_County['CO2limits_county'].corr(df_climateopinion2018_County['harmUS_county'])
df_climateopinion2018_County['CO2limits_county'].corr(df_climateopinion2018_County['happening_county'])

# WRITE: within-state variation
df_climateopinion2018_State[['happening','GeoName']]
df_climateopinion2018_State[['happening','GeoName']].describe()
df_climateopinion2018_County[ df_climateopinion2018_County['State'] == 'Alabama']['happening_county'].describe()
df_climateopinion2018_County[ df_climateopinion2018_County['State'] == 'California']['happening_county'].describe()
df_climateopinion2018_County[ df_climateopinion2018_County['State'] == 'Massachusetts']['happening_county'].describe()
df_climateopinion2018_County[ df_climateopinion2018_County['State'] == 'Tennessee']['happening_county'].describe()
df_climateopinion2018_County[ df_climateopinion2018_County['State'] == 'Texas']['happening_county'].describe()

# CONTINUE: right below after dic_states defined

# ************* ************* ************* ************* #

# CHANGE: 'DC':'Washington DC' => 'DC':'District of Columbia'
dic_states = {'AL':'Alabama', 'AK':'Alaska', 'AZ':'Arizona', 'AR':'Arkansas', 'CA':'California', 'CO':'Colorado',
'CT':'Connecticut', 'DC':'District of Columbia', 'DE':'Delaware', 'FL':'Florida', 'GA':'Georgia',
'HI':'Hawaii', 'ID':'Idaho', 'IL':'Illinois', 'IN':'Indiana', 'IA':'Iowa',
'KS':'Kansas', 'KY':'Kentucky', 'LA':'Louisiana', 'ME':'Maine', 'MD':'Maryland',
'MA':'Massachusetts', 'MI':'Michigan', 'MN':'Minnesota', 'MS':'Mississippi',
'MO':'Missouri', 'MT':'Montana', 'NE':'Nebraska', 'NV':'Nevada', 'NH':'New Hampshire',
'NJ':'New Jersey', 'NM':'New Mexico', 'NY':'New York', 'NC':'North Carolina',
'ND':'North Dakota', 'OH':'Ohio', 'OK':'Oklahoma', 'OR':'Oregon', 'PA':'Pennsylvania',
'RI':'Rhode Island', 'SC':'South Carolina', 'SD':'South Dakota', 'TN':'Tennessee',
'TX':'Texas', 'UT':'Utah', 'VT':'Vermont', 'VA':'Virginia', 'WA':'Washington', 'WV':'West Virginia',
'WI':'Wisconsin', 'WY':'Wyoming'}

df_States = pd.DataFrame([dic_states]).T.reset_index().rename(columns={'index':'State', 0:'GeoName'})

# MERGE: state level
df_climateopinion2018_State = df_climateopinion2018_State.merge( df_States, on=['GeoName'] )

# TRANSFORM
dic_states_reversed = {value : key for (key, value) in dic_states.items()}

# NEW: merging will take place in Sec 6 with Key: @*$#
df_climateopinion2018_County['County, State2'] = df_climateopinion2018_County['County'].str.replace(' County','') + ', ' + df_climateopinion2018_County['State'].map(dic_states_reversed)

###############

# NEW columns
df_States['heatwave2018'] = df_States['State'].map(dic_heatwave2018).replace(np.nan, 0).astype(int)
df_States['coldwave2019'] = df_States['State'].map(dic_coldwave2019).replace(np.nan, 0).astype(int)
df_States['USCAcohortABC'] = df_States['State'].map(dic_USCAcohortABC).replace(np.nan, 0).astype(int)

# NEW: social capital
#df_socialcapital_state = pd.read_excel('/Users/YUMA/Desktop/3rdpaper/Datasets/social-capital-project-social-capital-index-data.xlsx', sheet_name='State Index', skiprows=2) 
#df_socialcapital_county = pd.read_excel('/Users/YUMA/Desktop/3rdpaper/Datasets/social-capital-project-social-capital-index-data.xlsx', sheet_name='County Index', skiprows=2) 

#%%
   

#%% Section 6. MSRB transactions, merging Sec 5 data, and statistics
'''
# INPUT: MSRB transactional data of [df_BBuniverse, df_GBuniverse]
# INPUT2: HHI ownership? => but no differential according to LW

# Number of trades over the quarter after disaster hit => since the no. of munis are different btw GB and BB, need to .groupby('cusip') and examine the dist
# underwriters discount is only for primary market and spread is similar to yield

 => when_issued_indicator is for Primary/Secondary market distinction

See ZEROCOUPONBOND_59261APJ0.xlsx for yield computation
Par Value: https://www.investopedia.com/terms/p/parvalue.asp

Number of Trades: Calculated as the total number of trades over the quarter (90-days) after disaster.
Turnover: Calculated as the total sum of par value trades over the quarter (90-days) after disaster divided by the total issuance amount.

'''
# BEFORE trade_date_int, current_yield
if export_dataset == 1:    
    df_GBuniverse_GBnoncallable.to_csv( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_GBuniverse_GBnoncallable_MMerged.csv"), index=False)    
    df_BBuniverse_BBnoncallable.to_csv( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_BBuniverse_BBnoncallable_MMerged.csv"), index=False)    
    df_GBuniverse_GBcallable.to_csv( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_GBuniverse_GBcallable_MMerged.csv"), index=False)    
    df_BBuniverse_BBcallable.to_csv( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_BBuniverse_BBcallable_MMerged.csv"), index=False)    
    
    df_Transaction_GBnoncallable.to_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_Transaction_GBnoncallable.csv"), index=False)    
    df_Transaction_BBnoncallable.to_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_Transaction_BBnoncallable.csv"), index=False)    
    df_Transaction_GBcallable.to_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_Transaction_GBcallable.csv"), index=False)    
    df_Transaction_BBcallable.to_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_Transaction_BBcallable.csv"), index=False)    

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@ START FROM SECTION 5 AND THEN HERE: SECTION 5 WILL AFFECT SECTION 6 @@@@@@@@@@@    
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@ df_GBuniverse_GBnoncallable is needed not only due to convenience but LW REPLICATION @@
    
# BEFORE trade_date_int, current_yield    
if import_dataset == 1: 
    df_GBuniverse_GBnoncallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_GBuniverse_GBnoncallable_MMerged.csv'))
    df_BBuniverse_BBnoncallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_BBuniverse_BBnoncallable_MMerged.csv'))
    df_GBuniverse_GBcallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_GBuniverse_GBcallable_MMerged.csv'))
    df_BBuniverse_BBcallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_BBuniverse_BBcallable_MMerged.csv'))

    df_Transaction_GBnoncallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_Transaction_GBnoncallable.csv'))
    df_Transaction_BBnoncallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_Transaction_BBnoncallable.csv'))
    df_Transaction_GBcallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_Transaction_GBcallable.csv'))
    df_Transaction_BBcallable = pd.read_csv(os.path.join(directory_output, dic_noncallable_couponmatching_value, 'df_Transaction_BBcallable.csv'))

# *************** *************** *************** *************** *************** *************** *************** #
    
#db = wrds.Connection(); 'ymws21'; 'Apfel3159###';
# LOOP1: 
for dataframename3 in ['df_GBuniverse_GBnoncallable','df_BBuniverse_BBnoncallable','df_GBuniverse_GBcallable','df_BBuniverse_BBcallable']: #
    
    df_Basic_Transaction = pd.DataFrame(); count = 0;
    
    # LOOP2: CUSIP
    for cusip in globals()[dataframename3]['cusip']:
        print(dataframename3 + " No." + str(count) + " – CUSIP is: " + cusip + "\n")
        
        # ,trade_type_indicator, weighted_price_indicator, settlement_date, assumed_settlement_date
        appendedtransactioninfo = db.raw_sql("SELECT cusip, rtrs_control_number, security_description, dated_date, coupon, maturity_date, yield, dollar_price, offer_price_takedown_indicator, trade_date, when_issued_indicator, par_traded \
                   FROM msrb.msrb \
                   WHERE cusip \
                   IN ('" + cusip + "') \
                   ")
            
        df_Basic_Transaction = df_Basic_Transaction.append(appendedtransactioninfo)
        
        # LOOP2 END
        count += 1
        
    # LOOP1 END
    globals()[ 'df_Transaction_' + dataframename3.split('_')[2] ] = df_Basic_Transaction.copy()

if alarm_on == 1:
    os.system('say "your third computation is finished"')

# *************** *************** *************** ***************** *************** *************** *************** #       
# *************** *************** *************** [START FROM HERE] *************** *************** *************** #   

# [Step 0.A: Fundamental]    
# NEW columns: DATE
df_Transaction_GBnoncallable['trade_date_int'] = ( df_Transaction_GBnoncallable['trade_date'].astype(str).str[:4] + df_Transaction_GBnoncallable['trade_date'].astype(str).str[5:7] + df_Transaction_GBnoncallable['trade_date'].astype(str).str[8:10] ).astype(int)
df_Transaction_BBnoncallable['trade_date_int'] = ( df_Transaction_BBnoncallable['trade_date'].astype(str).str[:4] + df_Transaction_BBnoncallable['trade_date'].astype(str).str[5:7] + df_Transaction_BBnoncallable['trade_date'].astype(str).str[8:10] ).astype(int)
df_Transaction_GBcallable['trade_date_int'] = ( df_Transaction_GBcallable['trade_date'].astype(str).str[:4] + df_Transaction_GBcallable['trade_date'].astype(str).str[5:7] + df_Transaction_GBcallable['trade_date'].astype(str).str[8:10] ).astype(int)
df_Transaction_BBcallable['trade_date_int'] = ( df_Transaction_BBcallable['trade_date'].astype(str).str[:4] + df_Transaction_BBcallable['trade_date'].astype(str).str[5:7] + df_Transaction_BBcallable['trade_date'].astype(str).str[8:10] ).astype(int)

# NEW columns: current_yield
df_Transaction_GBnoncallable['current_yield'] = df_Transaction_GBnoncallable['coupon'] * 100 / df_Transaction_GBnoncallable['dollar_price']
df_Transaction_BBnoncallable['current_yield'] = df_Transaction_BBnoncallable['coupon'] * 100 / df_Transaction_BBnoncallable['dollar_price']
df_Transaction_GBcallable['current_yield'] = df_Transaction_GBcallable['coupon'] * 100 / df_Transaction_GBcallable['dollar_price']
df_Transaction_BBcallable['current_yield'] = df_Transaction_BBcallable['coupon'] * 100 / df_Transaction_BBcallable['dollar_price']

# MERGE 0: basis + transaction
df_GBuniverse_GBnoncallable_wTrans = pd.merge( df_GBuniverse_GBnoncallable.drop(columns=['dated_date','maturity_date','coupon']), df_Transaction_GBnoncallable.rename(columns={'yield':'ytm'}), on=['cusip'] )
df_BBuniverse_BBnoncallable_wTrans = pd.merge( df_BBuniverse_BBnoncallable.drop(columns=['dated_date','maturity_date','coupon']), df_Transaction_BBnoncallable.rename(columns={'yield':'ytm'}), on=['cusip'] )
df_GBuniverse_GBcallable_wTrans = pd.merge( df_GBuniverse_GBcallable.drop(columns=['dated_date','maturity_date','coupon']), df_Transaction_GBcallable.rename(columns={'yield':'ytm'}), on=['cusip'] )
df_BBuniverse_BBcallable_wTrans = pd.merge( df_BBuniverse_BBcallable.drop(columns=['dated_date','maturity_date','coupon']), df_Transaction_BBcallable.rename(columns={'yield':'ytm'}), on=['cusip'] )

# CHECK: row num vs rtrs_control_number
if check_dataset == 1: 
    len( df_GBuniverse_GBnoncallable_wTrans ) == len( df_GBuniverse_GBnoncallable_wTrans['rtrs_control_number'].unique() )
    len( df_BBuniverse_BBnoncallable_wTrans ) == len( df_BBuniverse_BBnoncallable_wTrans['rtrs_control_number'].unique() )
    len( df_GBuniverse_GBcallable_wTrans ) == len( df_GBuniverse_GBcallable_wTrans['rtrs_control_number'].unique() )
    len( df_BBuniverse_BBcallable_wTrans ) == len( df_BBuniverse_BBcallable_wTrans['rtrs_control_number'].unique() )
    
    # EXPORT
    df_BBuniverse_BBnoncallable_wTrans['duplicated'] = df_BBuniverse_BBnoncallable_wTrans['rtrs_control_number'].duplicated()    
    df_BBuniverse_BBnoncallable_wTrans.to_excel( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_BBuniverse_BBnoncallable_wTrans_TradeNumTEST.xlsx"), index=False)
    
# DUPLICATE: accodring to above CHECK => all fine
df_GBuniverse_GBnoncallable_wTrans = df_GBuniverse_GBnoncallable_wTrans.drop_duplicates()
df_BBuniverse_BBnoncallable_wTrans = df_BBuniverse_BBnoncallable_wTrans.drop_duplicates()
df_GBuniverse_GBcallable_wTrans = df_GBuniverse_GBcallable_wTrans.drop_duplicates()
df_BBuniverse_BBcallable_wTrans = df_BBuniverse_BBcallable_wTrans.drop_duplicates()

# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #

# [Step 0.B: Merging climate opinion, hazards, USCA]

# @@@ CHECK: confirmed state (from fidelity) >> State (from BBG US county of issuance); "DC" <=> "11" @@@
if check_dataset == 1: 
    a1 = df_GBuniverse_GBnoncallable_wTrans[['state','State']][ df_GBuniverse_GBnoncallable_wTrans['state'] != df_GBuniverse_GBnoncallable_wTrans['State'] ]
    a2 = df_BBuniverse_BBnoncallable_wTrans[['state','State']][ df_BBuniverse_BBnoncallable_wTrans['state'] != df_BBuniverse_BBnoncallable_wTrans['State'] ]
    a3 = df_GBuniverse_GBcallable_wTrans[['state','State']][ df_GBuniverse_GBcallable_wTrans['state'] != df_GBuniverse_GBcallable_wTrans['State'] ]
    a4 = df_BBuniverse_BBcallable_wTrans[['state','State']][ df_BBuniverse_BBcallable_wTrans['state'] != df_BBuniverse_BBcallable_wTrans['State'] ]

# MERGE 1: climate opinion; avoid losing small percentage of obs due to N/A in U.S. county info => state >> State
df_GBuniverse_GBnoncallable_wTrans = pd.merge( df_GBuniverse_GBnoncallable_wTrans, df_climateopinion2018_State.rename(columns={'State':'state'}), on='state' )
df_BBuniverse_BBnoncallable_wTrans = pd.merge( df_BBuniverse_BBnoncallable_wTrans, df_climateopinion2018_State.rename(columns={'State':'state'}), on='state' )
df_GBuniverse_GBcallable_wTrans = pd.merge( df_GBuniverse_GBcallable_wTrans, df_climateopinion2018_State.rename(columns={'State':'state'}), on='state' )
df_BBuniverse_BBcallable_wTrans = pd.merge( df_BBuniverse_BBcallable_wTrans, df_climateopinion2018_State.rename(columns={'State':'state'}), on='state' )

'''
@@@@@@ COUNTY LEVEL OPINION df_climateopinion2018_County => MERGE HERE INSTEAD OF RIGHT BEFORE THE ANALYSIS @@@@@@@
Key: @*$#

'''
# MERGE 1+: how='left': climate opinion; obs loss not substantial at all
df_GBuniverse_GBnoncallable_wTrans = pd.merge( df_GBuniverse_GBnoncallable_wTrans, df_climateopinion2018_County.rename(columns={'County, State2':'U.S. County Of Issuance'}), on='U.S. County Of Issuance', how='left' )
df_BBuniverse_BBnoncallable_wTrans = pd.merge( df_BBuniverse_BBnoncallable_wTrans, df_climateopinion2018_County.rename(columns={'County, State2':'U.S. County Of Issuance'}), on='U.S. County Of Issuance', how='left' )
df_GBuniverse_GBcallable_wTrans = pd.merge( df_GBuniverse_GBcallable_wTrans, df_climateopinion2018_County.rename(columns={'County, State2':'U.S. County Of Issuance'}), on='U.S. County Of Issuance', how='left' )
df_BBuniverse_BBcallable_wTrans = pd.merge( df_BBuniverse_BBcallable_wTrans, df_climateopinion2018_County.rename(columns={'County, State2':'U.S. County Of Issuance'}), on='U.S. County Of Issuance', how='left' )

##########

# MERGE 2: df_States, cat events => 'state' >> 'State'
df_GBuniverse_GBnoncallable_wTrans = pd.merge( df_GBuniverse_GBnoncallable_wTrans, df_States.drop(columns='GeoName').rename(columns={'State':'state'}), on='state' )
df_BBuniverse_BBnoncallable_wTrans = pd.merge( df_BBuniverse_BBnoncallable_wTrans, df_States.drop(columns='GeoName').rename(columns={'State':'state'}), on='state' )
df_GBuniverse_GBcallable_wTrans = pd.merge( df_GBuniverse_GBcallable_wTrans, df_States.drop(columns='GeoName').rename(columns={'State':'state'}), on='state' )
df_BBuniverse_BBcallable_wTrans = pd.merge( df_BBuniverse_BBcallable_wTrans, df_States.drop(columns='GeoName').rename(columns={'State':'state'}), on='state' )

# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== #

# [STEP 1: AGGREGATING DATAFRAME]

# NEW column: MOVED to end of Sec 5
#df_GBuniverse_GBnoncallable_wTrans['municalltype'] = 'GBnoncallable'
#df_BBuniverse_BBnoncallable_wTrans['municalltype'] = 'BBnoncallable'
#df_GBuniverse_GBcallable_wTrans['municalltype'] = 'GBcallable'
#df_BBuniverse_BBcallable_wTrans['municalltype'] = 'BBcallable'

# NEW CONCAT
df_GBuniverse_Comb = pd.concat( [df_GBuniverse_GBnoncallable, df_GBuniverse_GBcallable], axis=0 )
df_BBuniverse_Comb = pd.concat( [df_BBuniverse_BBnoncallable, df_BBuniverse_BBcallable], axis=0 )

# NEW CONCAT
df_GBuniverse_Comb_wTrans = pd.concat( [df_GBuniverse_GBnoncallable_wTrans, df_GBuniverse_GBcallable_wTrans], axis=0 )
df_BBuniverse_Comb_wTrans = pd.concat( [df_BBuniverse_BBnoncallable_wTrans, df_BBuniverse_BBcallable_wTrans], axis=0 )

# NEW CONCAT
df_NonCalluniverse_Comb_wTrans = pd.concat( [df_GBuniverse_GBnoncallable_wTrans, df_BBuniverse_BBnoncallable_wTrans], axis=0 )
df_Calluniverse_Comb_wTrans = pd.concat( [df_GBuniverse_GBcallable_wTrans, df_BBuniverse_BBcallable_wTrans], axis=0 )

# *********

# NEW column
df_GBuniverse_Comb_wTrans['munitype'] = 'GB'
df_BBuniverse_Comb_wTrans['munitype'] = 'BB'

# NEW CONCAT
df_MUNIuniverse_wTrans = pd.concat( [df_GBuniverse_Comb_wTrans, df_BBuniverse_Comb_wTrans], axis=0 )

# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== #

# [STEP 2: DROPPING OUTLIERS]
'''
[Pros and cons about dropping negative-zero ytm]
 pros => median-mean gap of ytm closes; can be justifiable by looking at dollar_price!
 cons => need to rematch bonds ? => NO, only a portion of transactions are likely ytm=0 within a bond; dropping is somewhat arbitrary given that theoretically possible and mentioned by MSRB_variables_09252018version.pdf
 
'''
# REDUCE
drop_ytm_positiveonly = 1; # DO NOT CHANGE
if drop_ytm_positiveonly == 1:
    df_GBuniverse_GBnoncallable_wTrans = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['ytm'] > 0 ]
    df_BBuniverse_BBnoncallable_wTrans = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['ytm'] > 0 ]
    df_GBuniverse_GBcallable_wTrans = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['ytm'] > 0 ]
    df_BBuniverse_BBcallable_wTrans = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['ytm'] > 0 ]

    df_GBuniverse_Comb_wTrans = df_GBuniverse_Comb_wTrans[ df_GBuniverse_Comb_wTrans['ytm'] > 0 ]
    df_BBuniverse_Comb_wTrans = df_BBuniverse_Comb_wTrans[ df_BBuniverse_Comb_wTrans['ytm'] > 0 ]    
    
    df_NonCalluniverse_Comb_wTrans = df_NonCalluniverse_Comb_wTrans[ df_NonCalluniverse_Comb_wTrans['ytm'] > 0 ]
    df_Calluniverse_Comb_wTrans = df_Calluniverse_Comb_wTrans[ df_Calluniverse_Comb_wTrans['ytm'] > 0 ]    
    
    df_MUNIuniverse_wTrans = df_MUNIuniverse_wTrans[ df_MUNIuniverse_wTrans['ytm'] > 0 ]        
    
# CHECK: ytm <= 0 is theoretically possible but here NOT justifiable due to dollar_price: https://www.investopedia.com/ask/answers/062315/what-does-negative-bond-yield-mean.asp#:~:text=Since%20the%20YTM%20calculation%20incorporates,sufficiently%20outweigh%20the%20initial%20investment.
# just because one bond has a transaction of ymt=0, it does not mean all transactions are zero ytm
if check_dataset == 1:
    # EXPORT
    df_MUNIuniverse_wTrans[['cusip','dated_date','trade_date','ytm','dollar_price','coupon','munitype','municalltype']].to_excel( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_ytmcheck_MUNIuniverse_wTrans.xlsx"), index=False)

# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== #

# [STEP 3: STATS & LW REPLICATION]

# TRANSFORM: prep for LW replication (i.e., sample upto July 2018)
# ERROR: generated when directly DL from MSRB, but avoided when importing from csv in previous sections
df_GBuniverse_GBnoncallable['dated_date_int'] = df_GBuniverse_GBnoncallable['dated_date'].str.replace("-","").astype(int)    
df_BBuniverse_BBnoncallable['dated_date_int'] = df_BBuniverse_BBnoncallable['dated_date'].str.replace("-","").astype(int)    
df_GBuniverse_GBcallable['dated_date_int'] = df_GBuniverse_GBcallable['dated_date'].str.replace("-","").astype(int)    
df_BBuniverse_BBcallable['dated_date_int'] = df_BBuniverse_BBcallable['dated_date'].str.replace("-","").astype(int)    

df_GBuniverse_Comb_wTrans['dated_date_int'] = df_GBuniverse_Comb_wTrans['dated_date'].str.replace("-","").astype(int)    
df_BBuniverse_Comb_wTrans['dated_date_int'] = df_BBuniverse_Comb_wTrans['dated_date'].str.replace("-","").astype(int)    

df_MUNIuniverse_wTrans['dated_date_int'] = df_MUNIuniverse_wTrans['dated_date'].str.replace("-","").astype(int)    

def sumstats(dataframe):
    print('**MEAN** values are:')
    print(dataframe.mean())
    print('\n')
    
    print('**MEDIAN** values are:')
    print(dataframe.median())
    return

def sumstats_Withpercentile_beforeLaTeX(dataframe,columnname,roundnum):
    
    df_beforeLaTeX = pd.DataFrame( data = { 'mean': [dataframe[columnname].mean()], 'std': [dataframe[columnname].std()], 'p1': [dataframe[columnname].quantile(0.01)], 'p25': [dataframe[columnname].quantile(0.25)], 'p50': [dataframe[columnname].quantile(0.5)], 'p75': [dataframe[columnname].quantile(0.75)], 'p99': [dataframe[columnname].quantile(0.99)], 'Obs': [len(dataframe[columnname].dropna())] } )
    
    return df_beforeLaTeX.round(roundnum) #.to_latex()

# STATS: static level (e.g., num of unique issuers)
len( df_MUNIuniverse_wTrans['cusip6'].unique() )
len( ( df_MUNIuniverse_wTrans['cusip6'] + '_' + df_MUNIuniverse_wTrans['dated_date'].astype(str) ).unique() ) # deal???

# STATS: sample period => starts from Sep 2014 although the restriction is after June 2013
sexam(df_MUNIuniverse_wTrans['dated_date_int'])

# STATS
sumstats( df_GBuniverse_GBnoncallable[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )
sumstats( df_BBuniverse_BBnoncallable[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )
sumstats( df_GBuniverse_GBcallable[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )
sumstats( df_BBuniverse_BBcallable[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )


# LATEX: key LLL
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBnoncallable,'originalissueamt',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBnoncallable,'coupon',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBnoncallable,'aggrating',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBnoncallable,'originalissueamt',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBnoncallable,'coupon',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBnoncallable,'aggrating',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBcallable,'originalissueamt',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBcallable,'coupon',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBcallable,'aggrating',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBcallable,'originalissueamt',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBcallable,'coupon',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBcallable,'aggrating',2)], axis=0 ).to_latex(index=False)

pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb,'originalissueamt',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb,'coupon',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb,'aggrating',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb,'originalissueamt',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb,'coupon',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb,'aggrating',2)], axis=0 ).to_latex(index=False)


# NEW child: LW REPLICATE sample up to July 2018 (p. 11)
df_GBuniverse_GBnoncallable_LW = df_GBuniverse_GBnoncallable[ df_GBuniverse_GBnoncallable['dated_date_int'] < 20180800]
df_BBuniverse_BBnoncallable_LW = df_BBuniverse_BBnoncallable[ df_BBuniverse_BBnoncallable['dated_date_int'] < 20180800]
df_GBuniverse_GBcallable_LW = df_GBuniverse_GBcallable[ df_GBuniverse_GBcallable['dated_date_int'] < 20180800]
df_BBuniverse_BBcallable_LW = df_BBuniverse_BBcallable[ df_BBuniverse_BBcallable['dated_date_int'] < 20180800]

# STATS
sumstats( df_GBuniverse_GBnoncallable_LW[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )
sumstats( df_BBuniverse_BBnoncallable_LW[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )
sumstats( df_GBuniverse_GBcallable_LW[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )
sumstats( df_BBuniverse_BBcallable_LW[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )


# NEW CONCAT
df_GBuniverse_Comb_LW = pd.concat( [df_GBuniverse_GBnoncallable_LW, df_GBuniverse_GBcallable_LW], axis=0 )
df_BBuniverse_Comb_LW = pd.concat( [df_BBuniverse_BBnoncallable_LW, df_BBuniverse_BBcallable_LW], axis=0 )

# STATS
df_GBuniverse_Comb_LW.shape
df_BBuniverse_Comb_LW.shape
sumstats( df_GBuniverse_Comb_LW[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )
sumstats( df_BBuniverse_Comb_LW[['coupon','originalissueamt','moodyscurrent_num','spcurrent_num','aggrating']] )

# LATEX: key LLL
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb_LW,'originalissueamt',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb_LW,'coupon',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb_LW,'aggrating',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb_LW,'originalissueamt',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb_LW,'coupon',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb_LW,'aggrating',2)], axis=0 ).to_latex(index=False)

# **********

# STATS: transaction level
sumstats( df_GBuniverse_GBnoncallable_wTrans[['ytm','dollar_price']] )
sumstats( df_BBuniverse_BBnoncallable_wTrans[['ytm','dollar_price']] )
sumstats( df_GBuniverse_GBcallable_wTrans[['ytm','dollar_price']] )
sumstats( df_BBuniverse_BBcallable_wTrans[['ytm','dollar_price']] )

sumstats( df_GBuniverse_Comb_wTrans[['ytm','dollar_price']] )
sumstats( df_BBuniverse_Comb_wTrans[['ytm','dollar_price']] )

sumstats( df_GBuniverse_Comb_wTrans[['ytm','dollar_price']][ df_GBuniverse_Comb_wTrans['offer_price_takedown_indicator'] == 'Y' ] )
sumstats( df_BBuniverse_Comb_wTrans[['ytm','dollar_price']][ df_BBuniverse_Comb_wTrans['offer_price_takedown_indicator'] == 'Y' ] )

# LATEX: key LLL (primary + secondary market)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBnoncallable_wTrans,'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBnoncallable_wTrans,'dollar_price',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBnoncallable_wTrans,'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBnoncallable_wTrans,'dollar_price',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBcallable_wTrans,'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_GBcallable_wTrans,'dollar_price',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBcallable_wTrans,'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_BBcallable_wTrans,'dollar_price',2)], axis=0 ).to_latex(index=False)

pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb_wTrans,'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb_wTrans,'dollar_price',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb_wTrans,'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb_wTrans,'dollar_price',2)], axis=0 ).to_latex(index=False)

# ****

# NEW child: LW replicate
df_GBuniverse_Comb_wTrans_LW = df_GBuniverse_Comb_wTrans[ df_GBuniverse_Comb_wTrans['dated_date_int'] < 20180800]
df_BBuniverse_Comb_wTrans_LW = df_BBuniverse_Comb_wTrans[ df_BBuniverse_Comb_wTrans['dated_date_int'] < 20180800]

# LATEX: key LLL (primary only)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb_wTrans[ df_GBuniverse_Comb_wTrans['offer_price_takedown_indicator'] == 'Y' ],'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb_wTrans[ df_GBuniverse_Comb_wTrans['offer_price_takedown_indicator'] == 'Y' ],'dollar_price',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb_wTrans[ df_BBuniverse_Comb_wTrans['offer_price_takedown_indicator'] == 'Y' ],'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb_wTrans[ df_BBuniverse_Comb_wTrans['offer_price_takedown_indicator'] == 'Y' ],'dollar_price',2)], axis=0 ).to_latex(index=False)

pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb_wTrans_LW[ df_GBuniverse_Comb_wTrans_LW['offer_price_takedown_indicator'] == 'Y' ],'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_GBuniverse_Comb_wTrans_LW[ df_GBuniverse_Comb_wTrans_LW['offer_price_takedown_indicator'] == 'Y' ],'dollar_price',2)], axis=0 ).to_latex(index=False)
pd.concat( [sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb_wTrans_LW[ df_BBuniverse_Comb_wTrans_LW['offer_price_takedown_indicator'] == 'Y' ],'ytm',2), sumstats_Withpercentile_beforeLaTeX(df_BBuniverse_Comb_wTrans_LW[ df_BBuniverse_Comb_wTrans_LW['offer_price_takedown_indicator'] == 'Y' ],'dollar_price',2)], axis=0 ).to_latex(index=False)

# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== #

# COPY: First time only
df_GBuniverse_GBnoncallable_wTrans_original = df_GBuniverse_GBnoncallable_wTrans.copy()
df_BBuniverse_BBnoncallable_wTrans_original = df_BBuniverse_BBnoncallable_wTrans.copy()
df_GBuniverse_GBcallable_wTrans_original = df_GBuniverse_GBcallable_wTrans.copy()
df_BBuniverse_BBcallable_wTrans_original = df_BBuniverse_BBcallable_wTrans.copy()

#df_GBuniverse_Comb_wTrans_original = df_GBuniverse_Comb_wTrans.copy()
#df_BBuniverse_Comb_wTrans_original = df_BBuniverse_Comb_wTrans.copy()
#df_NonCalluniverse_Comb_wTrans_original = df_NonCalluniverse_Comb_wTrans.copy()
#df_Calluniverse_Comb_wTrans_original = df_Calluniverse_Comb_wTrans.copy()
#df_MUNIuniverse_wTrans_original = df_MUNIuniverse_wTrans.copy()

#%%


#%% Section 7. Univariate analysis

# DIRECTLY STARTING FROM ANALYSIS SECTION
if export_dataset == 1:    
    df_GBuniverse_GBnoncallable_wTrans.to_csv(os.path.join(directory_output, "AnalysisSectionDirect", dic_noncallable_couponmatching_value, "df_GBuniverse_GBnoncallable_wTrans.csv"), index=False)    
    df_BBuniverse_BBnoncallable_wTrans.to_csv(os.path.join(directory_output, "AnalysisSectionDirect", dic_noncallable_couponmatching_value, "df_BBuniverse_BBnoncallable_wTrans.csv"), index=False)    
    df_GBuniverse_GBcallable_wTrans.to_csv(os.path.join(directory_output, "AnalysisSectionDirect", dic_noncallable_couponmatching_value, "df_GBuniverse_GBcallable_wTrans.csv"), index=False)    
    df_BBuniverse_BBcallable_wTrans.to_csv(os.path.join(directory_output, "AnalysisSectionDirect", dic_noncallable_couponmatching_value, "df_BBuniverse_BBcallable_wTrans.csv"), index=False)    
    
# DIRECTLY STARTING FROM ANALYSIS SECTION   
if import_dataset == 1: 
    df_GBuniverse_GBnoncallable_wTrans = pd.read_csv(os.path.join(directory_output, "AnalysisSectionDirect", dic_noncallable_couponmatching_value, "df_GBuniverse_GBnoncallable_wTrans.csv"))
    df_BBuniverse_BBnoncallable_wTrans = pd.read_csv(os.path.join(directory_output, "AnalysisSectionDirect", dic_noncallable_couponmatching_value, "df_BBuniverse_BBnoncallable_wTrans.csv"))
    df_GBuniverse_GBcallable_wTrans = pd.read_csv(os.path.join(directory_output, "AnalysisSectionDirect", dic_noncallable_couponmatching_value, "df_GBuniverse_GBcallable_wTrans.csv"))
    df_BBuniverse_BBcallable_wTrans = pd.read_csv(os.path.join(directory_output, "AnalysisSectionDirect", dic_noncallable_couponmatching_value, "df_BBuniverse_BBcallable_wTrans.csv"))
    
    # TRUNCATE: cannot remove it does NOT seem to be a data entry error => only appears when dic_noncallable_couponmatching_value is 0 (i.e., looser condition)
    #df_BBuniverse_BBnoncallable_wTrans = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['ytm'] > 17 ]

    # COPY: if imported from the above block
    df_GBuniverse_GBnoncallable_wTrans_original = df_GBuniverse_GBnoncallable_wTrans.copy()
    df_BBuniverse_BBnoncallable_wTrans_original = df_BBuniverse_BBnoncallable_wTrans.copy()
    df_GBuniverse_GBcallable_wTrans_original = df_GBuniverse_GBcallable_wTrans.copy()
    df_BBuniverse_BBcallable_wTrans_original = df_BBuniverse_BBcallable_wTrans.copy()
    
    # NEW CONCAT: if imported from the above block
    df_GBuniverse_Comb_wTrans = pd.concat( [df_GBuniverse_GBnoncallable_wTrans, df_GBuniverse_GBcallable_wTrans], axis=0 )
    df_BBuniverse_Comb_wTrans = pd.concat( [df_BBuniverse_BBnoncallable_wTrans, df_BBuniverse_BBcallable_wTrans], axis=0 )
    df_NonCalluniverse_Comb_wTrans = pd.concat( [df_GBuniverse_GBnoncallable_wTrans, df_BBuniverse_BBnoncallable_wTrans], axis=0 )
    df_Calluniverse_Comb_wTrans = pd.concat( [df_GBuniverse_GBcallable_wTrans, df_BBuniverse_BBcallable_wTrans], axis=0 )

    # NEW column
    df_GBuniverse_Comb_wTrans['munitype'] = 'GB'
    df_BBuniverse_Comb_wTrans['munitype'] = 'BB'
    df_GBuniverse_GBnoncallable_wTrans['municalltype'] = 'GBnoncallable'
    df_BBuniverse_BBnoncallable_wTrans['municalltype'] = 'BBnoncallable'
    df_GBuniverse_GBcallable_wTrans['municalltype'] = 'GBcallable'
    df_BBuniverse_BBcallable_wTrans['municalltype'] = 'BBcallable'
      
    # NEW CONCAT: placed in the end so that new columns included
    df_MUNIuniverse_wTrans = pd.concat( [df_GBuniverse_Comb_wTrans, df_BBuniverse_Comb_wTrans], axis=0 )
    
# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== #

# https://stackoverflow.com/questions/6871201/plot-two-histograms-on-single-chart-with-matplotlib
# https://stackoverflow.com/questions/32899463/how-can-i-overlay-two-graphs-in-seaborn
# https://datatofish.com/plot-histogram-python/

# SELECT: ALWAYS set to 0, no specification => 1 or 1.5 is primary market, aimed for LW replication; 2 is secondary
MarketSpecification = 0;

if MarketSpecification == 1:    
    # narrow down to primary market USING offer_price_takedown_indicator
    df_GBuniverse_GBnoncallable_wTrans = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['offer_price_takedown_indicator'] == 'Y' ]
    df_BBuniverse_BBnoncallable_wTrans = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['offer_price_takedown_indicator'] == 'Y' ]
    df_GBuniverse_GBcallable_wTrans = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['offer_price_takedown_indicator'] == 'Y' ]
    df_BBuniverse_BBcallable_wTrans = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['offer_price_takedown_indicator'] == 'Y' ] 

elif MarketSpecification == 1.5:    
    # narrow down to primary market USING when_issued_indicator
    df_GBuniverse_GBnoncallable_wTrans = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['when_issued_indicator'] == 'Y' ]
    df_BBuniverse_BBnoncallable_wTrans = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['when_issued_indicator'] == 'Y' ]
    df_GBuniverse_GBcallable_wTrans = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['when_issued_indicator'] == 'Y' ]
    df_BBuniverse_BBcallable_wTrans = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['when_issued_indicator'] == 'Y' ] 
    
elif MarketSpecification == 2:        
    # narrow down to secondary market WEAK
    df_GBuniverse_GBnoncallable_wTrans = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['offer_price_takedown_indicator'] != 'Y' ]
    df_BBuniverse_BBnoncallable_wTrans = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['offer_price_takedown_indicator'] != 'Y' ]
    df_GBuniverse_GBcallable_wTrans = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['offer_price_takedown_indicator'] != 'Y' ]
    df_BBuniverse_BBcallable_wTrans = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['offer_price_takedown_indicator'] != 'Y' ]

elif MarketSpecification == 3:        
    # narrow down to secondary market STRONG
    df_GBuniverse_GBnoncallable_wTrans = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['offer_price_takedown_indicator'] != 'Y') & (df_GBuniverse_GBnoncallable_wTrans['when_issued_indicator'] != 'Y') ]
    df_BBuniverse_BBnoncallable_wTrans = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['offer_price_takedown_indicator'] != 'Y') & (df_BBuniverse_BBnoncallable_wTrans['when_issued_indicator'] != 'Y') ]
    df_GBuniverse_GBcallable_wTrans = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['offer_price_takedown_indicator'] != 'Y') & (df_GBuniverse_GBcallable_wTrans['when_issued_indicator'] != 'Y') ]
    df_BBuniverse_BBcallable_wTrans = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['offer_price_takedown_indicator'] != 'Y') & (df_BBuniverse_BBcallable_wTrans['when_issued_indicator'] != 'Y') ]
    
# RECOVER: recover to full sample
df_GBuniverse_GBnoncallable_wTrans = df_GBuniverse_GBnoncallable_wTrans_original.copy()
df_BBuniverse_BBnoncallable_wTrans = df_BBuniverse_BBnoncallable_wTrans_original.copy()
df_GBuniverse_GBcallable_wTrans = df_GBuniverse_GBcallable_wTrans_original.copy()
df_BBuniverse_BBcallable_wTrans = df_BBuniverse_BBcallable_wTrans_original.copy()
    
# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== #

# df1 is GB, df2 is BB
def sumstats_Difftest_beforeLaTeX(dataframe1,dataframe2,state,roundnum): # state='ALL'
    
    if state == 'ALL':
        pass
    else:
        dataframe1 = dataframe1[ dataframe1['state'] == state ]
        dataframe2 = dataframe2[ dataframe2['state'] == state ]
        
    df_beforeLaTeX = pd.DataFrame( data = { 'GBnum': [len( dataframe1['cusip'].unique() )], 'BBnum': [len( dataframe2['cusip'].unique() )], 
                                                      'GByieldmean': [dataframe1['ytm'].mean()], 'BByieldmean': [dataframe2['ytm'].mean()], 'yieldmeanDiff': [dataframe1['ytm'].mean() - dataframe2['ytm'].mean()], 'yieldmeanPvalue': [stats.ttest_ind(dataframe1['ytm'], dataframe2['ytm'], equal_var = False)[1]], 
                                                      'GByieldmedian': [dataframe1['ytm'].median()], 'BByieldmedian': [dataframe2['ytm'].median()], 'yieldmedianDiff': [dataframe1['ytm'].median() - dataframe2['ytm'].median()], 'yieldmedianPvalue': [stats.mannwhitneyu(dataframe1['ytm'],dataframe2['ytm'])[1]], 
                                                      'GBissueamt': [dataframe1.drop_duplicates(subset=['cusip'])['originalissueamt'].sum()], 'BBissueamt': [dataframe2.drop_duplicates(subset=['cusip'])['originalissueamt'].sum()] } ) # checked the following and the result differs from with drop_duplicates(): dataframe1['originalissueamt'].sum()
    
    return df_beforeLaTeX.round(roundnum) #.to_latex()


# LATEX: key LLL => adjust primary+secondary OR primary only using ABOVE
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBnoncallable_wTrans,df_BBuniverse_BBnoncallable_wTrans,'AZ',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBnoncallable_wTrans,df_BBuniverse_BBnoncallable_wTrans,'CA',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBnoncallable_wTrans,df_BBuniverse_BBnoncallable_wTrans,'MA',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBnoncallable_wTrans,df_BBuniverse_BBnoncallable_wTrans,'NY',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBnoncallable_wTrans,df_BBuniverse_BBnoncallable_wTrans,'TX',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBnoncallable_wTrans,df_BBuniverse_BBnoncallable_wTrans,'DC',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBnoncallable_wTrans,df_BBuniverse_BBnoncallable_wTrans,'ALL',3).to_latex(index=False)

sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBcallable_wTrans,df_BBuniverse_BBcallable_wTrans,'AZ',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBcallable_wTrans,df_BBuniverse_BBcallable_wTrans,'CA',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBcallable_wTrans,df_BBuniverse_BBcallable_wTrans,'MA',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBcallable_wTrans,df_BBuniverse_BBcallable_wTrans,'NY',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBcallable_wTrans,df_BBuniverse_BBcallable_wTrans,'TX',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBcallable_wTrans,df_BBuniverse_BBcallable_wTrans,'DC',3).to_latex(index=False)
sumstats_Difftest_beforeLaTeX(df_GBuniverse_GBcallable_wTrans,df_BBuniverse_BBcallable_wTrans,'ALL',3).to_latex(index=False)

sumstats_Difftest_beforeLaTeX(df_GBuniverse_Comb_wTrans,df_BBuniverse_Comb_wTrans,'ALL',3).to_latex(index=False)


# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== #

# seaborn histogram overlay: https://python-graph-gallery.com/25-histogram-with-several-variables-seaborn/
# AttributeError: module 'seaborn' has no attribute 'plt' => just omit "sns.": https://stackoverflow.com/questions/47989743/sns-plt-show-not-working?noredirect=1&lq=1
# kde: https://seaborn.pydata.org/tutorial/distributions.html

TargetVar = 'ytm' # current_yield, dollar_price, ytm => DO NOT originalissueamt HERE FOR PURE COMPARISON (COMFOUNDED BY NUM TRANSACTIONS)
#originalissueamt = 3 => INFEASIBLE as GB-BB match will be destroyed

# Plain 
x = df_GBuniverse_GBnoncallable_wTrans[TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[TargetVar] 
z = df_GBuniverse_GBcallable_wTrans[TargetVar]; w = df_BBuniverse_BBcallable_wTrans[TargetVar] 
#x = df_GBuniverse_Comb_wTrans[TargetVar]; y = df_BBuniverse_Comb_wTrans[TargetVar]

# Plain originalissueamt => confounded by num of transactions so need .unique()
x = df_GBuniverse_GBnoncallable_wTrans.drop_duplicates(subset=['cusip'])['originalissueamt']; y = df_BBuniverse_BBnoncallable_wTrans.drop_duplicates(subset=['cusip'])['originalissueamt']
z = df_GBuniverse_GBcallable_wTrans.drop_duplicates(subset=['cusip'])['originalissueamt']; w = df_BBuniverse_BBcallable_wTrans.drop_duplicates(subset=['cusip'])['originalissueamt']

# 1DPLOT: plain
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.title('non-callable bond universe'); plt.legend()
x.describe()
y.describe()
x.mean() - y.mean()
x.median() - y.median()
welch_ttest(x,y)
stats.wilcoxon(x, y)
stats.mannwhitneyu(x,y)
stats.mood(x, y, axis=0)

sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.title('callable bond universe'); plt.legend()
z.describe()
w.describe()
z.mean() - w.mean()
z.median() - w.median()
welch_ttest(z,w)

# ************************ #

# Corona: strict => good even for callable
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20200500) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20200500) ][TargetVar] 
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20200500) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20200500) ][TargetVar] 
#x = df_GBuniverse_Comb_wTrans[ (df_GBuniverse_Comb_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_Comb_wTrans['trade_date_int'] < 20200500) ][TargetVar]; y = df_BBuniverse_Comb_wTrans[ (df_BBuniverse_Comb_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_Comb_wTrans['trade_date_int'] < 20200500) ][TargetVar] 

# Corona: 3M after mid March => max is 20200630 anyway: df_GBuniverse_GBnoncallable_wTrans['trade_date_int'].max()
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20200615) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20200615) ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20200615) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20200615) ][TargetVar]


# Coldwave: 2019 Jan–Feb => EXCELLENT
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190100) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190300) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190100) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190300) ][TargetVar] 
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190100) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190300) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190100) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190300) ][TargetVar] 

# Coldwave: 2019 Jan–March => MIXED RESULT -10 bps for noncallable, mean/median twisted for callable
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190100) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190100) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar] 
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190100) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190100) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar] 

# Coldwave: 2019 April–June 
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190400) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190400) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar] 
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190400) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190400) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar] 

# Coldwave: 2019 Feb–June 
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar] 
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar] 

# ****

# Coldwave: 2018 WORKS AS PLACEBO!
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20180100) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190100) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20180100) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190100) ][TargetVar] 
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20180100) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190100) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20180100) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190100) ][TargetVar] 

# Coldwave: 2019 Junl–2020 WORKS AS PLACEBO!
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190700) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20200100) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190700) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20200100) ][TargetVar] 
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190700) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20200100) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190700) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20200100) ][TargetVar] 

# 1DPLOT: general purpose
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.legend()
x.describe()
y.describe()
sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.legend()
z.describe()
w.describe()

'''
# Corona: after mid March with NewYork => clear ytm diff for both i) coupon indifferent, and ii) coupon match
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20201215) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'NY') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20201215) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'NY') ][TargetVar]
x = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20201215) & (df_GBuniverse_GBcallable_wTrans['state'] == 'NY') ][TargetVar]; y = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20201215) & (df_BBuniverse_BBcallable_wTrans['state'] == 'NY') ][TargetVar]

# Corona: after mid March with California => little ytm diff for i) coupon indiff, but actually different in ii) coupon match case
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20201215) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'CA') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20201215) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'CA') ][TargetVar]

# Corona: after mid March with Texas => little ytm diff for i) coupon indiff, but actually different in ii) coupon match case
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20201215) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'TX') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20201215) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'TX') ][TargetVar]

'''
##############

# Placebo 1: before corona
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150300) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20191215) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20150300) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20191215) ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20150300) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20191215) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20150300) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20191215) ][TargetVar]

# +++++++++++

# Placebo 2: plain in CA
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'CA') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'CA') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'CA') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'CA') ][TargetVar]

# Placebo 2.1: corona in CA during 2020 Feb–May (2020 Sep–Oct no sample data): https://trends.google.de/trends/explore?date=2020-01-01%202021-02-24&geo=US&q=corona%20%22air%20quality%22
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20200200) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20200600) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'CA') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20200200) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20200600) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'CA') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20200200) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20200600) & (df_GBuniverse_GBcallable_wTrans['state'] == 'CA') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20200200) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20200600) & (df_BBuniverse_BBcallable_wTrans['state'] == 'CA') ][TargetVar]

# Placebo 2.2: CA originalissueamt => confounded by num of transactions so need .unique()
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'CA') ].drop_duplicates(subset=['cusip'])['originalissueamt']; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'CA') ].drop_duplicates(subset=['cusip'])['originalissueamt']
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'CA') ].drop_duplicates(subset=['cusip'])['originalissueamt']; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'CA') ].drop_duplicates(subset=['cusip'])['originalissueamt']

# 1DPLOT: conditional on CA
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.title('non-callable bonds in California'); plt.legend()
x.mean() - y.mean()
x.median() - y.median()
welch_ttest(x,y)

sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.title('callable bonds in California'); plt.legend()
z.mean() - w.mean()
z.median() - w.median()
welch_ttest(z,w)

# EXPORT
if export_dataset == 1: 
    # NEW CONCAT
    df_MUNIuniverse_CAonly_wTrans = pd.concat( [ df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'CA') ], df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'CA') ], df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'CA') ], df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'CA') ] ], axis=0 )    
    df_MUNIuniverse_CAonly_wTrans.to_excel( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_MUNIuniverse_CAonly_wTrans.xlsx"), index=False)

# +++++++++++

# Placebo 3: plain in NY
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'NY') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'NY') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'NY') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'NY') ][TargetVar]

# Placebo 3.1: before corona in NY
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'NY') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'NY') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20191215) & (df_GBuniverse_GBcallable_wTrans['state'] == 'NY') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20191215) & (df_BBuniverse_BBcallable_wTrans['state'] == 'NY') ][TargetVar]

# Placebo 3.2: NY originalissueamt => confounded by num of transactions so need .unique()
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'NY') ].drop_duplicates(subset=['cusip'])['originalissueamt']; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'NY') ].drop_duplicates(subset=['cusip'])['originalissueamt']
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'NY') ].drop_duplicates(subset=['cusip'])['originalissueamt']; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'NY') ].drop_duplicates(subset=['cusip'])['originalissueamt']

# EXPORT
if export_dataset == 1: 
    # NEW CONCAT
    df_MUNIuniverse_NYonly_wTrans = pd.concat( [ df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'NY') ], df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'NY') ], df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'NY') ], df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'NY') ] ], axis=0 )    
    df_MUNIuniverse_NYonly_wTrans.to_excel( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_MUNIuniverse_NYonly_wTrans.xlsx"), index=False)

# 1DPLOT: conditional on NY
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.title('non-callable bonds in New York'); plt.legend()
x.mean() - y.mean()
x.median() - y.median()
welch_ttest(x,y)

sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.title('callable bonds in New York'); plt.legend()
z.mean() - w.mean()
z.median() - w.median()
welch_ttest(z,w)

# ******

# Placebo 4: plain in MA => @@@ EXPORT TO CHECK!!! @@@
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MA') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'MA') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'MA') ][TargetVar]

# Placebo 4.1: in MA after a while from June 2014 => considerable greenium even after 2018
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20180000) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20180000) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MA') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20180000) & (df_GBuniverse_GBcallable_wTrans['state'] == 'MA') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20180000) & (df_BBuniverse_BBcallable_wTrans['state'] == 'MA') ][TargetVar]

# Placebo 4.2: MA originalissueamt => confounded by num of transactions so need .unique()
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ].drop_duplicates(subset=['cusip'])['originalissueamt']; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MA') ].drop_duplicates(subset=['cusip'])['originalissueamt']
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'MA') ].drop_duplicates(subset=['cusip'])['originalissueamt']; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'MA') ].drop_duplicates(subset=['cusip'])['originalissueamt']

# 1DPLOT: conditional on MA
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.title('non-callable bonds in Massachusetts'); plt.legend()
x.mean() - y.mean()
x.median() - y.median()
welch_ttest(x,y)
stats.mannwhitneyu(x,y)
#stats.mood(x, y, axis=0) => weird

sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.title('callable bonds in Massachusetts'); plt.legend()
z.mean() - w.mean()
z.median() - w.median()
welch_ttest(z,w)
stats.mannwhitneyu(z,w)
#stats.mood(z, w, axis=0) => weird

# EXPORT
if export_dataset == 1: 
    # NEW CONCAT
    df_MUNIuniverse_MAonly_wTrans = pd.concat( [ df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ], df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MA') ], df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'MA') ], df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'MA') ] ], axis=0 )    
    df_MUNIuniverse_MAonly_wTrans.to_excel( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_MUNIuniverse_MAonly_wTrans.xlsx"), index=False)
    
# ******

# Placebo 5: before corona in AZ
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'AZ') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'AZ') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20191215) & (df_GBuniverse_GBcallable_wTrans['state'] == 'AZ') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20191215) & (df_BBuniverse_BBcallable_wTrans['state'] == 'AZ') ][TargetVar]

# Placebo 6: before corona in WA
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'WA') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'WA') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20191215) & (df_GBuniverse_GBcallable_wTrans['state'] == 'WA') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20191215) & (df_BBuniverse_BBcallable_wTrans['state'] == 'WA') ][TargetVar]

# Placebo 7: plain in MD
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MD') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MD') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'MD') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'MD') ][TargetVar]

# Placebo 7.1: before corona in MD
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MD') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MD') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20191215) & (df_GBuniverse_GBcallable_wTrans['state'] == 'MD') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20191215) & (df_BBuniverse_BBcallable_wTrans['state'] == 'MD') ][TargetVar]

# ******

# Placebo 8: plain in DC
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'DC') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'DC') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'DC') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'DC') ][TargetVar]

# Placebo 8.1: in DC after a while from June 2014
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20180000) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'DC') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20180000) & (df_BBuniverse_BBnoncallable_wTrans['state'] == 'DC') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20180000) & (df_GBuniverse_GBcallable_wTrans['state'] == 'DC') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20180000) & (df_BBuniverse_BBcallable_wTrans['state'] == 'DC') ][TargetVar]

# Placebo 8.2: DC originalissueamt => confounded by num of transactions so need .unique()
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'DC') ].drop_duplicates(subset=['cusip'])['originalissueamt']; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'DC') ].drop_duplicates(subset=['cusip'])['originalissueamt']
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'DC') ].drop_duplicates(subset=['cusip'])['originalissueamt']; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'DC') ].drop_duplicates(subset=['cusip'])['originalissueamt']


# 1DPLOT: conditional on DC
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.title('non-callable bonds in Washington, D.C.'); plt.legend()
x.mean() - y.mean()
x.median() - y.median()
welch_ttest(x,y)
sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.title('callable bonds in Washington, D.C.'); plt.legend()
z.mean() - w.mean()
z.median() - w.median()
welch_ttest(z,w)

# EXPORT
if export_dataset == 1: 
    # NEW CONCAT
    df_MUNIuniverse_DConly_wTrans = pd.concat( [ df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'DC') ], df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'DC') ], df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'DC') ], df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'DC') ] ], axis=0 )    
    df_MUNIuniverse_DConly_wTrans.to_excel( os.path.join(directory_output, dic_noncallable_couponmatching_value, "df_MUNIuniverse_DConly_wTrans.xlsx"), index=False)

# ******
    
# Placebo 9: plain in TX
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'TX') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'TX') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'TX') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'TX') ][TargetVar]

# Placebo 9.2: TX originalissueamt => confounded by num of transactions so need .unique()
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'TX') ].drop_duplicates(subset=['cusip'])['originalissueamt']; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'TX') ].drop_duplicates(subset=['cusip'])['originalissueamt']
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'TX') ].drop_duplicates(subset=['cusip'])['originalissueamt']; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'TX') ].drop_duplicates(subset=['cusip'])['originalissueamt']


# 1DPLOT: conditional on TX
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.title('non-callable bonds in Texas'); plt.legend()
x.mean() - y.mean()
x.median() - y.median()
welch_ttest(x,y)

sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.title('callable bonds in Texas'); plt.legend()
z.mean() - w.mean()
z.median() - w.median()
welch_ttest(z,w)

# ******
    
# Placebo 10: plain in AZ
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'AZ') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'AZ') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'AZ') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'AZ') ][TargetVar]

# Placebo 10.2: AZ originalissueamt => confounded by num of transactions so need .unique()
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'AZ') ].drop_duplicates(subset=['cusip'])['originalissueamt']; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'AZ') ].drop_duplicates(subset=['cusip'])['originalissueamt']
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'AZ') ].drop_duplicates(subset=['cusip'])['originalissueamt']; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'AZ') ].drop_duplicates(subset=['cusip'])['originalissueamt']


# 1DPLOT: conditional on AZ
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.title('non-callable bonds in Alizona'); plt.legend()
x.mean() - y.mean()
x.median() - y.median()
welch_ttest(x,y)

sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.title('callable bonds in Alizona'); plt.legend()
z.mean() - w.mean()
z.median() - w.median()
welch_ttest(z,w)

'''
# [state level agg stats]
df_GBuniverse_GBnoncallable_wTrans.groupby(['state'])['cusip'].count()
df_GBuniverse_GBcallable_wTrans.groupby(['state'])['cusip'].count()

'''
# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #

# [Climate Opinion Analysis] 

TargetVar = 'ytm' # current_yield, dollar_price, ytm

# [State level analysis]    
# Env awareness: 'happeningUD', 'humanUD', 'worriedUD', 'harmUSUD'
x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['humanUD'] == 1 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['humanUD'] == 1 ][TargetVar]
x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['humanUD'] == 0 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['humanUD'] == 0 ][TargetVar]

z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['humanUD'] == 1 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['humanUD'] == 1 ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['humanUD'] == 0 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['humanUD'] == 0 ][TargetVar]


x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['worriedUD'] == 1 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['worriedUD'] == 1 ][TargetVar]
x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['worriedUD'] == 0 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['worriedUD'] == 0 ][TargetVar]

z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['worriedUD'] == 1 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['worriedUD'] == 1 ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['worriedUD'] == 0 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['worriedUD'] == 0 ][TargetVar]


x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['CO2limitsUD'] == 1 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['CO2limitsUD'] == 1 ][TargetVar]
x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['CO2limitsUD'] == 0 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['CO2limitsUD'] == 0 ][TargetVar]

z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['CO2limitsUD'] == 1 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['CO2limitsUD'] == 1 ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['CO2limitsUD'] == 0 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['CO2limitsUD'] == 0 ][TargetVar]

# +++++++++++
# [County level analysis] Key: @*$#
# NO ANALYSIS => little variation in the low climate belife area!!!
'''
x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['humanUD_county'] == 1 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['humanUD_county'] == 1 ][TargetVar]
x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['humanUD_county'] == 0 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['humanUD_county'] == 0 ][TargetVar]

z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['humanUD_county'] == 1 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['humanUD_county'] == 1 ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['humanUD_county'] == 0 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['humanUD_county'] == 0 ][TargetVar]


x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['CO2limitsUD_county'] == 1 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['CO2limitsUD_county'] == 1 ][TargetVar]
x = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['CO2limitsUD_county'] == 0 ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans['CO2limitsUD_county'] == 0 ][TargetVar]

z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['CO2limitsUD_county'] == 1 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['CO2limitsUD_county'] == 1 ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans['CO2limitsUD_county'] == 0 ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans['CO2limitsUD_county'] == 0 ][TargetVar]

'''
# +++++++++++

# 1DPLOT: Differential pricing in high/low climate-belief area @@@@@@@@@@ CHANGE TITLE BASED ON VARIABLE @@@@@@@@@@@
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.title('High climate-belief area (XXX)'); plt.legend()
sns.distplot( x, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y, color="firebrick", label="BBnoncallable"); plt.title('Low climate-belief area (XXX)'); plt.legend()
x.describe()
y.describe()
sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.title('High climate-belief area (XXX)'); plt.legend()
sns.distplot( z, color="royalblue", label="GBcallable"); sns.distplot( w, color="darkorange", label="BBcallable"); plt.title('Low climate-belief area (XXX)'); plt.legend()
z.describe()
w.describe()

'''
# [Conditional on MA: https://www.climatebonds.net/files/files/Green%20City%20Playbook.pdf]

x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['humanUD'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['humanUD'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ][TargetVar]
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['humanUD'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['humanUD'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ][TargetVar]

z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['humanUD'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['humanUD'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['humanUD'] == 0 & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA')) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['humanUD'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ][TargetVar]


# [Env awareness + Corona strict]: not so workable
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['humanUD'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20200500) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['humanUD'] == 1) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20200500) ][TargetVar] 
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['humanUD'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20200500) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['humanUD'] == 0) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20200500) ][TargetVar] 

z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['humanUD'] == 1) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20200500) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['humanUD'] == 1) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20200500) ][TargetVar] 
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['humanUD'] == 0) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20200500) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['humanUD'] == 0) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20200500) ][TargetVar] 

'''
# ******************* #

# Coldwave 2019 Feb–April => @@@@@ GOOD  @@@@@
x1 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190500) ][TargetVar]; y1 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190500) ][TargetVar]
x0 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190500) ][TargetVar]; y0 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190500) ][TargetVar]

z1 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190500) ][TargetVar]; w1 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190500) ][TargetVar]
z0 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190500) ][TargetVar]; w0 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190500) ][TargetVar]


# Coldwave 2019 Feb–March => @@@@@ GOOD  @@@@@
x1 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; y1 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]
x0 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; y0 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]

z1 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; w1 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]
z0 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; w0 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]


# Coldwave 2019 Jan–March => @@@@@ GOOD  @@@@@
x1 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190100) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; y1 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190100) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]
x0 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190100) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; y0 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190100) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]

z1 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190100) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; w1 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190100) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]
z0 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190100) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]; w0 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190100) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190400) ][TargetVar]


# Coldwave 2019 April–June => placebo
x1 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190400) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; y1 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190400) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]
x0 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190400) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; y0 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190400) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]

z1 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190400) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; w1 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190400) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]
z0 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190400) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; w0 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190400) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]


# Coldwave 2019 May–June => placebo
x1 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190500) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; y1 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190500) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]
x0 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190500) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; y0 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190500) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]

z1 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 1) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190500) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; w1 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 1) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190500) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]
z0 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['coldwave2019'] == 0) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190500) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]; w0 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['coldwave2019'] == 0) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190500) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190700) ][TargetVar]


# 1DPLOT: coldwave
sns.distplot( x1, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y1, color="firebrick", label="BBnoncallable"); plt.title('Cold wave in 2019'); plt.legend()
x1.describe()
y1.describe()
sns.distplot( x0, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y0, color="firebrick", label="BBnoncallable"); plt.title('Cold wave in 2019'); plt.legend()
x0.describe()
y0.describe()

sns.distplot( z1, color="royalblue", label="GBcallable"); sns.distplot( w1, color="darkorange", label="BBcallable"); plt.title('Cold wave in 2019'); plt.legend()
z1.describe()
w1.describe()
sns.distplot( z0, color="royalblue", label="GBcallable"); sns.distplot( w0, color="darkorange", label="BBcallable"); plt.title('Cold wave in 2019'); plt.legend()
z0.describe()
w0.describe()

# ******************* #

# Heat wave 2018 July–Aug => 
x1 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['heatwave2018'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20180900) ][TargetVar]; y1 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['heatwave2018'] == 1) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20180900) ][TargetVar]
x0 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['heatwave2018'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20180900) ][TargetVar]; y0 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['heatwave2018'] == 0) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20180900) ][TargetVar]

z1 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['heatwave2018'] == 1) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20180900) ][TargetVar]; w1 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['heatwave2018'] == 1) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20180900) ][TargetVar]
z0 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['heatwave2018'] == 0) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20180900) ][TargetVar]; w0 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['heatwave2018'] == 0) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20180900) ][TargetVar]

# Heat wave 2018 July–Sep => 
x1 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['heatwave2018'] == 1) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]; y1 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['heatwave2018'] == 1) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]
x0 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['heatwave2018'] == 0) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]; y0 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['heatwave2018'] == 0) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]

z1 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['heatwave2018'] == 1) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]; w1 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['heatwave2018'] == 1) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]
z0 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['heatwave2018'] == 0) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]; w0 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['heatwave2018'] == 0) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]

# Heat wave 2018 July–Sep in CA => 
x1 = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'CA') & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]; y1 = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'CA') & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]
z1 = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'CA') & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20180700) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]; w1 = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'CA') & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20180700) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20181000) ][TargetVar]



# 1DPLOT: heat wave
sns.distplot( x1, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y1, color="firebrick", label="BBnoncallable"); plt.title('Heat wave in 2018'); plt.legend()
x1.describe()
y1.describe()
sns.distplot( x0, color="mediumseagreen", label="GBnoncallable"); sns.distplot( y0, color="firebrick", label="BBnoncallable"); plt.title('Heat wave in 2018'); plt.legend()
x0.describe()
y0.describe()

sns.distplot( z1, color="royalblue", label="GBcallable"); sns.distplot( w1, color="darkorange", label="BBcallable"); plt.title('Heat wave in 2018'); plt.legend()
z1.describe()
w1.describe()
sns.distplot( z0, color="royalblue", label="GBcallable"); sns.distplot( w0, color="darkorange", label="BBcallable"); plt.title('Heat wave in 2018'); plt.legend()
z0.describe()
w0.describe()

# ******************* #
'''
# USCA: 1M cohortABC
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20170600) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20170700) & (df_GBuniverse_GBnoncallable_wTrans['USCAcohortABC'] == 1) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20170600) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20170700) & (df_BBuniverse_BBnoncallable_wTrans['USCAcohortABC'] == 1) ][TargetVar]
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20170600) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20170700) & (df_GBuniverse_GBnoncallable_wTrans['USCAcohortABC'] == 0) ][TargetVar]; y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20170600) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20170700) & (df_BBuniverse_BBnoncallable_wTrans['USCAcohortABC'] == 0) ][TargetVar]

z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20170600) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20170700) & (df_GBuniverse_GBcallable_wTrans['USCAcohortABC'] == 1) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20170600) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20170700) & (df_BBuniverse_BBcallable_wTrans['USCAcohortABC'] == 1) ][TargetVar]
z = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20170600) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20170700) & (df_GBuniverse_GBcallable_wTrans['USCAcohortABC'] == 0) ][TargetVar]; w = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20170600) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20170700) & (df_BBuniverse_BBcallable_wTrans['USCAcohortABC'] == 0) ][TargetVar]

'''
# ******************* #

# DISPLOT
#sns.displot( x, color="mediumseagreen", label="Green Muni", kind="kde"); sns.displot( y, color="firebrick", label="Brown Muni", kind="kde"); plt.legend()

# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #
# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #

# [Univariate – Trading volume: LW focuses on secondary market]

def tradingvol(dataframe):
    # CHECK ok: The following equals to row num => dataframe.groupby(['cusip'])['rtrs_control_number'].count().sum()
    df_trading = dataframe.groupby(['cusip'])['rtrs_control_number'].count().reset_index().rename(columns={'rtrs_control_number':'RTRS'})

    return df_trading['RTRS']

def logtradingvol(dataframe):
    # CHECK ok: The following equals to row num => dataframe.groupby(['cusip'])['rtrs_control_number'].count().sum()
    df_trading = dataframe.groupby(['cusip'])['rtrs_control_number'].count().reset_index().rename(columns={'rtrs_control_number':'RTRS'})

    return np.log( df_trading['RTRS'])

# ************ #
    
# TRADING Plain
tx = tradingvol(df_GBuniverse_GBnoncallable_wTrans); ty = tradingvol(df_BBuniverse_BBnoncallable_wTrans)
tz = tradingvol(df_GBuniverse_GBcallable_wTrans); tw = tradingvol(df_BBuniverse_BBcallable_wTrans)

tx = logtradingvol(df_GBuniverse_GBnoncallable_wTrans); ty = logtradingvol(df_BBuniverse_BBnoncallable_wTrans)
tz = logtradingvol(df_GBuniverse_GBcallable_wTrans); tw = logtradingvol(df_BBuniverse_BBcallable_wTrans)


# TRADING Corona: 3M after mid March
tx = logtradingvol( df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20200615) ] ); ty = logtradingvol( df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20200615) ] )
tz = logtradingvol( df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20200315) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20200615) ] ); tw = logtradingvol( df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20200315) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20200615) ] )


# TRADING Coldwave
tx = tradingvol( df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190600) ] ); ty = tradingvol( df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190600) ] )
tz = tradingvol( df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190600) ] ); tw = tradingvol( df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190600) ] )

tx = logtradingvol( df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20190600) ] ); ty = logtradingvol( df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20190600) ] )
tz = logtradingvol( df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['trade_date_int'] > 20190200) & (df_GBuniverse_GBcallable_wTrans['trade_date_int'] < 20190600) ] ); tw = logtradingvol( df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['trade_date_int'] > 20190200) & (df_BBuniverse_BBcallable_wTrans['trade_date_int'] < 20190600) ] )


# TRADING Conditional on MA: https://www.climatebonds.net/files/files/Green%20City%20Playbook.pdf
tx = tradingvol( df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ] ); ty = tradingvol( df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MA') ] )
tz = tradingvol( df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'MA') ] ); tw = tradingvol( df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'MA') ] )

tx = logtradingvol( df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') ] ); ty = logtradingvol( df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MA') ] )
tz = logtradingvol( df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'MA') ] ); tw = logtradingvol( df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'MA') ] )


# TRADING Conditional on MA after a while from June 2014 => callable non existent
tx = tradingvol( df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150000) ] ); ty = tradingvol( df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150000) ] )
tz = tradingvol( df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150000) ] ); tw = tradingvol( df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150000) ] )

tx = logtradingvol( df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150000) ] ); ty = logtradingvol( df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150000) ] )
tz = logtradingvol( df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150000) ] ); tw = logtradingvol( df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150000) ] )

# ************ #

# TRADING 1DPLOT:
sns.distplot( tx, color="mediumseagreen", label="GBnoncallable"); sns.distplot( ty, color="firebrick", label="BBnoncallable"); plt.legend()
tx.describe()
ty.describe()
sns.distplot( tz, color="mediumseagreen", label="GBcallable"); sns.distplot( tw, color="firebrick", label="BBcallable"); plt.legend()
tz.describe()
tw.describe()

#####%%

# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #
# ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ ++++ #
'''
 - our first measure is based on the idea that institutional (retail) purchases are assigned to be purchases with par volume greater than or equal (less than) to $100,000.
 - Institutional ownership is defined as total sum of institutional purchases divided by total securities outstanding.
 
 
'''
# [Univariate – par_traded: Institutional investors]

# par_traded is raw while issue amt is in MM
df_GBuniverse_GBnoncallable_wTrans['par_over_amt'] = df_GBuniverse_GBnoncallable_wTrans['par_traded'] / ( df_GBuniverse_GBnoncallable_wTrans['originalissueamt'] * 1000000)
df_BBuniverse_BBnoncallable_wTrans['par_over_amt'] = df_BBuniverse_BBnoncallable_wTrans['par_traded'] / ( df_BBuniverse_BBnoncallable_wTrans['originalissueamt'] * 1000000)
df_GBuniverse_GBcallable_wTrans['par_over_amt'] = df_GBuniverse_GBcallable_wTrans['par_traded'] / ( df_GBuniverse_GBcallable_wTrans['originalissueamt'] * 1000000)
df_BBuniverse_BBcallable_wTrans['par_over_amt'] = df_BBuniverse_BBcallable_wTrans['par_traded'] / ( df_BBuniverse_BBcallable_wTrans['originalissueamt'] * 1000000)

#################

primary_indicator = 'offer_price_takedown_indicator' # 'offer_price_takedown_indicator', 'when_issued_indicator'

# primary plain (institutional + retail):
ix = df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans[primary_indicator] == 'Y' ]['par_over_amt']; iy = df_BBuniverse_BBnoncallable_wTrans[ df_BBuniverse_BBnoncallable_wTrans[primary_indicator] == 'Y' ]['par_over_amt']
iz = df_GBuniverse_GBcallable_wTrans[ df_GBuniverse_GBcallable_wTrans[primary_indicator] == 'Y' ]['par_over_amt']; iw = df_BBuniverse_BBcallable_wTrans[ df_BBuniverse_BBcallable_wTrans[primary_indicator] == 'Y' ]['par_over_amt']

# primary; par_traded > 100000 (institutional)
ix = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans[primary_indicator] == 'Y') & (df_GBuniverse_GBnoncallable_wTrans['par_traded'] >= 100000) ]['par_over_amt']; iy = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans[primary_indicator] == 'Y') & (df_BBuniverse_BBnoncallable_wTrans['par_traded'] >= 100000) ]['par_over_amt']
iz = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans[primary_indicator] == 'Y') & (df_GBuniverse_GBcallable_wTrans['par_traded'] >= 100000) ]['par_over_amt']; iw = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans[primary_indicator] == 'Y') & (df_BBuniverse_BBcallable_wTrans['par_traded'] >= 100000) ]['par_over_amt']

# primary; par_traded > 100000; STATE
ix = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBnoncallable_wTrans[primary_indicator] == 'Y') & (df_GBuniverse_GBnoncallable_wTrans['par_traded'] >= 100000) ]['par_over_amt']; iy = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['state'] == 'MA') & (df_BBuniverse_BBnoncallable_wTrans[primary_indicator] == 'Y') & (df_BBuniverse_BBnoncallable_wTrans['par_traded'] >= 100000) ]['par_over_amt']
iz = df_GBuniverse_GBcallable_wTrans[ (df_GBuniverse_GBcallable_wTrans['state'] == 'MA') & (df_GBuniverse_GBcallable_wTrans[primary_indicator] == 'Y') & (df_GBuniverse_GBcallable_wTrans['par_traded'] >= 100000) ]['par_over_amt']; iw = df_BBuniverse_BBcallable_wTrans[ (df_BBuniverse_BBcallable_wTrans['state'] == 'MA') & (df_BBuniverse_BBcallable_wTrans[primary_indicator] == 'Y') & (df_BBuniverse_BBcallable_wTrans['par_traded'] >= 100000) ]['par_over_amt']


sns.distplot( ix, color="mediumseagreen", label="GBnoncallable"); sns.distplot( iy, color="firebrick", label="BBnoncallable"); plt.legend()
ix.describe()
iy.describe()
sns.distplot( iz, color="royalblue", label="GBcallable"); sns.distplot( iw, color="darkorange", label="BBcallable"); plt.legend()
iz.describe()
iw.describe()

#%% Section 8. Kernel density plotting (bivariate)
'''
# Visualizing bivariate distributions 
 - https://seaborn.pydata.org/tutorial/distributions.html#visualizing-bivariate-distributions
 - https://seaborn.pydata.org/generated/seaborn.kdeplot.html

 => fill=True use or not?

# Color distrubution 
 - The easiest solution to make sure to have the same colors for the same categories in both plots would be to manually specify the colors at plot creation.
   , palette=["C0", "C1", "k"] => https://stackoverflow.com/questions/46173419/seaborn-change-bar-colour-according-to-hue-name
   
 - 100 colors: https://python-graph-gallery.com/100-calling-a-color-with-seaborn/  
 - color palette : https://stackoverflow.com/questions/54336695/selecting-colors-from-seaborn-palette
 - firebrick and maroon are similar but firebrick >> maroon, since more transparent and suits well with mediumseagreen
 
# Countour
 - How to label a seaborn contour plot: https://stackoverflow.com/questions/33169093/how-to-label-a-seaborn-contour-plot
 
 
'''

# ytm vs env awareness

# PRIMARY MARKET: even in PM, where the ytm diff is zero, there could be diff relying on env awareness
# => GB may experience lower ytm in high EA area, while higher ytm in low EA area


# DISPLOT
'''
sns.displot(df_GBuniverse_GBnoncallable_wTrans, x='ytm', y='coupon', color="mediumseagreen", label="Green Muni", kind="kde"); sns.displot( df_BBuniverse_BBnoncallable_wTrans, x='ytm', y='coupon', color="firebrick", label="Brown Muni", kind="kde"); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['human'])
df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['happening'])
df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['worried'])
df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['harmUS'])
df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['CO2limits'])
, x='ytm', y='happening', color="mediumseagreen", label="Green Muni", kind="kde"); sns.displot( df_BBuniverse_BBnoncallable_wTrans, x='ytm', y='happening', color="firebrick", label="Brown Muni", kind="kde"); plt.legend()
sns.displot(df_GBuniverse_GBnoncallable_wTrans, x='ytm', y='human', color="mediumseagreen", label="Green Muni", kind="kde"); sns.displot( df_BBuniverse_BBnoncallable_wTrans, x='ytm', y='human', color="firebrick", label="Brown Muni", kind="kde"); plt.legend()
sns.displot(df_GBuniverse_GBnoncallable_wTrans, x='ytm', y='worried', color="mediumseagreen", label="Green Muni", kind="kde"); sns.displot( df_BBuniverse_BBnoncallable_wTrans, x='ytm', y='worried', color="firebrick", label="Brown Muni", kind="kde"); plt.legend()
sns.displot(df_GBuniverse_GBnoncallable_wTrans, x='ytm', y='harmUS', color="mediumseagreen", label="Green Muni", kind="kde"); sns.displot( df_BBuniverse_BBnoncallable_wTrans, x='ytm', y='harmUS', color="firebrick", label="Brown Muni", kind="kde"); plt.legend()

'''

# [2DPLOT: STATE level]

# ytm: try both fill=True and False
sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='human', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at state level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='human', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at state level)'); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='happening', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at state level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='happening', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at state level)'); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='worried', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at state level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='worried', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at state level)'); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='harmUS', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at state level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='harmUS', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at state level)'); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='CO2limits', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at state level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='CO2limits', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at state level)'); plt.legend()


# CORR: if county level => ValueError: array must not contain infs or NaNs
df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['human'])
df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['happening'])
df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['worried'])
df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['harmUS'])
df_NonCalluniverse_Comb_wTrans['ytm'].corr(df_NonCalluniverse_Comb_wTrans['CO2limits'])

df_Calluniverse_Comb_wTrans['ytm'].corr(df_Calluniverse_Comb_wTrans['human'])
df_Calluniverse_Comb_wTrans['ytm'].corr(df_Calluniverse_Comb_wTrans['happening'])
df_Calluniverse_Comb_wTrans['ytm'].corr(df_Calluniverse_Comb_wTrans['worried'])
df_Calluniverse_Comb_wTrans['ytm'].corr(df_Calluniverse_Comb_wTrans['harmUS'])
df_Calluniverse_Comb_wTrans['ytm'].corr(df_Calluniverse_Comb_wTrans['CO2limits'])

df_GBuniverse_GBnoncallable_wTrans['ytm'].corr(df_GBuniverse_GBnoncallable_wTrans['human'])
df_GBuniverse_GBnoncallable_wTrans['ytm'].corr(df_GBuniverse_GBnoncallable_wTrans['happening'])
df_GBuniverse_GBnoncallable_wTrans['ytm'].corr(df_GBuniverse_GBnoncallable_wTrans['worried'])
df_GBuniverse_GBnoncallable_wTrans['ytm'].corr(df_GBuniverse_GBnoncallable_wTrans['harmUS'])
df_GBuniverse_GBnoncallable_wTrans['ytm'].corr(df_GBuniverse_GBnoncallable_wTrans['CO2limits'])

df_BBuniverse_BBnoncallable_wTrans['ytm'].corr(df_BBuniverse_BBnoncallable_wTrans['human'])
df_BBuniverse_BBnoncallable_wTrans['ytm'].corr(df_BBuniverse_BBnoncallable_wTrans['happening'])
df_BBuniverse_BBnoncallable_wTrans['ytm'].corr(df_BBuniverse_BBnoncallable_wTrans['worried'])
df_BBuniverse_BBnoncallable_wTrans['ytm'].corr(df_BBuniverse_BBnoncallable_wTrans['harmUS'])
df_BBuniverse_BBnoncallable_wTrans['ytm'].corr(df_BBuniverse_BBnoncallable_wTrans['CO2limits'])

df_GBuniverse_GBcallable_wTrans['ytm'].corr(df_GBuniverse_GBcallable_wTrans['human'])
df_GBuniverse_GBcallable_wTrans['ytm'].corr(df_GBuniverse_GBcallable_wTrans['happening'])
df_GBuniverse_GBcallable_wTrans['ytm'].corr(df_GBuniverse_GBcallable_wTrans['worried'])
df_GBuniverse_GBcallable_wTrans['ytm'].corr(df_GBuniverse_GBcallable_wTrans['harmUS'])
df_GBuniverse_GBcallable_wTrans['ytm'].corr(df_GBuniverse_GBcallable_wTrans['CO2limits'])

df_BBuniverse_BBcallable_wTrans['ytm'].corr(df_BBuniverse_BBcallable_wTrans['human'])
df_BBuniverse_BBcallable_wTrans['ytm'].corr(df_BBuniverse_BBcallable_wTrans['happening'])
df_BBuniverse_BBcallable_wTrans['ytm'].corr(df_BBuniverse_BBcallable_wTrans['worried'])
df_BBuniverse_BBcallable_wTrans['ytm'].corr(df_BBuniverse_BBcallable_wTrans['harmUS'])
df_BBuniverse_BBcallable_wTrans['ytm'].corr(df_BBuniverse_BBcallable_wTrans['CO2limits'])

'''
[Statistical significance]
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

pearsonr(df_NonCalluniverse_Comb_wTrans['ytm'], df_NonCalluniverse_Comb_wTrans['human'])
pearsonr(df_NonCalluniverse_Comb_wTrans['ytm'], df_NonCalluniverse_Comb_wTrans['CO2limits'])

spearmanr(df_NonCalluniverse_Comb_wTrans['ytm'], df_NonCalluniverse_Comb_wTrans['human'])
spearmanr(df_NonCalluniverse_Comb_wTrans['ytm'], df_NonCalluniverse_Comb_wTrans['CO2limits'])

'''
# +++++

# [2DPLOT: COUNTY level] # Key: @*$#

# ytm
sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='human_county', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at county level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='human_county', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at county level)'); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='happening_county', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at county level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='happening_county', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at county level)'); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='worried_county', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at county level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='worried_county', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at county level)'); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='harmUS_county', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at county level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='harmUS_county', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at county level)'); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='ytm', y='CO2limits_county', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.title('non-callable universe (climate belief measured at county level)'); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='ytm', y='CO2limits_county', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.title('callable universe (climate belief measured at county level)'); plt.legend()

# ANNOUNCE
os.system('say "visualization is finished"')

#####################

# current_yield
sns.displot(df_NonCalluniverse_Comb_wTrans, x='current_yield', y='human', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='current_yield', y='human', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='current_yield', y='happening', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='current_yield', y='happening', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='current_yield', y='worried', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='current_yield', y='worried', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.legend()

sns.displot(df_NonCalluniverse_Comb_wTrans, x='current_yield', y='harmUS', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()
sns.displot(df_Calluniverse_Comb_wTrans, x='current_yield', y='harmUS', hue='municalltype', kind='kde', palette=['royalblue','orangered'], fill=False); plt.legend()

# ******
'''
# [2DPLOT: 2 types MIXEDCALL]
sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='dollar_price', hue='munitype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()
sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='current_yield', hue='munitype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()

# [2DPLOT: 2 types MIXEDCALL]
sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='human', hue='munitype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()
sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='harmUS', hue='munitype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()
sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='worried', hue='munitype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()

'''
# ******
# [2DPLOT: 4 types => HARD to interpret]
sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='human', hue='municalltype', kind='kde', palette=['royalblue','mediumseagreen','orangered','firebrick'], fill=False); plt.legend()

# ******


# CHECK: confounder can drive the relation x='ytm', y='harmUS' above
sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='originalissueamt', hue='munitype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()
sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='originalissueamt', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()

sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='coupon', hue='munitype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()
sns.displot(df_MUNIuniverse_wTrans, x='ytm', y='coupon', hue='municalltype', kind='kde', palette=['mediumseagreen','firebrick'], fill=False); plt.legend()

# ANNOUNCE
os.system('say "visualization is finished"')

#%% 


#%% Section 9 Choropleth => USE R script "UScounty_chroropleth.R" which can read geoid_int
'''
# Understanding Geographic Identifiers (GEOIDs): https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html
 => Harris County, TX is 48201

# https://arilamstein.com/us-county-choropleths/
# https://academy.datawrapper.de/article/115-how-to-import-data-choropleth-map

# Choropleth Maps in Python: https://plotly.com/python/choropleth-maps/
# https://towardsdatascience.com/a-step-by-step-guide-to-interactive-choropleth-map-in-python-681f6bd853ce

# Mapchart.net: https://mapchart.net/usa-counties.html
# How to Make a US County Thematic Map Using Free Tools: https://flowingdata.com/2009/11/12/how-to-make-a-us-county-thematic-map-using-free-tools/

'''

# CHANGE: 'DC':'Washington DC' => 'DC':'District of Columbia'
dic_states = {'AL':'Alabama', 'AK':'Alaska', 'AZ':'Arizona', 'AR':'Arkansas', 'CA':'California', 'CO':'Colorado',
'CT':'Connecticut', 'DC':'District of Columbia', 'DE':'Delaware', 'FL':'Florida', 'GA':'Georgia',
'HI':'Hawaii', 'ID':'Idaho', 'IL':'Illinois', 'IN':'Indiana', 'IA':'Iowa',
'KS':'Kansas', 'KY':'Kentucky', 'LA':'Louisiana', 'ME':'Maine', 'MD':'Maryland',
'MA':'Massachusetts', 'MI':'Michigan', 'MN':'Minnesota', 'MS':'Mississippi',
'MO':'Missouri', 'MT':'Montana', 'NE':'Nebraska', 'NV':'Nevada', 'NH':'New Hampshire',
'NJ':'New Jersey', 'NM':'New Mexico', 'NY':'New York', 'NC':'North Carolina',
'ND':'North Dakota', 'OH':'Ohio', 'OK':'Oklahoma', 'OR':'Oregon', 'PA':'Pennsylvania',
'RI':'Rhode Island', 'SC':'South Carolina', 'SD':'South Dakota', 'TN':'Tennessee',
'TX':'Texas', 'UT':'Utah', 'VT':'Vermont', 'VA':'Virginia', 'WA':'Washington', 'WV':'West Virginia',
'WI':'Wisconsin', 'WY':'Wyoming'}    

df_USstate_template = pd.DataFrame([dic_states]).T.reset_index().rename(columns={'index':'state', 0:'ID'})

# County Approach : preparating for converting County string to GEOID via FIPScode
# INPUT: df_fipscode, df_fipscode_stateonly
# OUTPUT: df_fipscode_countyonly[['geoid_int','U.S. County Of Issuance']]

fips_countyapproach = 0;
if (fips_countyapproach == 1) & (noncallable_couponmatching == 1):

    # NEW: FIPS code => township is more granular than county level
    df_fipscode = pd.read_excel('/Users/YUMA/Desktop/3rdpaper/Datasets/FIPS code - US Census/all-geocodes-v2018.xlsx', sheet_name='v2018geocodes', skiprows=4) 
    
    # NOTE: R requires int
    df_fipscode['geoid_int'] = ( df_fipscode['State Code (FIPS)'].astype(str) + df_fipscode['County Code (FIPS)'].astype(str).str.zfill(3) ).astype(int)
    
    # CHECK
    #df_fipscode['State Code (FIPS)'].describe()
    
    # REDUCE: FIPS = 72 in df_fipscode_countyonly is Puerto Rico etc. => dissapears
    df_fipscode_countyonly = df_fipscode[ df_fipscode['Area Name (including legal/statistical area description)'].str.contains('County') ]
    
    # CHECK
    #df_fipscode_countyonly['State Code (FIPS)'].describe()
    
    # NOTE: can conclude that county subdivision code > 0 only for city, town etc.
    #df_fipscode_countyonly['County Subdivision Code (FIPS)'].describe()
    #df_fipscode_countyonly['County Code (FIPS)'].describe()
    
    # CHECK
    if check_dataset == 1:
        df_geoid_int_check = pd.merge( df_fipscode[['geoid_int','Area Name (including legal/statistical area description)']], df_climateopinion2016_county[['geoid_int','State','County']], on=['geoid_int'] )
        df_geoid_int_check[ df_geoid_int_check['County'] == 'Harlan County' ]
    
    # [ STATEFIPS for attaching state name to county name via State FIPS code ]
    # NEW: State FIPS code
    df_fipscode_stateonly = pd.read_excel('/Users/YUMA/Desktop/3rdpaper/Datasets/FIPS code - US Census/state-geocodes-v2018.xlsx', sheet_name='CODES14', skiprows=4, header=1) 
    
    # REDUCE: remove XXX region e.g. Northeast Region
    df_fipscode_stateonly = df_fipscode_stateonly[ df_fipscode_stateonly['State (FIPS)'] != 0 ][['State (FIPS)','Name']].reset_index().drop(columns=['index'])    
    
    # TRANSFORM
    dic_states_reversed = {value : key for (key, value) in dic_states.items()}
    df_fipscode_stateonly['state2'] = df_fipscode_stateonly['Name'].map(dic_states_reversed)
    
    # MERGE: attaching state name via State Code (FIPS)
    df_fipscode_countyonly = df_fipscode_countyonly.merge( df_fipscode_stateonly.rename(columns={'State (FIPS)':'State Code (FIPS)'}), on=['State Code (FIPS)'] )
    
    # NEW columns
    df_fipscode_countyonly['U.S. County Of Issuance'] = df_fipscode_countyonly['Area Name (including legal/statistical area description)'].str.replace(" County",", ") + df_fipscode_countyonly['state2']
    

# NEW child: BB universe is a result of matching and not of primary interest => only GB focus
df_Rplot_GB = df_GBuniverse_Comb_wTrans[['cusip','cusip6','U.S. County Of Issuance','state']].drop_duplicates()

# CHECK: w/o 'state'
#df_GBuniverse_Comb_wTrans[['cusip','cusip6','U.S. County Of Issuance']].drop_duplicates()

# TRANSFORM: AL => Alabama
#df_Rplot_GB_county_sumcount['U.S. County Of Issuance'] = df_Rplot_GB_county_sumcount['U.S. County Of Issuance'].str[:-3] + ' ' + df_Rplot_GB_county_sumcount['U.S. County Of Issuance'].str[-2:].map(dic_states)

# STEP 2: converting County string to GEOID via FIPScode
if (fips_countyapproach == 1) & (noncallable_couponmatching == 1):

    # NEW child: County level
    df_Rplot_GB_county_sumcount = df_Rplot_GB[['cusip','U.S. County Of Issuance']].groupby(['U.S. County Of Issuance']).count().reset_index().rename(columns={'cusip':'num_issuance'})

    # MERGE: NOT how='left'
    df_Rplot_GB_county_sumcount = df_Rplot_GB_county_sumcount.merge( df_fipscode_countyonly[['geoid_int','U.S. County Of Issuance']], on=['U.S. County Of Issuance'] )

    # EXPORT
    df_Rplot_GB_county_sumcount.drop(columns='U.S. County Of Issuance').rename(columns={'geoid_int':'region', 'num_issuance':'value'}).to_csv( directory_output + 'Rplot/df_Rplot_GB_county_sumcount.csv', index=False)
    
    # CHECK: how many dropped?
    len(df_Rplot_GB_county_sumcount)
    len(df_GBuniverse_Comb_wTrans['cusip6'].unique())

# ************* ************* ************* ************* ************* ************* ************* ************* #
# Online approach: https://app.datawrapper.de/map/5rMDq/data#

# State approach
if (noncallable_couponmatching == 1):
    
    # NOTE: Do NOT use US count of issue (i.e., 'State') but use instead (i.e., fidelity 'state') to get state-level data
    df_Rplot_GB_state_sumcount = df_Rplot_GB[['cusip','state']].groupby(['state']).count().reset_index().rename(columns={'cusip':'num_issuance'})
    
    # xxxTRANSFORM
    #df_Rplot_GB_state_sumcount['region'] = df_Rplot_GB_state_sumcount['state'].map(dic_states).str.lower()
    
    # MERGE: attach to template => how='left' is important so that NA (black) does not show in R
    df_USstate_template = df_USstate_template.merge( df_Rplot_GB_state_sumcount, on=['state'], how='left' )

    # TRANSFORM
    df_USstate_template['region'] = df_USstate_template['ID'].str.lower()
    df_USstate_template['value'] = df_USstate_template['num_issuance'].replace(np.nan,0)
    
    # EXPORT
    df_USstate_template[['region','value']].to_csv( directory_output + 'Rplot/df_Rplot_GB_state_sumcount.csv', index=False)    

#%%

# [BBG]
'''
# Add Field formula to Column name using dictionary
dic_BBGColumnField = {"CUSIP":"ID_CUSIP", "State":"STATE_CODE",	"Cpn":"CPN", "Dated Date	":"MUNI_DATED_DT", "Maturity	":"MATURITY", 
                      "Curr Yld (Ask)":"YLD_CUR_ASK", "Curr Yld (Bid)":"YLD_CUR_BID", "Curr Yld (Mid)":"YLD_CUR_MID", "Fed Tx Dtl":"FEDERAL_TAX_DETAIL", 
                      "Has Call Provision":"CALLABLE", "ISIN":"ID_ISIN", "Issuer":"ISSUER", 
                      "S&P Long Term Rating":"RTG_SP_LONG", "S&P Long Term Rating Date - Issue Level":"RTG_SP_LONG_RATING_DT", "Fitch Long Term Rating":"RTG_FITCH_LONG", "Moody's Long Term Rating":"RTG_MOODY_LONG_ISSUE_LEVEL", 
                      "Tax Prov":"MUNI_TAX_PROV", "Yield at Issue":"YIELD_ON_ISSUE_DATE",
                      "Base CUSIP":"ISSUER_CODE", "Coupon Type":"CPN_TYP", "Insurer":"INSURERS", "Issue Date":"ISSUE_DT",}

reversed_dictionary = {value : key for (key, value) in dic_BBGColumnField.items()}


# no formula: Issuer Name, Mty Size, Amt Out


# intersection: http://www.datasciencemadesimple.com/intersection-two-dataframe-pandas-python-2/

intersected_df = pd.merge(df_BBG_Green['CUSIP'], dic_BBGColumnField['CUSIP'], how='inner')

'''
# [MSRB commands]
'''
['rtrs_control_number', 'trade_type_indicator', 'cusip',
       'security_description', 'dated_date', 'coupon', 'maturity_date',
       'when_issued_indicator', 'assumed_settlement_date', 'trade_date',
       'time_of_trade', 'settlement_date', 'par_traded', 'dollar_price',
       'yield', 'brokers_broker_indicator', 'weighted_price_indicator',
       'offer_price_takedown_indicator', 'rtrs_publish_date',
       'rtrs_publish_time', 'version_number', 'uv_dollar_price_indicator',
       'ats_indicator', 'ntbc_indicator']

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #

# FOLLOW: https://wizardkingz.github.io/wrdsdataaccesspython-tutorial/
# R – Retrieving data from many CUSIPs using a SQL query on the WRDS MSRB dataset: https://stackoverflow.com/questions/64199024/retrieving-data-from-many-cusips-using-a-sql-query-on-the-wrds-msrb-dataset

# For a table with not too many rows (like some of the linktables) you can use get_table(), 
# Otherwise raw_sql()is advisable: https://arc.eaa-online.org/blog/retrieving-data-wrds-directly-using-python-r-and-stata


db.list_tables(library='msrb') # db.list_libraries()
db.describe_table(library='msrb', table='msrb')
db.describe_table(library='msrb', table='msrb_lookup')
db.describe_table(library='msrb', table='msrb_qvards')

db.get_table(library='msrb', table='msrb', obs=1000)
db.raw_sql('SELECT * FROM msrb.msrb limit 10')


db.raw_sql("SELECT cusip, dated_date, coupon, maturity_date, yield, offer_price_takedown_indicator \
           FROM msrb.msrb \
           WHERE cusip \
           IN ('040688HR7','040688LK7','040688LH4')")

db.raw_sql("SELECT cusip, dated_date, coupon, maturity_date, yield, offer_price_takedown_indicator \
           FROM msrb.msrb \
           WHERE cusip \
           LIKE '040688___'") # three characters

############

for cusip6dig in GBuniverse_CUSIPlist:
    print("Now cusip6dig is: " + cusip6dig + "\n")
    
    #, yield, offer_price_takedown_indicator
    temporary = db.raw_sql("SELECT cusip, dated_date, coupon, maturity_date \
               FROM msrb.msrb \
               WHERE cusip \
               LIKE '" + str(cusip6dig) + "___' \
               AND dated_date \
               IN ('2019-05-14')")
    
    df_BBuniverse = df_BBuniverse.append(temporary)

    count += 1
    
    if count > 3:
        break

'''

# [OLD: MSRB filtering from BBG info]
'''
# IMPORT
#fields_GBlist = ['']
df_GBuniverse = pd.read_excel(os.path.join(directory_input,'GREENINDICATOR_INCOMPLETE_BUTMORELABELS.xlsx'), sheet_name='Municipals (2)') #, usecols=fields_GBlist

df_GBuniverse = df_GBuniverse[ df_GBuniverse['Call Feature'] == '#N/A Field Not Applicable' ] # DROP: anytime, annual, semi-annual
df_GBuniverse = df_GBuniverse[ df_GBuniverse['Coupon Type'] != 'FLOATING' ]

# CHECK: OK
df_GBuniverse[ df_GBuniverse['Dated Date'] != df_GBuniverse['Issue Date'] ]

# EXPAND
#df_GBuniverse[['DatedDate_Day','DatedDate_Month','DatedDate_Year']] = df_GBuniverse['Dated Date'].str.split('.',expand=True,)
#df_GBuniverse[['Maturity_Day','Maturity_Month','Maturity_Year']] = df_GBuniverse['Maturity'].str.split('.',expand=True,)

df_GBuniverse["DatedDateInt"] = ( df_GBuniverse["Dated Date"].str[6:10] + df_GBuniverse["Dated Date"].str[3:5] + df_GBuniverse["Dated Date"].str[:2] ).astype(int)
df_GBuniverse["DatedDateHyphen"] = ( df_GBuniverse["Dated Date"].str[6:10] + "-" + df_GBuniverse["Dated Date"].str[3:5] + "-" + df_GBuniverse["Dated Date"].str[:2] )

df_GBuniverse["MaturityInt"] = ( df_GBuniverse["Maturity"].str[6:10] + df_GBuniverse["Maturity"].str[3:5] + df_GBuniverse["Maturity"].str[:2] ).astype(int)
df_GBuniverse["MaturityHyphen"] = ( df_GBuniverse["Maturity"].str[6:10] + "-" + df_GBuniverse["Maturity"].str[3:5] + "-" + df_GBuniverse["Maturity"].str[:2] )

df_GBuniverse["Maturity_rounded"] = round( (df_GBuniverse["MaturityInt"] - df_GBuniverse["DatedDateInt"])/10000, 0 )

excl_preJune2013 = 1;
if excl_preJune2013 == 1:
    df_GBuniverse = df_GBuniverse[ df_GBuniverse["DatedDateInt"] > 20130600 ] 
    
'''

# [Another approach to overlaying histrogram]
'''
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20150300) & (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20181215) ][TargetVar]
y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20150300) & (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20181215) ][TargetVar]

x_w = np.empty(x.shape)
x_w.fill(1/x.shape[0])
y_w = np.empty(y.shape)
y_w.fill(1/y.shape[0])
bins = np.linspace(0, 6, 30)

plt.hist([x, y], bins, weights=[x_w, y_w], label=['x', 'y'])
plt.legend(loc='upper right')
plt.show()

########

sexam( df_GBuniverse_GBnoncallable_wTrans[TargetVar] )
sexam( df_GBuniverse_GBnoncallable_wTrans[ df_GBuniverse_GBnoncallable_wTrans['when_issued_indicator'] == 'Y' ][TargetVar] )
sexam( df_BBuniverse_BBnoncallable_wTrans[TargetVar] )
sexam( df_GBuniverse_GBcallable_wTrans[TargetVar] )
sexam( df_BBuniverse_BBcallable_wTrans[TargetVar] )

'''

# [Unimportant cases: differential]
'''
# ------------ #
# Corona: placebo8 combining CA NY MA AZ WA:
x = df_GBuniverse_GBnoncallable_wTrans[ ((df_GBuniverse_GBnoncallable_wTrans['State'] == 'CA') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'NY') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'MA') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'AZ') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'WA')) ][TargetVar]
y = df_BBuniverse_BBnoncallable_wTrans[ ((df_BBuniverse_BBnoncallable_wTrans['State'] == 'CA') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'NY') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'MA') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'AZ') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'WA')) ][TargetVar]

x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20191215) & ((df_GBuniverse_GBnoncallable_wTrans['State'] == 'CA') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'NY') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'MA') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'AZ') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'WA')) ][TargetVar]
y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20191215) & ((df_BBuniverse_BBnoncallable_wTrans['State'] == 'CA') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'NY') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'MA') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'AZ') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'WA')) ][TargetVar]

x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] > 20190315) & ((df_GBuniverse_GBnoncallable_wTrans['State'] == 'CA') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'NY') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'MA') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'AZ') | (df_GBuniverse_GBnoncallable_wTrans['State'] == 'WA')) ][TargetVar]
y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] > 20190315) & ((df_BBuniverse_BBnoncallable_wTrans['State'] == 'CA') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'NY') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'MA') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'AZ') | (df_BBuniverse_BBnoncallable_wTrans['State'] == 'WA')) ][TargetVar]

# ------------ #

# Corona: placebo9 other than CA NY MA AZ WA:
x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['State'] != 'CA') & (df_GBuniverse_GBnoncallable_wTrans['State'] != 'NY') & (df_GBuniverse_GBnoncallable_wTrans['State'] != 'MA') & (df_GBuniverse_GBnoncallable_wTrans['State'] != 'AZ') & (df_GBuniverse_GBnoncallable_wTrans['State'] != 'WA') ][TargetVar]
y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['State'] != 'CA') & (df_BBuniverse_BBnoncallable_wTrans['State'] != 'NY') & (df_BBuniverse_BBnoncallable_wTrans['State'] != 'MA') & (df_BBuniverse_BBnoncallable_wTrans['State'] != 'AZ') & (df_BBuniverse_BBnoncallable_wTrans['State'] != 'WA') ][TargetVar]

x = df_GBuniverse_GBnoncallable_wTrans[ (df_GBuniverse_GBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_GBuniverse_GBnoncallable_wTrans['State'] != 'CA') & (df_GBuniverse_GBnoncallable_wTrans['State'] != 'NY') & (df_GBuniverse_GBnoncallable_wTrans['State'] != 'MA') & (df_GBuniverse_GBnoncallable_wTrans['State'] != 'AZ') & (df_GBuniverse_GBnoncallable_wTrans['State'] != 'WA') ][TargetVar]
y = df_BBuniverse_BBnoncallable_wTrans[ (df_BBuniverse_BBnoncallable_wTrans['trade_date_int'] < 20191215) & (df_BBuniverse_BBnoncallable_wTrans['State'] != 'CA') & (df_BBuniverse_BBnoncallable_wTrans['State'] != 'NY') & (df_BBuniverse_BBnoncallable_wTrans['State'] != 'MA') & (df_BBuniverse_BBnoncallable_wTrans['State'] != 'AZ') & (df_BBuniverse_BBnoncallable_wTrans['State'] != 'WA') ][TargetVar]

'''

# [kernel density: differential]
# 1D Kernel Density Estimation: https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
# Density-difference estimation: https://dl.acm.org/doi/10.1162/NECO_a_00492

# Kernel Bandwidth: Scott's vs. Silverman's rules: https://stats.stackexchange.com/questions/90656/kernel-bandwidth-scotts-vs-silvermans-rules
# Silverman's rules: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

'''
def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

x = make_data(1000)

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])


x_d = np.linspace(-4, 8, 1000)


logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)

'''
# REDUCE D: rating matching
'''
# @@@ Rating matching: required before comparison => Nan vs Nan also preserved in this setting, which is FINE @@@
df_GBuniverse_GBnoncallable['cusip6moodys'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['moodyscurrent'].astype(str)
df_GBuniverse_GBnoncallable['cusip6sp'] = df_GBuniverse_GBnoncallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBnoncallable['spcurrent'].astype(str)
df_GBuniverse_GBcallable['cusip6moodys'] = df_GBuniverse_GBcallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBcallable['moodyscurrent'].astype(str)
df_GBuniverse_GBcallable['cusip6sp'] = df_GBuniverse_GBcallable['cusip6'].astype(str) + '_' + df_GBuniverse_GBcallable['spcurrent'].astype(str)

df_BBuniverse_BBnoncallable['cusip6moodys'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['moodyscurrent'].astype(str)
df_BBuniverse_BBnoncallable['cusip6sp'] = df_BBuniverse_BBnoncallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBnoncallable['spcurrent'].astype(str)
df_BBuniverse_BBcallable['cusip6moodys'] = df_BBuniverse_BBcallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBcallable['moodyscurrent'].astype(str)
df_BBuniverse_BBcallable['cusip6sp'] = df_BBuniverse_BBcallable['cusip6'].astype(str) + '_' + df_BBuniverse_BBcallable['spcurrent'].astype(str)


ratingOpeANDOR = 1; 
if ratingOpeANDOR == 1:
    df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable[ df_GBuniverse_GBnoncallable['cusip6moodys'].isin( df_BBuniverse_BBnoncallable['cusip6moodys'] ) & df_GBuniverse_GBnoncallable['cusip6sp'].isin( df_BBuniverse_BBnoncallable['cusip6sp'] ) ]
    df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable[ df_BBuniverse_BBnoncallable['cusip6moodys'].isin( df_GBuniverse_GBnoncallable['cusip6moodys'] ) & df_BBuniverse_BBnoncallable['cusip6sp'].isin( df_GBuniverse_GBnoncallable['cusip6sp'] ) ]
    df_GBuniverse_GBcallable = df_GBuniverse_GBcallable[ df_GBuniverse_GBcallable['cusip6moodys'].isin( df_BBuniverse_BBcallable['cusip6moodys'] ) & df_GBuniverse_GBcallable['cusip6sp'].isin( df_BBuniverse_BBcallable['cusip6sp'] ) ]
    df_BBuniverse_BBcallable = df_BBuniverse_BBcallable[ df_BBuniverse_BBcallable['cusip6moodys'].isin( df_GBuniverse_GBcallable['cusip6moodys'] ) & df_BBuniverse_BBcallable['cusip6sp'].isin( df_GBuniverse_GBcallable['cusip6sp'] ) ]

elif ratingOpeANDOR == 0:
    df_GBuniverse_GBnoncallable = df_GBuniverse_GBnoncallable[ df_GBuniverse_GBnoncallable['cusip6moodys'].isin( df_BBuniverse_BBnoncallable['cusip6moodys'] ) | df_GBuniverse_GBnoncallable['cusip6sp'].isin( df_BBuniverse_BBnoncallable['cusip6sp'] ) ]
    df_BBuniverse_BBnoncallable = df_BBuniverse_BBnoncallable[ df_BBuniverse_BBnoncallable['cusip6moodys'].isin( df_GBuniverse_GBnoncallable['cusip6moodys'] ) | df_BBuniverse_BBnoncallable['cusip6sp'].isin( df_GBuniverse_GBnoncallable['cusip6sp'] ) ]
    df_GBuniverse_GBcallable = df_GBuniverse_GBcallable[ df_GBuniverse_GBcallable['cusip6moodys'].isin( df_BBuniverse_BBcallable['cusip6moodys'] ) | df_GBuniverse_GBcallable['cusip6sp'].isin( df_BBuniverse_BBcallable['cusip6sp'] ) ]
    df_BBuniverse_BBcallable = df_BBuniverse_BBcallable[ df_BBuniverse_BBcallable['cusip6moodys'].isin( df_GBuniverse_GBcallable['cusip6moodys'] ) | df_BBuniverse_BBcallable['cusip6sp'].isin( df_GBuniverse_GBcallable['cusip6sp'] ) ]

elif ratingOpeANDOR == 2: # 
    pass

'''