#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 7 12:06:52 2023

@author: Adolfo Molada Tebar; TIDOP Research Group. USAL.

Description: Python script to compute the ANOVA analysis for the BRDF paper
    
Last modified on Thue Oct 19 12:31:00 2023
    
"""
import os

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sns
import pingouin as pg

#%%
# working environment
# ------------------- 
current_dir = os.getcwd()

path_data_dbf = os.path.join(current_dir, "data", "dbf")   # original shp data from dbf attributes file

#%%
# READ, FILTERING & MERGE into a DataFrame
# ----------------------------------------
path_file_filters = os.path.join(path_data_dbf, "Filters.dbf")
path_file_no_filters = os.path.join(path_data_dbf, "noFilters.dbf")

# Read DBF using geopandas as DataFrame
filters_original = gpd.read_file(path_file_filters)
#print(filters_original.describe())
no_filters_original = gpd.read_file(path_file_no_filters)
#print(no_filters_original.describe())

# Filtering data  (seelct columns)
# --------------------------------
rf_col_names = ["Codigo", "rf_blue", "rf_blue_n", "rf_blue_s", "rf_green", "rf_green_n", "rf_green_s", 
            "rf_nir", "rf_nir_n", "rf_nir_s", "rf_red", "rf_red_n", "rf_red_s", "rf_redeg", "rf_redeg_n",
            "rf_redeg_s"]

filters_original = filters_original[rf_col_names]
#print(filters_original.describe())

om_col_names = ["Codigo","om_1", "om_1_n", "om_1_s", "om_2", "om_2_n", "om_2_s", "om_3", "om_3_n", 
             "om_3_s", "om_4", "om_4_n", "om_4_s", "om_5", "om_5_n", "om_5_s"]

no_filters_original = no_filters_original[om_col_names]
# rename col names
no_filters_original.rename(columns={"om_1": "om_blue",  "om_1_n": "om_blue_n",  "om_1_s": "om_blue_s",
                           "om_2": "om_green", "om_2_n": "om_green_n", "om_2_s": "om_green_s",
                           "om_3": "om_red",   "om_3_n": "om_red_n",   "om_3_s": "om_red_s", 
                           "om_4": "om_redeg", "om_4_n": "om_redeg_n", "om_4_s": "om_redeg_s",
                           "om_5": "om_nir",   "om_5_n": "om_nir_n",   "om_5_s": "om_nir_s"}, inplace=True)
#print(no_filters_original.describe())

# Merge into a DataFrame 
# ----------------------
cepas_data = filters_original.merge(no_filters_original, how='inner', on='Codigo')
print(cepas_data.describe())

num_cepas = cepas_data["Codigo"].count()
print("Total cepas = ", num_cepas) # 510 cepas, band, band_n, band_s data

#%%
# Search for cepas with negative reflectance (rf band data)
neg_reflectance = cepas_data.loc[(cepas_data['rf_red'] <0) | (cepas_data['rf_green'] <0) | (cepas_data['rf_blue'] <0) 
                                 | (cepas_data['rf_redeg'] <0) | (cepas_data['rf_nir'] <0) | (cepas_data['om_red'] <0) 
                                 | (cepas_data['om_red'] <0) | (cepas_data['om_blue'] <0) | (cepas_data['om_redeg'] <0) 
                                 | (cepas_data['om_nir'] <0) ]

num_cepas_neg = neg_reflectance["Codigo"].count()
print("Total cepas with negative reflectance = ", num_cepas_neg) # 510 cepas
cepa_neg_reflectance_ids = list(neg_reflectance["Codigo"])
print("Codigo (id) cepas with neg values: ", cepa_neg_reflectance_ids)

#%%
# REMOVE CEPAS WITH NEG REFLECTANCE
# ---------------------------------

cepas_non_neg = cepas_data.loc[(cepas_data['rf_red'] >0) & (cepas_data['rf_green'] >0) & (cepas_data['rf_blue'] >0) 
                                & (cepas_data['rf_redeg'] >0) & (cepas_data['rf_nir'] >0) & (cepas_data['om_red'] >0) 
                                & (cepas_data['om_red'] >0) & (cepas_data['om_blue'] >0) & (cepas_data['om_redeg'] >0) 
                                & (cepas_data['om_nir'] >0) ]

print(cepas_non_neg.describe())

num_cepas_non_neg = cepas_non_neg["Codigo"].count()
print("Total cepas non neg = ", num_cepas_non_neg) # 507 cepas

#%%
# Plot Histo non negative rf band data (Visual testing, neg values were removed)
cepas_non_neg["rf_red"].plot.hist(bins=150, color="red", alpha=0.5, label="R")
cepas_non_neg["rf_green"].plot.hist(bins=150, color="green", alpha=0.5, label="G")
cepas_non_neg["rf_blue"].plot.hist(bins=150, color="blue", alpha=0.5, label="B")
cepas_non_neg["rf_redeg"].plot.hist(bins=150, color="cyan", alpha=0.5, label="RE")
cepas_non_neg["rf_nir"].plot.hist(bins=150, color="salmon", alpha=0.5, label="Nir")
#plt.title("$BRDF$")

# band data
rf_red = cepas_non_neg["rf_red"]
rf_green = cepas_non_neg["rf_green"]
rf_blue = cepas_non_neg["rf_blue"]
rf_redeg = cepas_non_neg["rf_redeg"]
rf_nir = cepas_non_neg["rf_nir"]

mu_red, sigma_red = stats.norm.fit(rf_red)
mu_green, sigma_green = stats.norm.fit(rf_green)
mu_blue, sigma_blue = stats.norm.fit(rf_blue)
mu_redeg, sigma_redeg = stats.norm.fit(rf_redeg)
mu_nir, sigma_nir = stats.norm.fit(rf_nir)

x_hat = np.linspace(0, 1, num=100)
y_hat_red = stats.norm.pdf(x_hat, mu_red, sigma_red)
y_hat_green = stats.norm.pdf(x_hat, mu_green, sigma_green)
y_hat_blue = stats.norm.pdf(x_hat, mu_blue, sigma_blue)
y_hat_redeg = stats.norm.pdf(x_hat, mu_redeg, sigma_redeg)
y_hat_nir = stats.norm.pdf(x_hat, mu_nir, sigma_nir)

plt.plot(x_hat, y_hat_red, linewidth=2, color="red")#, label='normal')
plt.plot(x_hat, y_hat_green, linewidth=2, color="green")#, label='normal')
plt.plot(x_hat, y_hat_blue, linewidth=2, color="blue")#, label='normal')
plt.plot(x_hat, y_hat_redeg, linewidth=2, color="cyan")#, label='normal')
plt.plot(x_hat, y_hat_nir, linewidth=2, color="salmon")#, label='normal')

plt.xlabel("$reflectance$", linespacing=4)
plt.legend()
plt.show()

#%%
# Plot Histo om original band data
cepas_data["om_red"].plot.hist(bins=150, color="red", alpha=0.5, label="R")
cepas_data["om_green"].plot.hist(bins=150, color="green", alpha=0.5, label="G")
cepas_data["om_blue"].plot.hist(bins=150, color="blue", alpha=0.5, label="B")
cepas_data["om_redeg"].plot.hist(bins=150, color="cyan", alpha=0.5, label="RE")
cepas_data["om_nir"].plot.hist(bins=150, color="salmon", alpha=0.5, label="Nir")

# band data
om_red = cepas_non_neg["om_red"]
om_green = cepas_non_neg["om_green"]
om_blue = cepas_non_neg["om_blue"]
om_redeg = cepas_non_neg["om_redeg"]
om_nir = cepas_non_neg["om_nir"]

mu_red, sigma_red = stats.norm.fit(om_red)
mu_green, sigma_green = stats.norm.fit(om_green)
mu_blue, sigma_blue = stats.norm.fit(om_blue)
mu_redeg, sigma_redeg = stats.norm.fit(om_redeg)
mu_nir, sigma_nir = stats.norm.fit(om_nir)

x_hat = np.linspace(0, 1, num=100)
y_hat_red = stats.norm.pdf(x_hat, mu_red, sigma_red)
y_hat_green = stats.norm.pdf(x_hat, mu_green, sigma_green)
y_hat_blue = stats.norm.pdf(x_hat, mu_blue, sigma_blue)
y_hat_redeg = stats.norm.pdf(x_hat, mu_redeg, sigma_redeg)
y_hat_nir = stats.norm.pdf(x_hat, mu_nir, sigma_nir)

plt.plot(x_hat, y_hat_red, linewidth=2, color="red")#, label='normal')
plt.plot(x_hat, y_hat_green, linewidth=2, color="green")#, label='normal')
plt.plot(x_hat, y_hat_blue, linewidth=2, color="blue")#, label='normal')
plt.plot(x_hat, y_hat_redeg, linewidth=2, color="cyan")#, label='normal')
plt.plot(x_hat, y_hat_nir, linewidth=2, color="salmon")#, label='normal')

#plt.title("$Metashape$")
plt.xlabel("$reflectance$", linespacing=4)
plt.legend()
plt.show()

#%% 
# Compute NDVI
# ------------
def compute_ndvi(nir_value, red_value):
    ndvi_value = (nir_value - red_value)/(nir_value + red_value)
    return ndvi_value

#%%
# Apply to a DataFrame
cepas_non_neg["NDVI_BRDF"] = cepas_non_neg.apply(lambda x: compute_ndvi(x["rf_nir"], x["rf_red"]), axis=1)
cepas_non_neg["NDVI_Meta"] = cepas_non_neg.apply(lambda x: compute_ndvi(x["om_nir"], x["om_red"]), axis=1)

#%%
# Plot Histo NDVI data.
cepas_non_neg["NDVI_BRDF"].plot.hist(bins=150, color="darkgreen", alpha=0.6, label="BRDF")
cepas_non_neg["NDVI_Meta"].plot.hist(bins=150, color="limegreen", alpha=0.6, label="Metashape")
plt.xlabel("$NDVI$", linespacing=4)

# Add theoretical normal distribution
NDVI_BRDF = cepas_non_neg["NDVI_BRDF"]
NDVI_Meta = cepas_non_neg["NDVI_Meta"] 

mu_brdf, sigma_brdf = stats.norm.fit(NDVI_BRDF)
mu_meta, sigma_meta = stats.norm.fit(NDVI_Meta)

x_hat = np.linspace(-0.2, 0.9, num=100)
y_hat_brdf = stats.norm.pdf(x_hat, mu_brdf, sigma_brdf)
y_hat_meta = stats.norm.pdf(x_hat, mu_meta, sigma_meta)

plt.plot(x_hat, y_hat_brdf, linewidth=2, color="darkgreen")#, label='normal')
plt.plot(x_hat, y_hat_meta, linewidth=2, color="limegreen")#, label='normal')

plt.legend()
output_path = os.path.join(path_output_png, "Histo_NDVI.png")
plt.savefig(output_path, dpi=300)
plt.show()

#%%
print(cepas_non_neg[["NDVI_BRDF", "NDVI_Meta"]].describe())

#%%
# ANOVA
# -----

# 1) Assumption of normality: Normality test: grahical, analytical, hypothesis testing
# 2) Homogeneity of variance: Homoscedasticity test: Levene, Barlett, Fligner
# 3) Assumption of independence: BRDF / Metashape are independent methods

#%%
# Preparing data for analysis
# ---------------------------

# New Empty DataFrame
col_names = ["Method", "Method_Band", "Band", "cepa_id", "reflectance"]
data_anova = pd.DataFrame(columns=col_names) 

col_names_ndvi = ["Method", "cepa_id", "NDVI"]
data_anova_ndvi = pd.DataFrame(columns=col_names_ndvi) 

#%%
col_names = ["Codigo", "rf_red", "rf_green", "rf_blue", "rf_redeg", "rf_nir", "om_red", "om_green", "om_blue","om_redeg","om_nir","NDVI_BRDF", "NDVI_Meta"]
data_to_test = cepas_non_neg[col_names] # 507 cepas
print(data_to_test.describe()) 

#%%
# Fill DataFrame
band_names = ["rf_red", "rf_green", "rf_blue", "rf_redeg", "rf_nir", "om_red", "om_green", "om_blue","om_redeg","om_nir","NDVI_BRDF", "NDVI_Meta"]

cepa_normal_id = sorted(list(data_to_test["Codigo"]))

for cepa_id in cepa_normal_id:
    cepa_data = np.array(data_to_test[data_to_test["Codigo"]==cepa_id][band_names]).reshape(len(band_names))
    data_anova.loc[len(data_anova.index)] = ['BRDF',"M11","R", cepa_id, cepa_data[0]] 
    data_anova.loc[len(data_anova.index)] = ['BRDF',"M12","G", cepa_id, cepa_data[1]] 
    data_anova.loc[len(data_anova.index)] = ['BRDF',"M13","B", cepa_id, cepa_data[2]]
    data_anova.loc[len(data_anova.index)] = ['BRDF',"M14","RE", cepa_id, cepa_data[3]] 
    data_anova.loc[len(data_anova.index)] = ['BRDF',"M15","Nir", cepa_id, cepa_data[4]]
    data_anova.loc[len(data_anova.index)] = ['Metashape',"M21","R", cepa_id, cepa_data[5]] 
    data_anova.loc[len(data_anova.index)] = ['Metashape',"M22","G", cepa_id, cepa_data[6]] 
    data_anova.loc[len(data_anova.index)] = ['Metashape',"M23","B", cepa_id, cepa_data[7]] 
    data_anova.loc[len(data_anova.index)] = ['Metashape',"M24","RE", cepa_id, cepa_data[8]] 
    data_anova.loc[len(data_anova.index)] = ['Metashape',"M25","Nir", cepa_id, cepa_data[9]]                                                                               

    data_anova_ndvi.loc[len(data_anova_ndvi.index)] = ['BRDF', cepa_id, cepa_data[10]] 
    data_anova_ndvi.loc[len(data_anova_ndvi.index)] = ['Metashape', cepa_id, cepa_data[11]]                                                                               

#%%
# Data, using reflectance data
print(data_anova)

#%%
# Stats
print(data_anova.groupby('Method')['reflectance'].agg(['mean', 'std']))
print(data_anova.groupby(['Method', "Band"])['reflectance'].agg(['mean', 'std']))

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.boxplot(x="Method", y="reflectance",  data=data_anova, ax=ax)
plt.show()

#%%
# Plots
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
color_pal = {"R": "red", "G": "green", "B":"blue", "RE": "cyan", "Nir": "salmon"}
sns.boxplot(x="Method", y="reflectance", hue='Band', data=data_anova, palette=color_pal, ax=ax)
plt.show()

#%%
# 1) NORMALITY TEST
# -----------------
M1_data = data_anova["reflectance"][data_anova["Method"]=="BRDF"]
M2_data = data_anova["reflectance"][data_anova["Method"]=="Metashape"]

print(pg.normality(data=M1_data)) # Shapiro-Wilk
print(pg.normality(data=M2_data))

# Note: Reflectance data fails normality test, so the ANOVA test cannot be used. 
# Instead, a non-parametric test is required.

#%%
# Testing normality using scipy
shapiro_statistic, shapiro_p_value = stats.shapiro(M1_data) # Shapiro-Wilk test
print(f"Shapiro-Wilk test: statistic = {shapiro_statistic}, p-value = {shapiro_p_value}")
    
shapiro_statistic, shapiro_p_value = stats.shapiro(M2_data) # Shapiro-Wilk test
print(f"Shapiro-Wilk test: statistic = {shapiro_statistic}, p-value = {shapiro_p_value}")

# Note: Same results were obtained

#%%
# Normality test plot: Quantile-Quantile plot. 
# https://pingouin-stats.org/build/html/generated/pingouin.qqplot.html#pingouin.qqplot
def display_qqplot(band_name, band_data):
    plt.figure(figsize=(8,7))
    pg.qqplot(band_data, dist='norm')#, sparams=(mean, std))
    fig_name = f"QQPlot_{band_name}.png"
    plt.show()

#%%
display_qqplot("BRDF Reflectance", M1_data)

#%%
display_qqplot("Metashape Reflectance", M2_data)

#%%
# Plot Histo with theoretical normal distribution for each om band
method_name = ["BRDF", "Metashape"]
method_data = [M1_data, M2_data]

color_list = ["blue", "cyan"]

for i in range(0, len(method_name)):
    band_name = method_name[i]
    band_data = method_data[i]

    # mean and std
    mu, sigma = stats.norm.fit(band_data)

    # Valores teóricos de la normal en el rango observado
    #x_hat = np.linspace(min(band_data), max(band_data), num=100)
    x_hat = np.linspace(0, 1, num=100)
    y_hat = stats.norm.pdf(x_hat, mu, sigma)
    # significance level values
    y_m = stats.norm.pdf(mu, mu, sigma)
    v1 = mu + sigma*2.5
    v2 = mu - sigma*2.5
    y_1 = stats.norm.pdf(v1, mu, sigma)
    y_2 = stats.norm.pdf(v2, mu, sigma)
    # Gráfico
    plt.figure(figsize=(7,4))
    plt.title(f"Histogram")
    plt.plot(x_hat, y_hat, linewidth=2)#, label='normal')
    # significance interval 
    plt.plot([v1, v1], [0, y_1], linewidth=1, color="r")
    plt.plot([mu, mu], [0, y_m], linewidth=1, color="b")
    plt.plot([v2, v2], [0, y_2], linewidth=1, color="r")
    
    plt.hist(x=band_data, density=True, bins=150, color=color_list[i], alpha=0.5)
    
    plt.xlabel(f"{band_name} band data")
    plt.ylabel("$frequency$")
    plt.show()

#%%
# In addition: Normality test for rf bands
#              ---------------------------

col_names = ["rf_red", "rf_green", "rf_blue", "rf_redeg", "rf_nir"]
for band_name in col_names:
    print(f"Normality test band: {band_name}")
    band_data = data_to_test[band_name]
    print(pg.normality(data=band_data)) # Test de normalidad Shapiro-Wilk
    display_qqplot(band_name, band_data)

# Note: better results for R2 when the outlayers are removed

#%%
#             Normality test for om bands
#             ---------------------------

col_names = ["om_red", "om_green", "om_blue","om_redeg","om_nir"]
for band_name in col_names:
    print(f"Normality test band: {band_name}")
    band_data = data_to_test[band_name]
    print(pg.normality(data=band_data))
    display_qqplot(band_name, band_data)

#%%
# Non-parametric
# --------------
# U test Mann-Whitney
mwu_test = pg.mwu(x=M1_data, y=M2_data, alternative='two-sided')
pg.print_table(mwu_test, floatfmt='.3f')

# Method to compare mean between two groups (Equivalent to T-Student for normal data) 
# H0 rejected, since p_value<0.05, i.e. a significance difference was found between groups

#%%
# Test H Kruskal-Wallis (ANOVA for non parametric data)
# BRDF vs Metashape
kruskal_test = pg.kruskal(data=data_anova, dv="reflectance", between="Method", detailed=True)
pg.print_table(kruskal_test, floatfmt='.3f')

# H0 rejected, since p_value<0.05, i.e. a significance difference was found between groups

#%% 
# Scipy

#%%
# U test Mann-Whitney
mann_statistic, p_value_mann = scipy.stats.mannwhitneyu(M1_data,M2_data) # Same result
print(mann_statistic, p_value_mann)

#%%
# Test H
kruskal_statistic, p_value_kruskal = stats.kruskal(M1_data, M2_data)
#kruskal_statistic, p_value_kruskal = stats.mstats.kruskalwallis(np.array(M1_data),np.array(M2_data)) # same result
print(kruskal_statistic, p_value_kruskal)

#%%
M11 = data_anova["reflectance"][data_anova["Method_Band"]=="M11"]
M12 = data_anova["reflectance"][data_anova["Method_Band"]=="M12"]
M13 = data_anova["reflectance"][data_anova["Method_Band"]=="M13"]
M14 = data_anova["reflectance"][data_anova["Method_Band"]=="M14"]
M15 = data_anova["reflectance"][data_anova["Method_Band"]=="M15"]

M21 = data_anova["reflectance"][data_anova["Method_Band"]=="M21"]
M22 = data_anova["reflectance"][data_anova["Method_Band"]=="M22"]
M23 = data_anova["reflectance"][data_anova["Method_Band"]=="M23"]
M24 = data_anova["reflectance"][data_anova["Method_Band"]=="M24"]
M25 = data_anova["reflectance"][data_anova["Method_Band"]=="M25"]                                

kruskal_statistic, p_value_kruskal = stats.kruskal(M11, M12, M13, M14, M15)#, M21, M22, M23, M24, M25)
#kruskal_statistic, p_value_kruskal = stats.mstats.kruskalwallis(np.array(M1_data),np.array(M2_data)) # same result
print(kruskal_statistic, p_value_kruskal)

#%%
# Yuen’s t-test
# Use the trim keyword to perform a trimmed (Yuen) t-test. For example, using 20% trimming, trim=.2, the test will reduce the impact of one (np.floor(trim*len(a))) element from each tail of sample a. It will have no effect on sample b because np.floor(trim*len(b)) is 0.
yuen_statistic, p_value_yuen = scipy.stats.ttest_ind(M1_data,M2_data, trim=0.2)
print(yuen_statistic, p_value_yuen)

#%%
# BRDF vs Metashape, using band data
kruskal_test = pg.kruskal(data=data_anova, dv="reflectance", between="Method_Band", detailed=True)
pg.print_table(kruskal_test, floatfmt='.3f')

# H0 rejected, as p_value<0.05, i.e. a significant difference was found at least in the means 
# between two groups.

#%%
# Bonferoni
bonferoni_test = pg.pairwise_tests(data=data_anova, dv='reflectance', between="Method_Band", parametric=False, 
                                   padjust='bonf', effsize='hedges')
pg.print_table(bonferoni_test, floatfmt='.3f')

# H1 for both methods, considering band data

#%%
# Same results using scipy
#print(stats.tukey_hsd(M11, M12, M13, M14, M15, M21, M22, M23, M24, M25)) # parametric

#%%
# Boxplot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.boxplot(x="method_band", y="reflectance", hue="method", data=data_anova, ax=ax)
output_path = os.path.join(path_output_png, "BoxPlot_reflectance_method_method_band.png")
plt.savefig(output_path, dpi=300)
plt.show()

#%%
# ANOVA for NDVI data
# -------------------

print(data_anova_ndvi)

#%%
# Stats
print(data_anova_ndvi.groupby('method')['NDVI'].agg(['mean', 'std']))

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.boxplot(x="Method", y="NDVI",  data=data_anova_ndvi, ax=ax)

#%%
# NORMALITY TEST
# ---------------
M1_ndvi_data = data_anova_ndvi["NDVI"][data_anova_ndvi["Method"]=="BRDF"]
M2_ndvi_data = data_anova_ndvi["NDVI"][data_anova_ndvi["Method"]=="Metashape"]

print(pg.normality(data=M1_ndvi_data))
print(pg.normality(data=M2_ndvi_data))

# Note: NDVI data fails normality test, so the parametric ANOVA test cannot be used. 
# Instead, a non-parametric test is required.

#%%
# Plot Histo
M1_ndvi_data.plot.hist(bins=150, color="purple", alpha=0.5, label="BRDF")
M2_ndvi_data.plot.hist(bins=150, color="salmon", alpha=0.5, label="Metashape")
plt.xlabel("$NDVI$", linespacing=4)
plt.legend()
plt.show()

#%%
display_qqplot("NDVI BRDF", M1_ndvi_data)

#%%
display_qqplot("NDVI Metashape", M2_ndvi_data)

#%%
# Plot Histo with theoretical normal distribution for each om band
method_name = ["BRDF", "Metashape"]
method_data = [M1_ndvi_data, M2_ndvi_data]

color_list = ["purple", "salmon"]

for i in range(0, len(method_name)):
    band_name = method_name[i]
    band_data = method_data[i]

    # mean and std
    mu, sigma = stats.norm.fit(band_data)

    x_hat = np.linspace(0, 1, num=100)
    y_hat = stats.norm.pdf(x_hat, mu, sigma)
    # significance level values
    y_m = stats.norm.pdf(mu, mu, sigma)
    v1 = mu + sigma*2.5
    v2 = mu - sigma*2.5
    y_1 = stats.norm.pdf(v1, mu, sigma)
    y_2 = stats.norm.pdf(v2, mu, sigma)
    
    plt.figure(figsize=(7,4))
    plt.title(f"Histogram")
    plt.plot(x_hat, y_hat, linewidth=2)#, label='normal')
    # significance interval 
    plt.plot([v1, v1], [0, y_1], linewidth=1, color="r")
    plt.plot([mu, mu], [0, y_m], linewidth=1, color="b")
    plt.plot([v2, v2], [0, y_2], linewidth=1, color="r")
    
    plt.hist(x=band_data, density=True, bins=150, color=color_list[i], alpha=0.5)
    
    plt.xlabel(f"NDVI {band_name} data")
    plt.ylabel("$frequency$")
    fig_name = f"Histo_{band_name}_nd.png"
    output_path = os.path.join(path_output_png, fig_name)
    plt.savefig(output_path, dpi=300)    
    plt.show()

#%%
# Non-parametric
# --------------
# U test Mann.Whitney
mwu_test = pg.mwu(x=M1_ndvi_data, y=M2_ndvi_data, alternative='two-sided')
pg.print_table(mwu_test, floatfmt='.3f')

# Method to compare mean between two groups (Equivalent to T-Student for normal data) 
# H0 rejected, since p_value<0.05, i.e. a significance difference was found between groups

#%%
# Test H Kruskal-Wallis (ANOVA for non parametric data)
# BRDF vs Metashape
kruskal_test = pg.kruskal(data=data_anova_ndvi, dv="NDVI", between="Method", detailed=True)
pg.print_table(kruskal_test, floatfmt='.3f')

# H0 rejected, since p_value<0.05, i.e. a significance difference was found between groups

#%%
yuen_statistic, p_value_yuen = scipy.stats.ttest_ind(M1_ndvi_data,M2_ndvi_data, trim=0.2)
print(yuen_statistic, p_value_yuen)

#%%
# Boxplot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.boxplot(x="Method", y="NDVI",  data=data_anova_ndvi, ax=ax)

#%%
# Difference Analysis

#%%
# function to compute difference for each cepa id between rf & om data for each band
def compute_difference(rf_value_band, om_value_band):
    #diff = (rf_value_band - om_value_band)
    diff = abs(rf_value_band - om_value_band)
    return diff

#%%
# Apply to a DataFrame
cepas_non_neg["R_diff"] = cepas_non_neg.apply(lambda x: compute_difference(x["om_red"], x["rf_red"]), axis=1)
cepas_non_neg["G_diff"] = cepas_non_neg.apply(lambda x: compute_difference(x["om_green"], x["rf_green"]), axis=1)
cepas_non_neg["B_diff"] = cepas_non_neg.apply(lambda x: compute_difference(x["om_blue"], x["rf_blue"]), axis=1)
cepas_non_neg["RE_diff"] = cepas_non_neg.apply(lambda x: compute_difference(x["om_redeg"], x["rf_redeg"]), axis=1)
cepas_non_neg["Nir_diff"] = cepas_non_neg.apply(lambda x: compute_difference(x["om_nir"], x["rf_nir"]), axis=1)
cepas_non_neg["NDVI_diff"] = cepas_non_neg.apply(lambda x: compute_difference(x["NDVI_Meta"], x["NDVI_BRDF"]), axis=1)

#%%
print(cepas_non_neg[["R_diff", "G_diff", "B_diff", "RE_diff", "Nir_diff", "NDVI_diff"]].describe())

#%%
# Plot Histo band differences
cepas_non_neg["R_diff"].plot.hist(bins=150, color="red", alpha=0.5, label="R")
cepas_non_neg["G_diff"].plot.hist(bins=150, color="green", alpha=0.5, label="G")
cepas_non_neg["B_diff"].plot.hist(bins=150, color="blue", alpha=0.5, label="B")
cepas_non_neg["RE_diff"].plot.hist(bins=150, color="cyan", alpha=0.5, label="RE")
cepas_non_neg["Nir_diff"].plot.hist(bins=150, color="salmon", alpha=0.5, label="Nir")

#plt.title("$Metashape$")
plt.xlabel("$reflectance \; difference$", linespacing=4)
plt.legend()
plt.show()

#%%
# Plot Histo NDVI differences
cepas_non_neg["NDVI_diff"].plot.hist(bins=150, color="purple", alpha=0.5, label="NDVI")

plt.xlabel("$NDVI \; difference$", linespacing=4)
plt.legend()
plt.show()

#%%
# END
# ----------------------------------------------------------------
print("End")
