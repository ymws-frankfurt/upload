#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 01:04:37 2023

@author: ymws
"""
import pandas as pd

df = pd.DataFrame({'A1': [100, 100, 100],
                   'A2': [100, 100, 300],
                   'A3': [200, 100, 300],
                   'B1': [400, 300, 200],
                   'B2': [400, 300, 100]},
                  index=['1', '2', '3'])



df_w = pd.DataFrame({'Group': ['A','A','A','B','B'],
                   'Subgroup': ['A1','A2','A3','B1','B2'],
                   'Weight': [0.5, 0.2, 0.3, 0.4, 0.6]},
                  index=['1', '2', '3', '4', '5'])


#%% to be generated
#df_w.groupby('Group')['Weight'].sum().reset_index()


df_weightedave0 = pd.DataFrame({'A': [130,100,200],
                   'B': [400,300,140]},
                  index=['1', '2', '3'])


#%% sol1

df_weightedave = pd.DataFrame(index=df.index)

for group in df_w['Group'].unique():
    group_df = df_w[df_w['Group'] == group]
    group_cols = group_df['Subgroup'].values
    group_weights = group_df['Weight'].values
    weighted_averages = (df[group_cols] * group_weights).sum(axis=1)
    df_weightedave[group] = weighted_averages
    
#%% sol2 bad
    
df = pd.merge(df, df_w, on='Subgroup')

# Calculate the weighted average
df_weightedave = df.groupby('Group').apply(lambda x: (x * x['Weight']).sum() / x['Weight'].sum())

#%% sol 3 bad

df_weightedave0 = df.groupby(df_w['Group'], axis=1).apply(lambda x: x.mul(df_w['Weight']).sum(axis=1))

#%% BEST


# Create a pivot table from df_w
pivot_table = df_w.pivot(index='Group', columns='Subgroup', values='Weight')

# nanをゼロにしないとうまくいかない
df_weightedave0 = df @ pivot_table.T.fillna(0)
