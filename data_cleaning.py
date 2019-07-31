# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:19:31 2019

@author: singh
"""

import pandas as pd
import numpy as np

df = pd.read_csv('final_data.csv')

df['title'].replace({'\n' : ''}, inplace=True, regex=True)
for i, title in enumerate(df['title']):
    df.loc[i, 'title'] = title.strip()
    
df.dropna(axis=0, inplace=True)

print('The shape of dataset after cleaning is: {}'.format(df.shape))

df.to_csv('clean_data.csv')