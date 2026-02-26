#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv('C:\\Users\\Luke Friscia\\OneDrive\\CompBME\\Module 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# Make a plot of the active cases over time
# Chatgpt was used in order to troubleshoot and assist in coding ----> OpenAI. (2026). ChatGPT (February 23 version) [Large language model]. https://chat.openai.com/

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Select early exponential growth phase (adjust range if needed)
early_data = data[data['day'] < 15]   # adjust cutoff if needed

t = early_data['day']
I = early_data['active reported daily cases']

# Remove zeros (can't log 0)
mask = I > 0
t = t[mask]
I = I[mask]

# Take log
log_I = np.log(I)

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(t, log_I)

r = slope
print("Estimated growth rate r =", r)

D = 9 # average infectious period in days (guess)
gamma = 1 / D

R0 = 1 + r / gamma
print("Estimated R0 =", R0)

