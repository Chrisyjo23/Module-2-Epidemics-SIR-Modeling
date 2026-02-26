#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
#%%
# Load the data
data = pd.read_csv('C:\\Users\\Luke Friscia\\OneDrive\\CompBME\\Module 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
print(data.columns)
#%%
# Make a plot of the active cases over time
# Chatgpt was used in order to troubleshoot and assist in coding ----> OpenAI. (2026). ChatGPT (February 23 version) [Large language model]. https://chat.openai.com/


plt.plot(data['day'], data['active reported daily cases'], marker='o')
plt.xlabel('Day')
plt.ylabel('Active Infections')
plt.title('Day vs Active Infections (DATA RELEASE #1)')
plt.show()