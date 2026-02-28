#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
#%%
# Load the data
data = pd.read_csv('C:\\Users\\chris_5y66qc1\\Downloads\\Academic File (Spring 2026)\\Computational BME\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
print(data.columns)
#%%
# Make a plot of the active cases over time
# Chatgpt was used in order to troubleshoot and assist in coding ----> OpenAI. (2026). ChatGPT (February 23 version) [Large language model]. https://chat.openai.com/


plt.plot(data['day'], data['active reported daily cases'], marker='o')
plt.xlabel('Day')
plt.ylabel('Active Infections')
plt.title('Day vs Active Infections (DATA RELEASE #1)')
plt.show()

# Select early exponential growth phase (adjust range if needed)

early_data = data[data['day'] < 15]   # adjust cutoff if needed

t = early_data['day']
I = early_data['active reported daily cases']

# Remove zeros 
mask = I > 0
t = t[mask]
I = I[mask]

# Take log
log_I = np.log(I)

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(t, log_I)

r = slope
print("Estimated growth rate r =", r)
plt.figure()

# Estimate R0 from r
D = 9  # assumed infectious period in days
gamma = 1 / D

R0 = 1 + r / gamma
print("Estimated R0 =", R0)

# Scatter original data
plt.scatter(data['day'], data['active reported daily cases'], label='Data', color='blue')

# Exponential fit curve
t_fit = np.linspace(data['day'].min(), data['day'].max(), 100)
I_fit = np.exp(intercept) * np.exp(r * t_fit)

plt.plot(t_fit, I_fit)

plt.xlabel("Day")
plt.ylabel("Active Infections")
plt.title("Exponential Fit to Early Infection Growth")
plt.show()
