import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Chatgpt was used in order to troubleshoot and assist in coding ----> OpenAI. (2026). ChatGPT (February 23 version) [Large language model]. https://chat.openai.com/


# Load Data (Release #2)

data = pd.read_csv("C:\\Users\\Luke Friscia\\OneDrive\\CompBME\\Module 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#2.csv")

t_data = data["day"].values
I_data = data["active reported daily cases"].values

# Initial conditions

N = 25000          # population estimate
I0 = I_data[0]
E0 = 5             # small exposed guess
R0 = 0
S0 = N - I0 - E0

dt = 1
days = 120
time = np.arange(0, days, dt)

# Euler SEIR function

def run_seir(beta, sigma, gamma):

    S = np.zeros(len(time))
    E = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    for i in range(len(time)-1):

        dS = -beta * S[i] * I[i] / N
        dE = beta * S[i] * I[i] / N - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]

        S[i+1] = S[i] + dS * dt
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
        R[i+1] = R[i] + dR * dt

    return S, E, I, R

# Fit parameters (grid search)

beta_vals = np.linspace(0.1, 1.0, 20)
sigma_vals = np.linspace(0.1, 0.5, 10)
gamma_vals = np.linspace(0.05, 0.3, 10)

best_sse = np.inf
best_params = None

for b in beta_vals:
    for s in sigma_vals:
        for g in gamma_vals:

            S,E,I,R = run_seir(b,s,g)

            model = I[:len(I_data)]
            sse = np.sum((model - I_data)**2)

            if sse < best_sse:
                best_sse = sse
                best_params = (b,s,g)

beta, sigma, gamma = best_params

print("Best beta =", beta)
print("Best sigma =", sigma)
print("Best gamma =", gamma)
print("Best SSE =", best_sse)

# Run model with best params

S,E,I,R = run_seir(beta, sigma, gamma)

# Plot model vs data

plt.figure()

plt.scatter(t_data, I_data, label="Data")
plt.plot(time, I, label="SEIR Model")

plt.xlabel("Day")
plt.ylabel("Active infections")
plt.title("SEIR Model Fit to Data")
plt.legend()

plt.show()


# Predict peak
peak_day = np.argmax(I)
peak_cases = np.max(I)

print("Peak day =", peak_day)
print("Peak infections =", peak_cases)



# Load new data (Release #3)

data3 = pd.read_csv("C:\\Users\\Luke Friscia\\OneDrive\\CompBME\\Module 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#3.csv")

t_data3 = data3["day"].values
I_data3 = data3["active reported daily cases"].values


# Calculate TRUE % relative error

true_peak_day = t_data3[np.argmax(I_data3)]
true_peak_cases = np.max(I_data3)

error_peak_cases = abs(true_peak_cases - peak_cases) / true_peak_cases * 100
error_peak_day = abs(true_peak_day - peak_day) / true_peak_day * 100

print("True peak day =", true_peak_day)
print("True peak infections =", true_peak_cases)

print("Percent relative error (peak cases) =", error_peak_cases, "%")
print("Percent relative error (peak day) =", error_peak_day, "%")


# Plot model vs true data

plt.figure()

plt.scatter(t_data3, I_data3, label="True Data (0-120)")
plt.plot(time, I, label="SEIR Model Prediction")

plt.xlabel("Day")
plt.ylabel("Active infections")
plt.title("Model Prediction vs True Data (120 Days)")
plt.legend()

plt.show()


# VT Scenario

VT_population = 40000

S0 = VT_population - 1 - E0
I0 = 1
R0 = 0

S,E,I,R = run_seir(beta, sigma, gamma)

baseline_I = I.copy()


# 1. Masking mandate

def run_masking():

    S = np.zeros(len(time))
    E = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0],E[0],I[0],R[0] = S0,E0,I0,R0

    for i in range(len(time)-1):

        if time[i] >= 70:
            beta_mod = beta * 0.6
        else:
            beta_mod = beta

        dS = -beta_mod * S[i] * I[i] / VT_population
        dE = beta_mod * S[i] * I[i] / VT_population - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]

        S[i+1] = S[i] + dS * dt
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
        R[i+1] = R[i] + dR * dt

    return I

mask_I = run_masking()


# 2. Vaccine campaign

def run_vaccine_campaign():

    S = np.zeros(len(time))
    E = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0],E[0],I[0],R[0] = S0,E0,I0,R0

    for i in range(len(time)-1):

        if time[i] == 70:
            vaccinated = 2000 * 0.9
            S[i] -= vaccinated
            R[i] += vaccinated

        dS = -beta * S[i] * I[i] / VT_population
        dE = beta * S[i] * I[i] / VT_population - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]

        S[i+1] = S[i] + dS * dt
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
        R[i+1] = R[i] + dR * dt

    return I

vaccine_I = run_vaccine_campaign()


# 3. Testing + quarantine

gamma_quarantine = 1/((1/gamma) - 2)

def run_testing():

    S = np.zeros(len(time))
    E = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0],E[0],I[0],R[0] = S0,E0,I0,R0

    for i in range(len(time)-1):

        if time[i] >= 70:
            gamma_mod = gamma_quarantine
        else:
            gamma_mod = gamma

        dS = -beta * S[i] * I[i] / VT_population
        dE = beta * S[i] * I[i] / VT_population - sigma * E[i]
        dI = sigma * E[i] - gamma_mod * I[i]
        dR = gamma_mod * I[i]

        S[i+1] = S[i] + dS * dt
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
        R[i+1] = R[i] + dR * dt

    return I

testing_I = run_testing()


# 4. Vaccine rollout

def run_vaccine_rollout():

    S = np.zeros(len(time))
    E = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0],E[0],I[0],R[0] = S0,E0,I0,R0

    for i in range(len(time)-1):

        if time[i] in [70,80,90]:
            vaccinated = 1000 * 0.9
            S[i] -= vaccinated
            R[i] += vaccinated

        dS = -beta * S[i] * I[i] / VT_population
        dE = beta * S[i] * I[i] / VT_population - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]

        S[i+1] = S[i] + dS * dt
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
        R[i+1] = R[i] + dR * dt

    return I

rollout_I = run_vaccine_rollout()


# 5. School closure

def run_school_closure():

    S = np.zeros(len(time))
    E = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0],E[0],I[0],R[0] = S0,E0,I0,R0

    for i in range(len(time)-1):

        if 70 <= time[i] < 84:
            beta_mod = beta * 0.2
        else:
            beta_mod = beta

        dS = -beta_mod * S[i] * I[i] / VT_population
        dE = beta_mod * S[i] * I[i] / VT_population - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]

        S[i+1] = S[i] + dS * dt
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
        R[i+1] = R[i] + dR * dt

    return I

closure_I = run_school_closure()


# Plot all interventions

plt.figure()

plt.plot(time, baseline_I, label="Baseline")
plt.plot(time, mask_I, label="Masking")
plt.plot(time, vaccine_I, label="Vaccine Campaign")
plt.plot(time, rollout_I, label="Vaccine Rollout")
plt.plot(time, testing_I, label="Testing + Quarantine")
plt.plot(time, closure_I, label="School Closure")

plt.xlabel("Day")
plt.ylabel("Active infections")
plt.title("Intervention Comparison (VT Scenario)")
plt.legend()

plt.show()