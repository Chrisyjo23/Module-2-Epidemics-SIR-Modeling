import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Chatgpt was used in order to troubleshoot and assist in coding ----> OpenAI. (2026). ChatGPT (February 23 version) [Large language model]. https://chat.openai.com/


# Load Data (Release #2)

data = pd.read_csv("C:\\Users\\chris_5y66qc1\\Downloads\\Academic File (Spring 2026)\\Computational BME\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#2.csv")

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



# True % Relative Error


# Data peak values
data_peak_cases = np.max(I_data)
data_peak_day = t_data[np.argmax(I_data)]

# Model peak values (already calculated but repeated for clarity)
model_peak_cases = np.max(I)
model_peak_day = np.argmax(I)

# True percent relative error

error_peak_cases = abs(data_peak_cases - model_peak_cases) / data_peak_cases * 100

error_peak_day = abs(data_peak_day - model_peak_day) / data_peak_day * 100

print("Data peak infections =", data_peak_cases)
print("Model peak infections =", model_peak_cases)
print("Percent error in peak infections =", error_peak_cases, "%")

print()

print("Data peak day =", data_peak_day)
print("Model peak day =", model_peak_day)
print("Percent error in peak day =", error_peak_day, "%")



# INTERVENTION SIMULATIONS


intervention_start = 70


# Mask Mandate (40% reduction in transmission)

def run_mask_intervention(beta, sigma, gamma):

    S = np.zeros(len(time))
    E = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    for i in range(len(time)-1):

        if i >= intervention_start:
            beta_eff = beta * 0.6
        else:
            beta_eff = beta

        dS = -beta_eff * S[i] * I[i] / N
        dE = beta_eff * S[i] * I[i] / N - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]

        S[i+1] = S[i] + dS * dt
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
        R[i+1] = R[i] + dR * dt

    return S,E,I,R


# Vaccine Campaign (2000 students vaccinated once on day 70)

def run_vaccine_campaign(beta, sigma, gamma):

    S = np.zeros(len(time))
    E = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    for i in range(len(time)-1):

        if i == intervention_start:
            vaccinated = 2000 * 0.9
            S[i] -= vaccinated
            R[i] += vaccinated

        dS = -beta * S[i] * I[i] / N
        dE = beta * S[i] * I[i] / N - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]

        S[i+1] = S[i] + dS * dt
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
        R[i+1] = R[i] + dR * dt

    return S,E,I,R


# Vaccine Rollout (1000 students vaccinated on days 70, 80, 90)

def run_vaccine_rollout(beta, sigma, gamma):

    S = np.zeros(len(time))
    E = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    rollout_days = [70,80,90]

    for i in range(len(time)-1):

        if i in rollout_days:
            vaccinated = 1000 * 0.9
            S[i] -= vaccinated
            R[i] += vaccinated

        dS = -beta * S[i] * I[i] / N
        dE = beta * S[i] * I[i] / N - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]

        S[i+1] = S[i] + dS * dt
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
        R[i+1] = R[i] + dR * dt

    return S,E,I,R


# Run intervention models

S_mask,E_mask,I_mask,R_mask = run_mask_intervention(beta,sigma,gamma)

S_vac,E_vac,I_vac,R_vac = run_vaccine_campaign(beta,sigma,gamma)

S_roll,E_roll,I_roll,R_roll = run_vaccine_rollout(beta,sigma,gamma)

# Plot interventions

plt.figure()

plt.plot(time, I, label="Baseline")
plt.plot(time, I_mask, label="Mask Mandate")
plt.plot(time, I_vac, label="Vaccine Campaign")
plt.plot(time, I_roll, label="Vaccine Rollout")

plt.xlabel("Day")
plt.ylabel("Active infections")
plt.title("Intervention Comparison")
plt.legend()

plt.show()