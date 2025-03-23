
import numpy as np
import pandas as pd

def normalise_and_compute_influence(df, alpha=0.5, beta=0.25, gamma=0.25):
    """
    Normalizes GDP, energy usage, and investment; computes country influence score.

    Parameters:
    - df: pandas DataFrame with raw economic data
    - alpha, beta, gamma: weights for GDP, energy usage, and investment respectively

    Returns:
    - df: modified DataFrame with normalized columns and influence score
    """
    df = df.copy()

    # normalize core metrics
    df["GDP_norm"] = df["GDP (constant 2015 US$)"] / df["GDP (constant 2015 US$)"].sum()
    df["investment_norm"] = df["Gross capital formation (constant 2015 US$)"] / df["Gross capital formation (constant 2015 US$)"].sum()
    df["energy_norm"] = df["energy_usage"] / df["energy_usage"].sum()

    # compute influence
    df["influence"] = (
        alpha * df["GDP_norm"] +
        beta * df["energy_norm"] +
        gamma * df["investment_norm"]
    )

    # normalize influence
    df["influence"] /= df["influence"].sum()

    return df




def compute_static_game_variables_v2(df, base_alpha=1.0, scaling_factor=100000, payoff_scale=10000):
    """
    Initializes all static variables for the game model.
    Includes:
    - alpha_i
    - gdp_penalty
    - C_0_i (initial cost)
    - C_t (initial round cost)
    - EconomicGains
    - ClimatePayoff
    - Pressure_0
    - Pressure_t

    Returns:
    - df: modified DataFrame with new static columns
    """
    df = df.copy()

    # carbon intensity (dirtiness) scaled
    max_ci = df["Carbon intensity of GDP (kg CO2e per constant 2015 US$ of GDP)"].max()
    df["alpha_i"] = base_alpha * (
        df["Carbon intensity of GDP (kg CO2e per constant 2015 US$ of GDP)"] / max_ci
    )

    # development penalty
    gdp_ratio = df["GDP per capita"] / df["GDP per capita"].max()
    df["gdp_penalty"] = np.exp(-1 * gdp_ratio) + 1

    # cost of adoption
    df["C_0_i"] = (
        df["alpha_i"] *
        df["GDP_norm"] *
        (1 + df["Fossil fuel energy consumption (% of total)"] / 100) *
        df["gdp_penalty"]
    ) * scaling_factor 

    df["C_t"] = df["C_0_i"]

    # economic gains from green transition
    df["EconomicGains"] = (
        (1 - df["alpha_i"]) *
        df["GDP_norm"] *
        (1 - df["Fossil fuel energy consumption (% of total)"] / 100) 
    ) * scaling_factor

    # climate payoff
    df["ClimatePayoff"] = (
        (1 - df["Vulnerability Score"] / 100) *
        df["GDP_norm"] *
        payoff_scale
    )

    # initial pressure
    df["Pressure_0"] = (
        df["GDP_norm"] *
        (1 - df["alpha_i"]) *
        0.05 *
        scaling_factor
    )

    df["Pressure_t"] = df["Pressure_0"]

    df["Perceived_Climate_B"] = 0

    return df

def initialize_G_and_W_and_T(df, min_gdp_threshold, theta=0.6):
    df["G"] = ((df["EconomicGains"] > df["C_0_i"]) & 
               (df["GDP per capita"] > min_gdp_threshold)).astype(int)
    W_t = (df["G"] * df["influence"]).sum()
    T = theta * df['influence'].sum()
    return df, W_t, T




###COSTS
def compute_C_t(C_t, W_t, Z=0.4):
    """
    Return cost of adoption for all players, independent of their G status.
    Actual cost is only applied if a player adopts.
    """
    return C_t * (1 - Z * (1 + W_t))



###PRESSURE


def update_pressure(df, W_t, gamma=1.05):
    """
    Update country-specific pressure based on prior value and global adoption.

    Returns a pd.Series
    """
    return df["Pressure_t"] * (gamma * (1 + W_t))




###Perceived Climate Benefit
def compute_perceived_climate_benefit(df, T_d, U_t):
    """
    Compute perceived climate benefit at time t.

    Parameters:
    - df: DataFrame containing ClimatePayoff per country
    - T_d: threshold dummy (0 or 1)
    - U_t: urgency scalar at time t (e.g. (t / N)^λ)

    Returns:
    - Series of perceived climate benefits for each country
    """
    return (1 - T_d) * U_t * df["ClimatePayoff"] * 0.25




###Real Climate Benefits
def compute_real_climate_benefit(df, T_d):
    """
    Compute real climate benefit at time t.

    Parameters:
    - df: DataFrame containing ClimatePayoff per country
    - T_d: threshold dummy (0 or 1)

    Returns:
    - Series of real climate benefits for each country
    """
    return T_d * df["ClimatePayoff"]



def best_response(df, min_gdp_threshold=5000):
    return ((df["Payoff_if_adopt"] > df["Payoff_if_free_ride"]) & 
            (df["GDP per capita"] > min_gdp_threshold)).astype(int)



def run_simulation(df, N=10, lambda_u=1, gamma=1.05, Z=0.1, theta=0.8, min_gdp_threshold=5000):
    """
    Simulates N rounds of the Climate Catastrophe Game.

    Parameters:
    - df: initialized DataFrame with all static variables
    - N: number of rounds
    - lambda_u: urgency growth exponent
    - gamma: pressure growth rate
    - Z: cost reduction rate from global adoption
    - theta: adoption threshold proportion of total influence
    - min_gdp_threshold: GDP per capita cutoff for eligibility to adopt

    Returns:
    - history: list of DataFrames, one per round with key metrics
    """
    df = compute_static_game_variables_v2(df)
    df = normalise_and_compute_influence(df)
    df, W_t, T = initialize_G_and_W_and_T(df, min_gdp_threshold, theta)

    history = []

    for t in range(1, N + 1):
        # compute urgency and threshold dummy
        U_t = ((t / N) ** lambda_u) * 0.1
        T_d = 1 if W_t >= T else 0

        # update moving variables
        df["C_t"] = compute_C_t(df["C_0_i"], W_t, Z)
        df["Pressure_t"] = update_pressure(df, W_t, gamma)
        df["Perceived_Climate_B"] = compute_perceived_climate_benefit(df, T_d, U_t)
        df["Real_Climate_B"] = compute_real_climate_benefit(df, T_d)

        # compute payoffs
        df["Payoff_if_adopt"] = (
            df["EconomicGains"] + df["Perceived_Climate_B"] + df["Real_Climate_B"] - df["C_t"]
        )
        df["Payoff_if_free_ride"] = (
            - df["Pressure_t"] + df["Real_Climate_B"] 
        )

        # update strategies
        df["G"] = best_response(df, min_gdp_threshold)

        # update W_t based on new adoptions
        W_t = (df["G"] * df["influence"]).sum()

        # snapshot state for this round
        snapshot = df.copy()
        snapshot["round"] = t
        snapshot["W_t"] = W_t
        snapshot["U_t"] = U_t
        snapshot["T_d"] = T_d

        history.append(snapshot[[
            "round", "Country Name", "G", "W_t", "U_t", "T_d",
            "C_t", "EconomicGains", "Pressure_t",
            "Perceived_Climate_B", "Real_Climate_B",
            "Payoff_if_adopt", "Payoff_if_free_ride"
        ]])

    return history

