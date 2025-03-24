import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from game_functions import (
    normalise_and_compute_influence,
    compute_static_game_variables_v2,
    initialize_G_and_W_and_T,
    compute_C_t,
    update_pressure,
    compute_perceived_climate_benefit,
    compute_real_climate_benefit,
    best_response,
    run_simulation
)

# --- data load ---
@st.cache_data
def load_raw_data(path):
    return pd.read_csv(path)

# --- page title ---
st.title("Climate Game Simulator")

st.markdown(
    """
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <p>This is a basic climate game simulation based on a simple dynamic threshold public goods model.</p>
        <p>
            📄 <a href='https://docs.google.com/document/d/1tJWnz3HHwFr0vCLX9vjgdrN7TkcHYmGK_M03o42I3Zw/edit?tab=t.sk0jy4ynmiir#heading=h.e5eaa2o7yc29' target='_blank'>
            Click here to read the documentation explaining the game.</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


raw_df = load_raw_data("cleaned_data.csv")
all_countries = sorted(raw_df["Country Name"].unique())

# --- sidebar: dynamic toggle w/ buttons ---
st.sidebar.header("Included Countries")

if "selected_countries" not in st.session_state:
    st.session_state.selected_countries = all_countries

col1, col2 = st.sidebar.columns(2)
if col1.button("Select All"):
    st.session_state.selected_countries = all_countries
if col2.button("Clear All"):
    st.session_state.selected_countries = []

selected_countries = st.sidebar.multiselect(
    "Countries in simulation",
    options=all_countries,
    default=st.session_state.selected_countries
)

# keep in sync
st.session_state.selected_countries = selected_countries

# --- filter + preprocess ---
df = raw_df[raw_df["Country Name"].isin(selected_countries)].copy()
df = normalise_and_compute_influence(df)
df = compute_static_game_variables_v2(df)

# --- sidebar: simulation parameters ---
st.sidebar.header("Simulation Parameters")

N = st.sidebar.slider("Rounds (N)", 5, 20, 10)
st.sidebar.caption("Number of rounds — represents time passing (e.g., years).")

lambda_u = st.sidebar.slider("Urgency Growth (λ)", 0.0, 5.0, 1.0)
st.sidebar.caption("Controls how sharply urgency increases.")

gamma = st.sidebar.slider("Pressure Growth Rate (γ)", 0.5, 2.0, 0.8)
st.sidebar.caption("How strongly political pressure grows with global adoption.")

Z = st.sidebar.slider("Cost Reduction Rate (Z)", 0.0, 1.0, 0.1)
st.sidebar.caption("How much adoption reduces future transition costs.")

theta = st.sidebar.slider("Threshold (θ)", 0.6, 1.0, 0.6)
st.sidebar.caption("Influence needed to trigger climate success.")

min_gdp_threshold = st.sidebar.slider("Minimum GDP per capita to adopt", 0, 20000, 5000)
st.sidebar.caption("Below this GDP, countries can't adopt.")

# --- run simulation ---
history = run_simulation(
    df.copy(),
    N=N,
    lambda_u=lambda_u,
    gamma=gamma,
    Z=Z,
    theta=theta,
    min_gdp_threshold=min_gdp_threshold
)

results = pd.concat(history, ignore_index=True)
global_adoption = results.groupby("round")["W_t"].mean()

# --- plot 1: global adoption trajectory ---
st.subheader("Global Adoption Trajectory")

fig, ax = plt.subplots(figsize=(8, 5))
sns.set_style("whitegrid")
sns.lineplot(data=global_adoption, marker="o", ax=ax)

ax.set_xlabel("Round")
ax.set_ylabel("Global Adoption Effort (Wₜ)")
ax.set_title("Global Influence-Weighted Adoption Over Time")
st.pyplot(fig)

# --- additional plot: % of countries adopting (unweighted) ---
st.subheader("Share of Countries Adopting")

adopt_counts = results.groupby("round")["G"].mean()

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.lineplot(x=adopt_counts.index, y=adopt_counts.values * 100, marker="s", ax=ax2)

ax2.set_xlabel("Round")
ax2.set_ylabel("% of Countries Adopting")
ax2.set_title("Adoption Share (Unweighted)")
ax2.grid(True)
st.pyplot(fig2)













# --- plot 2: country-level payoff comparison ---
st.sidebar.header("Visualise Country Payoffs")

country_focus = st.sidebar.selectbox("Track a country", sorted(df["Country Name"].unique()))
country_data = results[results["Country Name"] == country_focus]

st.subheader(f"Payoff Comparison – {country_focus} (select any country)")

fig_payoff, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(x="round", y="Payoff_if_adopt", data=country_data, label="Payoff: Adopt", marker="o", ax=ax)
sns.lineplot(x="round", y="Payoff_if_free_ride", data=country_data, label="Payoff: Free-Ride", marker="o", ax=ax)

ax.set_title(f"Adoption vs. Free-Ride Payoffs – {country_focus}")
ax.set_xlabel("Round")
ax.set_ylabel("Payoff (scaled)")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.legend()
st.pyplot(fig_payoff)
