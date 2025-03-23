# 🌍 Dynamic Climate Game

a streamlit simulation of green energy adoption as a threshold public goods game.
based on this game i designed:https://docs.google.com/document/d/1tJWnz3HHwFr0vCLX9vjgdrN7TkcHYmGK_M03o42I3Zw/edit?tab=t.sk0jy4ynmiir
this project models strategic decision-making between countries under climate pressure, where each round represents a time step. countries choose whether to invest in renewables or free-ride based on cost, influence, vulnerability, and global trends.

---

## 📦 features

- simulate multi-round climate cooperation
- urgency, pressure, and cost dynamics
- income-based adoption constraints
- payoff comparison: adopt vs. free-ride
- visualisations of global and country-level behavior

---

## 📁 structure

- `app.py` – main streamlit app
- `game_functions.py` – simulation logic
- `cleaned_data.csv` – input dataset (from world bank and vulnerability index)
- `requirements.txt` – dependencies

---

## 🚀 run locally

```bash
pip install -r requirements.txt
streamlit run app.py
