import streamlit as st
import requests
import datetime

import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8800")

st.set_page_config(page_title="FIFA Match Predictor", page_icon="⚽")

st.title("⚽ FIFA Match Predictor")

def get_teams():
    try:
        response = requests.get(f"{API_URL}/teams")
        response.raise_for_status()
        return response.json().get("teams", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return []

def predict_match(data):
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error making prediction: {e}")
        return None

# Fetch teams on load
teams = get_teams()

if not teams:
    st.warning("No teams found. Please check if the API is running.")
else:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("Home Team", teams)
        
        with col2:
            away_team = st.selectbox("Away Team", teams)
            
        tournament = st.selectbox(
            "Tournament",
            [
                'FIFA World Cup qualification',
                'UEFA Euro qualification',
                'UEFA Euro',
                'FIFA World Cup',
                'Copa América',
                'UEFA Nations League'
            ]
        )
        
        neutral = st.checkbox("Neutral Field")
        match_date = st.date_input("Match Date", datetime.date.today())
        
        submitted = st.form_submit_button("Predict Result")
        
        if submitted:
            if home_team == away_team:
                st.error("Home team and Away team must be different.")
            else:
                payload = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "tournament": tournament,
                    "neutral": neutral,
                    "date": match_date.isoformat()
                }
                
                with st.spinner("Predicting..."):
                    result = predict_match(payload)

                if result:
                    st.divider()
                    st.subheader("Prediction Results")
                    
                    winner_prob = result['probability_home_win']
                    loser_prob = result['probability_nome_nowin'] # Note: Using API field name
                    
                    if result['home_team_win']:
                        st.success(f"**Winner:** {home_team} 🏆")
                    else:
                        st.info(f"**Winner:** {away_team} (or Draw)")
                        
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric(label=f"{home_team} Win Probability", value=f"{winner_prob:.2%}")
                    with col_res2:
                        st.metric(label=f"{home_team} Non-Win Probability", value=f"{loser_prob:.2%}")

