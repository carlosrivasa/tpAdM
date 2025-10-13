import pandas as pd


# Función: calcula estadísticas recientes por equipo
def compute_team_form(df, team, date, window_years=3):
    """Devuelve winrate, goles a favor y goles en contra promedio
    de un equipo en los últimos window_years años antes de 'date'."""
    cutoff = date - pd.DateOffset(years=window_years)
    past_matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < date) &
        (df["date"] >= cutoff)
    ]
    if past_matches.empty:
        return pd.Series([0.5, 1.0, 1.0])  # valores neutros de winrate, promedio de goles a favor y goles en contra
    
    # resultados desde perspectiva del team
    wins, draws, losses, gf, ga = 0, 0, 0, 0, 0
    for _, row in past_matches.iterrows():
        if row["home_team"] == team:
            gf += row["home_score"]
            ga += row["away_score"]
            if row["home_score"] > row["away_score"]:
                wins += 1
            elif row["home_score"] == row["away_score"]:
                draws += 1
            else:
                losses += 1
        else:  # team fue visitante
            gf += row["away_score"]
            ga += row["home_score"]
            if row["away_score"] > row["home_score"]:
                wins += 1
            elif row["away_score"] == row["home_score"]:
                draws += 1
            else:
                losses += 1
    
    total = wins + draws + losses
    winrate = wins / total if total > 0 else 0.5
    avg_gf = gf / total if total > 0 else 1.0
    avg_ga = ga / total if total > 0 else 1.0
    
    return pd.Series([winrate, avg_gf, avg_ga])


# Crear features de fuerza para cada partido
# features = []
# for idx, row in results.iterrows():
#     home_stats = compute_team_form(results, row["home_team"], row["date"], window_years=3)
#     away_stats = compute_team_form(results, row["away_team"], row["date"], window_years=3)
    
#     features.append([
#         row["home_team"], row["away_team"], row["date"], row["result"],
#         home_stats[0], home_stats[1], home_stats[2],
#         away_stats[0], away_stats[1], away_stats[2]
#     ])

# df_feat = pd.DataFrame(features, columns=[
#     "home_team","away_team","date","result",
#     "home_winrate","home_gf","home_ga",
#     "away_winrate","away_gf","away_ga"
# ])

# print(df_feat.head())