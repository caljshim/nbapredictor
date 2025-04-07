from data import BetPredictionModel, predict_stat
import pandas as pd
import os
from pymongo import MongoClient
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

client = MongoClient(os.getenv("DATABASE_URL"))
db = client['nba_database']

months = ['December', 'February', 'January', 'March', 'November', 'October']

teams_data = pd.DataFrame(list(db['teams'].find()))
players_data = pd.DataFrame(list(db['players'].find()))
gamelogs_data = pd.DataFrame(list(db['gamelogs'].find()))
injuries_data = pd.DataFrame(list(db['injuries'].find()))
matchups_data = pd.DataFrame(list(db['matchups'].find()))
usages_data = pd.DataFrame(list(db['usages'].find()))

player_name = "Anthony Edwards"
opp_team = "DEN"
stat_to_predict = "pts"

model = BetPredictionModel(input_size=125)
model.load_state_dict(torch.load(f"models/{stat_to_predict}_model.pth", weights_only=True))
model.to(device)

#'pts', 'ast', 'reb', '3p', '3pa', 'blk', 'fg', 'drb', 'fga', 'ft', 'fp', 'orb', 'tov', 'stl'

predicted_stat, reasons = predict_stat(player_name, opp_team, model, gamelogs_data, players_data, teams_data, matchups_data, injuries_data, usages_data, stat_to_predict)
print(f"Predicted {stat_to_predict} for {player_name}: {predicted_stat}")
print("Top contributing factors:")
for reason, value in reasons:
    print(f"- {reason}: {value:.4f}")