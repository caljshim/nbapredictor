from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shap
import numpy as np
from rapidfuzz import process
import json
from datetime import datetime

month_dict = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


load_dotenv()
client = MongoClient(os.getenv("DATABASE_URL"))

db = client['nba_database']

def get_closest_player_name(player_name, usages_data):
    all_names = usages_data['name'].tolist()
    result = process.extractOne(player_name, all_names)

    if result is None:
        return None
    
    best_match, score, _ = result

    if score >= 40:
        return best_match
    return None

class NBADataset(Dataset):
    def __init__(self, game_logs, players_data, stat_to_predict, matchups_data, team_data, injury_data, usages_data):
        self.game_logs = game_logs
        self.players_data = players_data
        self.matchups_data = matchups_data
        self.team_data = team_data
        self.injury_data = injury_data
        self.stat_to_predict = stat_to_predict
        self.player_id_map = {player: idx for idx, player in enumerate(self.players_data['name'].unique())}
        self.usages_data = usages_data

    def __len__(self):
        return len(self.game_logs)

    def get_player_features(self, player_name):
        player_data = self.players_data[self.players_data['name'] == player_name]

        if player_data.empty:
            player_data = self.players_data[self.players_data['name'] == get_closest_player_name(player_name, self.players_data)]

        if player_data.empty:
            features = [0.0] * 8
        else:
            player_data = player_data.iloc[0]

            stats = player_data['curr_season_avgs']
            
            features = [
                float(stats['games']) if stats['games'] != '-' else 0.0,
                float(stats['pts']) if stats['pts'] != '-' else 0.0,
                float(stats['trb']) if stats['trb'] != '-' else 0.0,
                float(stats['ast']) if stats['ast'] != '-' else 0.0,
                float(stats['fg%']) if stats['fg%'] != '-' else 0.0,
                float(stats['fg3%']) if stats['fg3%'] != '-' else 0.0,
                float(stats['ft%']) if stats['ft%'] != '-' else 0.0,
                float(stats['efg%']) if stats['efg%'] != '-' else 0.0,
            ]

        return features

    def get_team_features(self, team_name):
        try:
            team_data = self.team_data[self.team_data['abbrev'] == team_name]
        except KeyError as e:
            print(f"{e}, {team_name}")

        if team_data.empty:
            raise ValueError(f"No data found for team: {team_name}")
        
        team_data = team_data.iloc[0]

        stats = team_data['teamStats']

        features = [
            float(stats['MP']) if stats['MP'] != '-' else 0.0,
            float(stats['FG']) if stats['FG'] != '-' else 0.0,
            float(stats['FGA']) if stats['FGA'] != '-' else 0.0,
            float(stats['FG%']) if stats['FG%'] != '-' else 0.0,
            float(stats['3P']) if stats['3P'] != '-' else 0.0,
            float(stats['3PA']) if stats['3PA'] != '-' else 0.0,
            float(stats['3P%']) if stats['3P%'] != '-' else 0.0,
            float(stats['2P']) if stats['2P'] != '-' else 0.0,
            float(stats['2PA']) if stats['2PA'] != '-' else 0.0,
            float(stats['2P%']) if stats['2P%'] != '-' else 0.0,
            float(stats['FT']) if stats['FT'] != '-' else 0.0,
            float(stats['FTA']) if stats['FTA'] != '-' else 0.0,
            float(stats['FT%']) if stats['FT%'] != '-' else 0.0,
            float(stats['ORB']) if stats['ORB'] != '-' else 0.0,
            float(stats['DRB']) if stats['DRB'] != '-' else 0.0,
            float(stats['TRB']) if stats['TRB'] != '-' else 0.0,
            float(stats['AST']) if stats['AST'] != '-' else 0.0,
            float(stats['STL']) if stats['STL'] != '-' else 0.0,
            float(stats['BLK']) if stats['BLK'] != '-' else 0.0,
            float(stats['TOV']) if stats['TOV'] != '-' else 0.0,
            float(stats['PF']) if stats['PF'] != '-' else 0.0,
            float(stats['PTS']) if stats['PTS'] != '-' else 0.0,
        ]

        return features
    
    def get_rolling_averages(self, game_log, player_name, opp_name):
        player_gamelogs = self.game_logs[self.game_logs['name'] == player_name]
        if player_gamelogs.empty:
            player_gamelogs = self.game_logs[self.game_logs['name'] == get_closest_player_name(player_name, self.game_logs)]

        player_gamelogs = player_gamelogs.copy()
        game_log_date = pd.to_datetime(game_log['date'])
        player_gamelogs['date'] = pd.to_datetime(player_gamelogs['date'])

        player_gamelogs = player_gamelogs[player_gamelogs['date'] < game_log_date]
        player_gamelogs = player_gamelogs.sort_values(by='date', ascending=False)
        player_gamelogs = player_gamelogs.head(5)

        if not player_gamelogs.empty:
            rolling_cols = [
                'h/a', 'w/l', 'min',
                'fgm', 'fga', 'fg%', '3pm', '3pa', '3p%', 'ftm', 'fta', 'ft%', 'oreb', 'dreb', 'reb',
                'ast', 'tov', 'stl', 'blk', 'blka', 'pf', 'pfd', 'pts', 'plus_minus', 'fp'
            ]

        for col in rolling_cols:
            player_gamelogs_vs_opp[col] = player_gamelogs_vs_opp[col].replace('-', 0.0)
            player_gamelogs_vs_opp[col] = pd.to_numeric(player_gamelogs_vs_opp[col], errors='coerce').fillna(0.0)
            player_gamelogs_vs_opp[f'rolling_avg_{col}_vs_opp'] = player_gamelogs_vs_opp[col].rolling(window=5).mean().fillna(0.0)

            features = player_gamelogs.iloc[-1][[f'rolling_avg_{col}' for col in rolling_cols]].values.tolist()
        else:
            features = [0.0] * 18

        player_gamelogs_vs_opp = player_gamelogs[player_gamelogs['opp'] == opp_name].copy()

        if not player_gamelogs_vs_opp.empty:
            window_size = min(len(player_gamelogs_vs_opp), 5)

            for col in rolling_cols:
                player_gamelogs_vs_opp[f'rolling_avg_{col}_vs_opp'] = player_gamelogs_vs_opp[col].rolling(window=window_size).mean().fillna(0.0)

            features += player_gamelogs_vs_opp.iloc[-1][[f'rolling_avg_{col}_vs_opp' for col in rolling_cols]].values.tolist()
        else:
            features += [0.0] * 18

        return features
    
    def get_matchup_features(self, off_player, opp_team):
        opp_team_roster = self.team_data[self.team_data['abbrev'] == opp_team]
        opp_team_roster = opp_team_roster['roster'].iloc[0]

        off_player_pos = self.players_data[self.players_data['name'] == off_player]
        if off_player_pos.empty:
            off_player_pos = self.players_data[self.players_data['name'] == get_closest_player_name(off_player, self.players_data)]
        off_player_pos = off_player_pos['bio'].iloc[0]['pos']

        filtered_matchups = []

        for player in opp_team_roster:
            temp_player = self.players_data[self.players_data['name'] == player]
            if temp_player.empty:
                temp_player = self.players_data[self.players_data['name'] == get_closest_player_name(player, self.players_data)]
            if not temp_player.empty:
                if temp_player['bio'].iloc[0]['pos'] == off_player_pos:
                    matchup = self.matchups_data[
                        (self.matchups_data['off_player'] == off_player) & 
                        (self.matchups_data['def_player'] == player)
                    ]

                    if not matchup.empty:
                        stats = matchup['stats'].values[0]
                        try:
                            stats['matchup_min'] = sum(int(x) * 60**i for i, x in enumerate(reversed(stats['matchup_min'].split(':'))))
                        except:
                            pass

                        for stat, value in stats.items():
                            try:
                                stats[stat] = float(value)
                            except ValueError:
                                stats[stat] = 0.0

                        exclude_keys = ["matchup_min", "partial_poss", "team_pts", "fg%", "3p%"]
                        gp = stats.get('gp', 1)

                        averaged_stats = {
                            'off_player': self.player_id_map.get(off_player, -1),
                            'def_player': self.player_id_map.get(player, -1)
                        }

                        for stat, value in stats.items():
                            if stat not in exclude_keys:
                                averaged_stats[stat] = value / gp
                            else:
                                averaged_stats[stat] = value

                        filtered_matchups.append(averaged_stats)

        filtered_matchups = filtered_matchups[:1]

        if filtered_matchups:
            filtered_matchups_df = pd.DataFrame(filtered_matchups)

            features = filtered_matchups_df.drop(columns=['off_player', 'def_player']).values.flatten()
        else:
            features = [0.0] * 17

        return list(features)
    
    # def get_injury_features(self, player_name, opp_team):
    #     player_info = self.players_data[self.players_data['name'] == player_name]
    #     if player_info.empty:
    #         player_info = self.players_data[self.players_data['name'] == get_closest_player_name(player_name, self.players_data)]
    #     player_info = player_info.iloc[0]

    #     player_team = player_info['team']
    #     player_position = player_info['bio']['pos']
    #     player_usage = self.usages_data[self.usages_data['name'] == player_name]

    #     if player_usage.empty:
    #         player_usage = self.usages_data[self.usages_data['name'] == get_closest_player_name(player_name, self.players_data)]
    #         if player_usage.empty:
    #             return [0.0] * 20

    #     injured_players = self.injury_data[
    #         (self.injury_data['status'] == 'Out') | 
    #         (self.injury_data['status'] == 'Out For Season')
    #     ]

    #     if injured_players.empty:
    #         return [0.0] * 20

    #     injured_players_opp = injured_players[injured_players['team'] == opp_team]
    #     injured_players_team = injured_players[injured_players['team'] == player_team]

    #     injured_teammates_names = injured_players_team['name'].tolist()
    #     current_roster = self.team_data.loc[self.team_data['abbrev'] == player_team, 'roster']
    #     current_roster = current_roster.tolist()[0]

    #     healthy_teammates_usages = self.usages_data[self.usages_data['team'] == player_team]
    #     healthy_teammates_usages = healthy_teammates_usages.loc[~self.usages_data['name'].isin(injured_teammates_names) & self.usages_data['name'].isin(current_roster)]
                
    #     features = []

    #     # === TEAMMATE INJURIES IMPACT ===
    #     if injured_players_team.empty:
    #         return [0.0] * 20
    #     else:
    #         for _, injured in injured_players_team.iterrows():
    #             injured_player = self.players_data[self.players_data['name'] == injured['name']]
    #             injured_usage = self.usages_data[self.usages_data['name'] == injured['name']]

    #             if injured_player.empty:
    #                 continue

    #             if injured_usage.empty:
    #                 injured_usage = self.usages_data[self.usages_data['name'] == get_closest_player_name(injured['name'], self.usages_data)]
    #                 if injured_usage.empty:
    #                     continue

    #             injured_player = injured_player.iloc[0]
    #             injured_position = injured_player['bio']['pos']

    #             if player_position == injured_position:
    #                 player_usage = player_usage.iloc[0]
    #                 injured_usage = injured_usage.iloc[0]

    #                 if isinstance(player_usage, ObjectId) or isinstance(injured_usage, ObjectId):
    #                     return [0.0] * 20

    #                 stats_to_redistribute = ["%pts", "%reb", "%ast", "%fga", "%3pa", "%fta", "%fgm", "%3pm", 
    #                                          "%ftm", "%dreb", "%oreb", "%blk", "%stl", "%tov", "%pf", "%blka", "%pfd", "usg%"]
                    
    #                 for stat in stats_to_redistribute:
    #                     if player_usage[stat] == "0.0" or injured_usage[stat] == "0.0":
    #                         features.append(0.0)
    #                     else:
    #                         team_total_usage = 0.0
    #                         team_total_minutes = 0.0
    #                         for _, healthy in healthy_teammates_usages.iterrows():
    #                             healthy_player = self.players_data[self.players_data['name'] == healthy['name']]

    #                             if healthy_player.empty:
    #                                 healthy_player = self.players_data[self.players_data['name'] == get_closest_player_name(healthy['name'], self.players_data)]
    #                                 if healthy_player.empty:
    #                                     continue

    #                             healthy_player = healthy_player.iloc[0]
    #                             if healthy_player['bio']['pos'] == player_position:
    #                                 team_total_usage += float(healthy[stat])
    #                                 team_total_minutes += float(healthy['min'])

    #                         team_total_usage += float(injured_usage[stat])
    #                         team_total_minutes += float(injured_usage['min'])

    #                         injured_weight = float(injured_usage['min']) / team_total_minutes
    #                         weighted_injury_usage = float(injured_usage[stat]) * injured_weight

    #                         player_weight = float(player_usage['min']) / team_total_minutes
    #                         redistributed_usage = weighted_injury_usage * player_weight

    #                         player_new_usage = float(player_usage[stat]) + redistributed_usage

    #                         features.append(player_new_usage)

    #                 if len(features) == 18 and all(value == 0.0 for value in features):
    #                     features.append(0.0)
    #                     features.append(0.0)
    #                 else:
    #                     features.append(float(player_usage['min']))
    #                     features.append(float(player_usage['gp']))

    #     # === OPPONENT INJURIES IMPACT ===
    #     # for _, injured in injured_players_opp.iterrows():
    #     #     matchup_data = self.matchups_data[
    #     #         ((self.matchups_data['off_player'] == player_name) &
    #     #         (self.matchups_data['def_player'] == injured['name'])) |
    #     #         ((self.matchups_data['def_player'] == player_name) &
    #     #         (self.matchups_data['off_player'] == injured['name']))
    #     #     ]

    #     if len(features) < 20:
    #         return [0.0] * 20

    #     return features

    
    def get_usage_features(self, game_log, player_name):
        month = game_log['date'].split('-')
        month = int(month[1])
        month_name = month_dict[month]

        usages_data = self.usages_data[self.usages_data['name'] == player_name]

        if usages_data.empty:
            usages_data = usages_data[usages_data['name'] == get_closest_player_name(player_name, usages_data)]

        usages_data = usages_data[usages_data['month'] == month_name]

        if usages_data.empty:
                features = [0.0] * 20
        else:
            stats = usages_data.iloc[0]
            
            features = [
                float(stats['%pts']),
                float(stats['%reb']),
                float(stats['%ast']),
                float(stats['%fga']),
                float(stats['%3pa']),
                float(stats['%fta']),
                float(stats['%fgm']),
                float(stats['%3pm']),
                float(stats['%ftm']),
                float(stats['%dreb']),
                float(stats['%oreb']),
                float(stats['%blk']),
                float(stats['%stl']),
                float(stats['%tov']),
                float(stats['%pf']),
                float(stats['%blka']),
                float(stats['%pfd']),
                float(stats['usg%']),
                float(stats['min']),
                float(stats['gp']),
            ]

        return features
    
    def get_feature_names(self):
        feature_names = []

        player_stats = ["Games Played","Points Per Game","Rebounds Per Game","Assists Per Game","Field Goal Percentage","Three-Point Percentage","Free Throw Percentage","Effective Field Goal Percentage"]
        feature_names.extend([f"Player {stat}" for stat in player_stats])

        team_stats = ["Minutes Played","Field Goals Made","Field Goals Attempted","Field Goal Percentage","Three-Pointers Made","Three-Pointers Attempted","Three-Point Percentage","Two-Pointers Made","Two-Pointers Attempted","Two-Point Percentage","Free Throws Made","Free Throws Attempted","Free Throw Percentage","Offensive Rebounds","Defensive Rebounds","Total Rebounds","Assists","Steals","Blocks","Turnovers","Personal Fouls","Points"]
        feature_names.extend([f"Team {stat}" for stat in team_stats])  # Current team
        feature_names.extend([f"Opponent {stat}" for stat in team_stats])

        rolling_avg_stats = ["Points", "Field Goal Percentage", "Three-Point Percentage", "Three-Point Attempts", "Assists", "Blocks", "Defensive Rebounds", "Field Goals Made", "Field Goals Attempted", "Free Throws Made", "Free Throws Attempted", "Game Score", "Minutes Played", "Offensive Rebounds", "Personal Fouls", "Steals", "Turnovers", "Total Rebounds"]
        feature_names.extend([f"Rolling Avg {stat} (Last 5 games)" for stat in rolling_avg_stats])
        feature_names.extend([f"Rolling Avg {stat} vs Opponent" for stat in rolling_avg_stats])
        # "Defensive Player", "Offensive Player", 
        matchup_stats = ["Games Played", "Minutes", "Partial Possessions", "Points", "Team Points", "Assists", "Turnovers", "Blocks", "Field Goals Made", "Field Goals Attempted", "Field Goal Percentage", "Three-Point Makes", "Three-Point Attempts", "Three-Point Percentage", "Free Throws Made", "Free Throws Attempted", "Shooting Fouls Drawn"]
        feature_names.extend([f"Matchup {stat}" for stat in matchup_stats])

        usages_stats = ["Points Percentage", "Rebound Percentage", "Assist Percentage", "Field Goal Attempt Percentage", "Three-Point Attempt Percentage", "Free Throw Attempt Percentage", "Field Goals Made Percentage", "Three-Point Makes Percentage", "Free Throws Made Percentage", "Defensive Rebound Percentage", "Offensive Rebound Percentage", "Block Percentage", "Steal Percentage", "Turnover Percentage", "Personal Foul Percentage", "Blocks Against Percentage", "Personal Fouls Drawn Percentage", "Percentage", "Minutes Played", "Games Played"]
        feature_names.extend([f'Usage {stat}' for stat in usages_stats])

        # injury_stats = ["Points Percentage", "Rebound Percentage", "Assist Percentage", "Field Goal Attempt Percentage", "Three-Point Attempt Percentage", "Free Throw Attempt Percentage", "Field Goals Made Percentage", "Three-Point Makes Percentage", "Free Throws Made Percentage", "Defensive Rebound Percentage", "Offensive Rebound Percentage", "Block Percentage", "Steal Percentage", "Turnover Percentage", "Personal Foul Percentage", "Blocks Against Percentage", "Personal Fouls Drawn Percentage", "Usage Percentage", "Minutes Played", "Games Played"]
        # feature_names.extend([f'Injury: Boosted {stat}' for stat in injury_stats])

        return feature_names

    def __getitem__(self, idx):
        game_log = self.game_logs.iloc[idx]
        player_name = game_log['name']
        team_name = game_log['tm']
        opp_name = game_log['opp']

        features = (
            self.get_player_features(player_name) + 
            self.get_team_features(team_name) + 
            self.get_team_features(opp_name) +
            self.get_rolling_averages(game_log, player_name, opp_name) +
            self.get_matchup_features(player_name, opp_name) + 
            self.get_usage_features(game_log, player_name)
        )

        label = float(game_log[self.stat_to_predict])

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return features_tensor, label_tensor

class BetPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(BetPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x

#datasets
teams_data = pd.DataFrame(list(db['teams'].find()))
players_data = pd.DataFrame(list(db['players'].find()))
gamelogs_data = pd.DataFrame(list(db['gamelogs'].find()))
injuries_data = pd.DataFrame(list(db['injuries'].find()))
matchups_data = pd.DataFrame(list(db['matchups'].find()))
usages_data = pd.DataFrame(list(db['usages'].find()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model_for_stat(stat):
    dataset = NBADataset(gamelogs_data, players_data, stat, matchups_data, teams_data, injuries_data, usages_data)

    train_dataset, temp_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    #ai stuff
    model = BetPredictionModel(input_size=125)

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0  # Initialize total training loss for the epoch
        # Loop over batches from the train_loader
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs.squeeze(), labels)

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs.squeeze(), labels)
                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            test_loss = criterion(outputs.squeeze(), labels)
            total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f'Test Loss: {avg_test_loss:.4f} for {stat}')

    torch.save(model.state_dict(), f"models/{stat}_model.pth")

    try:
        with open("stat_mean_loss.json", "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}

    metadata[stat] = {"loss" : avg_test_loss}
    with open("stat_mean_loss.json", "w") as f:
        json.dump(metadata, f, indent=4)

# for stat in ['pts', 'ast', 'trb']:
#     train_model_for_stat(stat)

def explain_prediction(model, features_tensor, feature_names, top_n=5):
    model.eval()

    features_tensor = features_tensor.detach().to(device)
    
    background = torch.randn((100, 125)).to(device)
    explainer = shap.GradientExplainer(model, background)
    
    shap_values = explainer.shap_values(features_tensor)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    importance = np.abs(shap_values).flatten()
    
    ranked_indices = np.argsort(importance)[::-1]
    top_features = [(feature_names[i], importance[i]) for i in ranked_indices[:top_n]]
    
    return top_features

def predict_stat(player_name, opp_team, model, game_logs, player_data, team_data, matchup_data, injury_data, usages_data, stat_to_predict):
    dataset = NBADataset(game_logs, player_data, stat_to_predict, matchup_data, team_data, injury_data, usages_data)
    
    game_log = game_logs[game_logs['name'] == get_closest_player_name(player_name, usages_data)]

    if game_log.empty:
        game_log = game_logs[game_logs['name'] == get_closest_player_name(player_name, game_logs)]

    game_log = game_log.copy()
    game_log['date'] = pd.to_datetime(game_log['date'])
    game_log = game_log.sort_values(by='date', ascending=False).iloc[0]
    game_log['date'] = game_log['date'].strftime('%Y-%m-%d')

    player_features = dataset.get_player_features(player_name)
    team_features = dataset.get_team_features(game_log['tm'])
    opponent_features = dataset.get_team_features(opp_team)
    rolling_avg_features = dataset.get_rolling_averages(game_log, player_name, opp_team)
    matchup_features = dataset.get_matchup_features(player_name, opp_team)
    usages_features = dataset.get_usage_features(game_log, player_name)

    features = player_features + team_features + opponent_features + rolling_avg_features + matchup_features + usages_features
    feature_names = dataset.get_feature_names()

    if len(features) != len(feature_names):
        print(f"Mismatch lengths: features len: {len(features)}, feature_names len: {len(feature_names)}")
        print(f"{feature_names[len(features):]}")

    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

    model.eval()

    with torch.no_grad():
        predicted_stat = model(features_tensor).item()

    top_reasons = explain_prediction(model, features_tensor, feature_names, 5)

    return predicted_stat, top_reasons