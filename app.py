from data import BetPredictionModel, predict_stat
import numpy as np
import pandas as pd
import os
import dotenv
from pymongo import MongoClient
from prizepicksapi import call_endpoint
import json
import torch
import tkinter as tk
from tkinter import ttk
from scipy.stats import norm

client = MongoClient(os.getenv("DATABASE_URL"))
db = client['nba_database']
usages_db = client['nba_usages']
months = ['December', 'February', 'January', 'March', 'November', 'October']

teams_data = pd.DataFrame(list(db['teams'].find()))
players_data = pd.DataFrame(list(db['players'].find()))
gamelogs_data = pd.DataFrame(list(db['gamelogs'].find()))
injuries_data = pd.DataFrame(list(db['injuries'].find()))
matchups_data = pd.DataFrame(list(db['matchups'].find()))
usages_data = pd.DataFrame(list(db['usages'].find()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prizepicks_statmap = {
    'Points' : 'pts',
    'Assists' : 'ast',
    'Rebounds': 'trb',
    'Points (Combo)' : 'pts',
    'Assists (Combo)' : 'ast',
    'Rebounds (Combo)' : 'trb',
    '3-PT Made (Combo)' : '3p',
    'Pts' : 'pts',
    'Rebs' : 'trb',
    'Asts' : 'ast',
    'FG Made' : 'fg',
    'Blocked Shots' : 'blk',
    '3-PT Made' : '3p',
    '3-PT Attempted' : '3pa',
    'FG Made' : 'fg',
    'FG Attempted' : 'fga',
    'Defensive Rebounds' : 'drb',
    'Offensive Rebounds' : 'orb',
    'Free Throws Made' : 'ft',
    'Turnovers' : 'tov',
    'Steals' : 'stl',
    'Blks' : 'blk',
    'Stls' : 'stl',
}

#'pts', 'ast', 'trb', '3p', '3pa', 'blk', 'fg', 'drb', 'fga', 'ft', 'gmsc', 'orb', 'tov', 'stl'

def run_predictions():
    url = 'https://partner-api.prizepicks.com/projections?league_id=7&per_page=1000'
    df = call_endpoint(url, include_new_player_attributes=True)

    results = []

    for _, row in df.iterrows():
        #Delete this if statement later
        if row['attributes.stat_type'] not in ['Fantasy Score', 'Dunks']:
            player_name = [row['attributes.name']]
            player_display_name = row['attributes.name']

            if len(player_name[0].split(' + ')) > 1:
                player_name = player_name[0].split(' + ')

            stat_to_predict = [row['attributes.stat_type']]
            stat_name = row['attributes.stat_type']

            if len(stat_to_predict[0].split('+')) > 1:
                stat_to_predict = stat_to_predict[0].split('+')

            for i in range(len(stat_to_predict)):
                stat_to_predict[i] = prizepicks_statmap[stat_to_predict[i]]

            line = row['attributes.line_score']
            opp_team = [row['attributes.description']]
            odds_type = row['attributes.odds_type']

            if len(opp_team[0].split('/')) > 1:
                opp_team = opp_team[0].split('/')

            with open("stat_mean_loss.json", "r") as f:
                metadata = json.load(f)

            predicted_stat_sum = 0.0
            explanations = []
            stds = []

            for name, opp in zip(player_name, opp_team):   
                for stat in stat_to_predict:
                    model = BetPredictionModel(input_size=125)
                    model.load_state_dict(torch.load(f"models/{stat}_model.pth", weights_only=True))
                    model.to(device)

                    predicted_stat, explanation = predict_stat(name, opp, model, gamelogs_data, players_data, teams_data, matchups_data, injuries_data, usages_data, stat)
                    predicted_stat_sum += predicted_stat
                    explanations.append(explanation)
                    stds.append(np.sqrt(metadata.get(stat, {}).get("loss", None)))

            combined_std = np.sqrt(sum(s**2 for s in stds))

            if odds_type in ['goblin', 'demon']:
                z_score = (line - predicted_stat_sum) / combined_std
                probability = 1 - norm.cdf(z_score)
                if odds_type == 'goblin':
                    stat_name += '\U0001F47D'
                else:
                    stat_name += '\U0001F47F'
            else:
                probability = abs(predicted_stat_sum - line) / combined_std

            if probability >= 1.5:
                stat_name += '\U0001F4A3'

            results.append((player_display_name, stat_name, predicted_stat_sum, line, probability, explanations))

    results.sort(key=lambda x: x[4], reverse=True)

    update_ui(results)

explanations = {}

def update_ui(results):
    for row in tree.get_children():
        tree.delete(row)
    
    idx = 0
    for player, stat, pred, line, prob, expl in results:
        row_id = idx
        explanations[row_id] = expl
        idx += 1
        if pred > line:
            color = "green"
        else:
            color = "red"

        tree.insert("", "end", values=(player, stat, f"{pred:.2f}", f"{line:.2f}", f"{prob:.2f}"), tags=(color,), iid=(str(row_id)))

def on_item_hover(event):
    item = tree.identify_row(event.y)
    
    if item:
        row_id = int(item)
        detailed_explanation = explanations.get(row_id, [])
        explanation_text = ''
        
        if detailed_explanation:
            for explanation in detailed_explanation:
                explanation_text = "\n".join([f"{reason}: {float(weight):.4f}" for reason, weight in explanation])
            tooltip_label.config(text=explanation_text)

            row_bbox = tree.bbox(item)
            if row_bbox:
                row_x, row_y, _, row_height = row_bbox
                tooltip_label.place(x=row_x + 200, y=row_y + row_height // 2)  # Align with the row
        else:
            tooltip_label.place_forget()
    else:
        tooltip_label.place_forget()

# UI Setup
root = tk.Tk()
root.title("CAL LOCK BAR")

frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

tree = ttk.Treeview(frame, columns=("Player", "Stat", "Predicted", "Line", "Probability"), show="headings")
for col in ("Player", "Stat", "Predicted", "Line", "Probability"):
    tree.heading(col, text=col)
    tree.column(col, width=100)

scrollbar = tk.Scrollbar(frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)

tree.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

tree.tag_configure("green", background="lightgreen")
tree.tag_configure("red", background="lightcoral")

frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

tooltip_label = tk.Label(root, bg="lightyellow", relief="solid", width=40, height=10, justify="left", anchor="nw")

# Bind hover event to show explanation
tree.bind("<Motion>", on_item_hover)

btn = tk.Button(root, text="Run Predictions", command=run_predictions)
btn.pack()

root.mainloop()