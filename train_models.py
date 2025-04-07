from data import train_model_for_stat
from scraper import addTeamsToDatabase, addPlayersToDatabase, addPlayerGameLogsToDatabase, addPlayerUsageRatesToDatabase, addInjuryReportsToDatabase
import time
import schedule
import datetime
from pymongo import MongoClient
import os
import pandas as pd

client = MongoClient(os.getenv("DATABASE_URL"))
db = client['nba_database']

teams_data = pd.DataFrame(list(db['teams'].find()))
players_data = pd.DataFrame(list(db['players'].find()))
gamelogs_data = pd.DataFrame(list(db['gamelogs'].find()))
injuries_data = pd.DataFrame(list(db['injuries'].find()))
matchups_data = pd.DataFrame(list(db['matchups'].find()))
usages_data = pd.DataFrame(list(db['usages'].find()))

def start_training():
    # addTeamsToDatabase()
    # addPlayersToDatabase()
    addPlayerGameLogsToDatabase()
    addPlayerUsageRatesToDatabase()
    addInjuryReportsToDatabase()

    time.sleep(10)
    print(f"Training started at {datetime.datetime.now()}.")

    for stat in ['pts', 'ast', 'trb', '3p', '3pa', 'blk', 'fg', 'drb', 'fga', 'ft', 'fp', 'orb', 'tov', 'stl']:
        start_time = time.time()
        train_model_for_stat(stat)
        end_time = time.time()
        finish = (end_time - start_time) / 60
        print(f"Training finished in: {finish:.4f} minutes.")

# schedule.every().day.at("00:00").do(start_training)

# while True:
#     schedule.run_pending()
#     time.sleep(60)

start_training()