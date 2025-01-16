import requests
from bs4 import BeautifulSoup, Comment
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
client = MongoClient(os.getenv("DATABASE_URL"))

db = client['nba_database']
collection = db['gamelogs']

nba_teams = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}

def getRosterFromTeam(team):
    url = f"https://www.basketball-reference.com/teams/{team}/2025.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', id='roster')

    if not table:
        print("Table with id 'roster' not found!")

    # Extract the data
    links = []
    names = []
    rows = table.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) > 1:
            target_cell = cells[0]
            a_tag = target_cell.find('a', href=True)
            names.append(a_tag.get_text(strip=True))
            links.append(a_tag['href'])

    return names, links

def getTeamStats(team):
    url = f"https://www.basketball-reference.com/teams/{team}/2025.html#team_and_opponent"
    driver = webdriver.Chrome()
    driver.get(url)

    stats = []

    table = driver.find_element(By.ID, 'team_and_opponent')
    body = table.find_element(By.TAG_NAME, 'tbody')
    row = table.find_elements(By.TAG_NAME, 'tr')
    data = row[3].find_elements(By.TAG_NAME, 'td')

    for stat in data:
        stats.append(stat.text)
    return stats

teamStats = getTeamStats('CLE')
data = {
    "name" : nba_teams['CLE'],
    "roster" : getRosterFromTeam()[0],
    "teamStats" : {
        "MP": teamStats[1],
        "FG": teamStats[2],
        "FGA": teamStats[3],
        "FG%": teamStats[4],
        "3P": teamStats[5],
        "3PA": teamStats[6],
        "3P%": teamStats[7],
        "2P": teamStats[8],
        "2PA": teamStats[9],
        "2P%": teamStats[10],
        "FT": teamStats[11],
        "FTA": teamStats[12],
        "FT%": teamStats[13],
        "ORB": teamStats[14],
        "DRB": teamStats[15],
        "TRB": teamStats[16],
        "AST": teamStats[17],
        "STL": teamStats[18],
        "BLK": teamStats[19],
        "TOV": teamStats[20],
        "PF": teamStats[21],
        "PTS": teamStats[22]
    }
}

collection = db['teams']
collection.insert_one(data)

print('done')

def getInjuryReports():
    url = f"https://www.basketball-reference.com/friv/injuries.fcgi"
    driver = webdriver.Chrome()
    driver.get(url)

    injuredPlayers = []

    table = driver.find_element(By.ID, 'injuries')
    body = table.find_element(By.TAG_NAME, 'tbody')
    rows = body.find_elements(By.TAG_NAME, 'tr')

    for row in rows:
        injury = []
        heading = row.find_element(By.TAG_NAME, 'th')
        # name = heading.find_element(By.TAG_NAME, 'a')
        injury.append(heading.text)
        cols = row.find_elements(By.TAG_NAME, 'td')
        for col in cols:
            injury.append(col.text)

        injuredPlayers.append(injury)

    return injuredPlayers[:5]

# names, links = getRosterFromTeam('CLE')

# for name, link in zip(names, links):
#     print(name, link[11:])

# print(pd.DataFrame(getInjuryReports()))


def getPlayerGameStats(playerId):
    url = f'https://www.basketball-reference.com/players/a/{playerId}/gamelog/2025'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='pgl_basic')

        if not table:
            print("Table with id 'pgl_basic' not found.")
        
        gamelog = []
        rows = table.find_all('tr')
        for i in range(len(rows)):
            cells = rows[i].find_all('td')
            temp = []
            for cell in cells:
                a_tag = cell.find('a')
                if a_tag:
                    temp.append(a_tag.get_text(strip=True))
                else:
                    temp.append(cell.get_text(strip=True))

            gamelog.append(temp)
    return gamelog


# ['1', '2024-10-23', '26-185', 'CLE', '@', 'TOR', 'W (+30)', '1', '26:05', '5', '7', '.714', '0', '0', '', '4', '5', '.800', '2', '5', '7', '1', '1', '4', '1', '1', '14', '16.7', '+8']
# print(getPlayerGameStats('allenja01'))