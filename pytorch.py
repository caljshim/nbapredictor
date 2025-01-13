import requests
from bs4 import BeautifulSoup, Comment
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv("DATABASE_URL"))

db = client['nba_database']
collection = db['gamelogs']

nba_abbreviations = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
    "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
    "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS",
    "TOR", "UTA", "WAS"
]

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

# names, links = getRosterFromTeam('CLE')

# for name, link in zip(names, links):
#     print(name, link[11:])

print(getTeamStats('CLE'))


def getPlayerGameStats(name, playerId):
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