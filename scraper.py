import requests
from bs4 import BeautifulSoup, Comment
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from nba_api.stats.endpoints import leagueseasonmatchups, boxscoreusagev3
from nba_api.stats.static import teams
import urllib3
from dotenv import load_dotenv
import os
import pandas as pd
import unicodedata
import time
import random
from datetime import datetime, timedelta

load_dotenv()
client = MongoClient(os.getenv("DATABASE_URL"))

db = client['nba_database']

nba_teams = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BRK": "Brooklyn Nets",
    "CHO": "Charlotte Hornets",
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
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}

nba_teams_reversed = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

abbrev_fix = {
    "ATL": "ATL",
    "BOS": "BOS",
    "BRK": "BKN",
    "CHO": "CHA",
    "CHI": "CHI",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GSW": "GSW",
    "HOU": "HOU",
    "IND": "IND",
    "LAC": "LAC",
    "LAL": "LAL",
    "MEM": "MEM",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NOP": "NOP",
    "NYK": "NYK",
    "OKC": "OKC",
    "ORL": "ORL",
    "PHI": "PHI",
    "PHO": "PHX",
    "POR": "POR",
    "SAC": "SAC",
    "SAS": "SAS",
    "TOR": "TOR",
    "UTA": "UTA",
    "WAS": "WAS"
}


def fix_encoding(text):
    try:
        fixed_text = text.encode('latin1').decode('utf-8')
        return unicodedata.normalize('NFC', fixed_text)
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

def getRosterFromTeam(team):
    url = f"https://www.basketball-reference.com/teams/{team}/2025.html"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='roster')

        if not table:
            print("Table with id 'roster' not found!")
            return

        # Extract the data
        links = []
        names = []
        bios = []
        rows = table.find_all('tr')
        for row in rows:
            bio = []
            number = row.find('th')
            bio.append(number.get_text(strip=True))
            cells = row.find_all('td')
            for i in range(len(cells)):
                if i == 0:
                    target_cell = cells[i]
                    a_tag = target_cell.find('a', href=True)
                    if target_cell.find('span', class_='note'):
                        continue
                    names.append(fix_encoding(a_tag.get_text(strip=True)))
                    links.append(a_tag['href'])
                else:
                    bio.append(cells[i].get_text(strip=True))
            bios.append(bio)
    elif response.status_code == 429:
        print(f'Failed to connect: status code {response.status_code}')
        print(int(response.headers.get('Retry-After')) / 60.0)
        return
    else:
        print(f'Failed to connect: status code {response.status_code}')
        return
    
    bios.pop(0)

    return names, links, bios

# def getRosterFromTeam(team):
#     url = f"https://www.basketball-reference.com/teams/{team}/2025.html"
#     driver = webdriver.Chrome()
#     driver.get(url)

#     table = driver.find_element(By.ID, 'roster')

#     if not table:
#         print("Table with id 'roster' not found!")
#         return

#     # Extract the data
#     links = []
#     names = []
#     rows = table.find_elements(By.TAG_NAME, 'tr')
#     for row in rows:
#         cells = row.find_elements(By.TAG_NAME, 'td')
#         if len(cells) > 1:
#             target_cell = cells[0]
#             a_tag = target_cell.find_element(By.TAG_NAME, 'a')
#             names.append(unidecode(a_tag.text))
#             links.append(a_tag.get_attribute('href'))

#     return names, links

def getTeamStats(team):
    url = f"https://www.basketball-reference.com/teams/{team}/2025.html#team_and_opponent"

    for _ in range(3):
        try:
            driver = webdriver.Chrome()
            driver.get(url)
            break
        except (urllib3.exceptions.ProtocolError, WebDriverException) as e:
            print(f"Error: {e}")
            time.sleep(5)

    stats = []

    table = driver.find_element(By.ID, 'team_and_opponent')
    body = table.find_element(By.TAG_NAME, 'tbody')
    row = table.find_elements(By.TAG_NAME, 'tr')
    data = row[3].find_elements(By.TAG_NAME, 'td')

    for stat in data:
        stats.append(stat.text)

    driver.quit()
    return stats

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

    driver.quit()

    return injuredPlayers

def getPlayerCurrSeasonStats(playerHtml):
    url = f"https://www.basketball-reference.com{playerHtml}"
    response = requests.get(url)
    avgs = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            summary = soup.find('div', class_="stats_pullout")
            if not summary:
                raise ValueError("Summary not found")
            
            poptips1 = summary.find('div', class_="p1")
            poptips2 = summary.find('div', class_="p2")

            if not poptips1 or not poptips2:
                raise ValueError("Poptips not found")
            
            for poptip in poptips1.find_all('div'):
                stat = poptip.find('p')
                if stat:
                    avgs.append(stat.get_text(strip=True))

            for poptip in poptips2.find_all('div'):
                stat = poptip.find('p')
                if stat:
                    avgs.append(stat.get_text(strip=True))

        except ValueError as e:
            print(f"Error: {e} at URL: {url}")
        except Exception as e:
            print(f"Unexpected error: {e} at URL: {url}")

        return avgs
    else:
        print(f'Failed to connect: status code {response.status_code} at URL: {url}')
        if 'Retry-After' in response.headers:
            print(int(response.headers.get('Retry-After')) / 60.0)
        return

# def getPlayerGameStats():
#     url = 'https://www.nba.com/stats/players/boxscores'
#     driver = webdriver.Chrome()
#     driver.get(url)
#     gamelogs = []

#     try:
#         dropdown_div = driver.find_element(By.CLASS_NAME, "Pagination_pageDropdown__KgjBU")
#         dropdown_element = dropdown_div.find_element(By.CLASS_NAME, "DropDown_select__4pIg9")
#         dropdown = Select(dropdown_element)
#         dropdown.select_by_value('-1')

#         table = driver.find_element(By.CLASS_NAME, "Crom_table__p1iZz")
#         body = table.find_element(By.TAG_NAME, "tbody")
#         rows = body.find_elements(By.TAG_NAME, "tr")
        
#         for row in rows:
#             cols = row.find_elements(By.TAG_NAME, "td")
#             stats = [col.text for col in cols]
#             gamelogs.append(stats)
#             print(gamelogs)

#     except Exception as e:
#         print(f"Unexpected error: {e} at URL: {url}")

#     driver.quit()
#     return gamelogs

def getPlayerGameStats():
    url = "https://stats.nba.com/stats/playergamelogs"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Referer": "https://www.nba.com/",
    }
    
    params = {
        "Season": "2024-25",
        "SeasonType": "Regular Season"
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        headers = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]
        
        df = pd.DataFrame(rows, columns=headers)
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

def getPlayerMatchups():
    season = '2024-25'  # Season format: 'YYYY-YY'
    season_type = 'Regular Season'  # Options: 'Regular Season', 'Playoffs', etc.
    per_mode = 'Totals'  # Options: 'Totals', 'PerGame'

    matchup_data = leagueseasonmatchups.LeagueSeasonMatchups(
        season=season,
        season_type_playoffs=season_type,
        per_mode_simple=per_mode
    )

    df = pd.DataFrame(matchup_data.get_data_frames()[0])

    return df

def getPlayerUsagePercentages():

    usage_data = boxscoreusagev3.BoxScoreUsageV3(
    )

    df = pd.DataFrame(usage_data.get_data_frames()[0])

    return df

print(getPlayerUsagePercentages())


#adding players and teams to database

def addPlayersToDatabase():
    db['players'].drop()
    for team in nba_teams:
        names, links, bios = getRosterFromTeam(team)
        collection = db['players']
        time.sleep(random.uniform(1, 3))

        for name, link, bio in zip(names, links, bios):
            avgs = getPlayerCurrSeasonStats(link)
            if avgs:
                data = {
                    "name" : name,
                    "team" : abbrev_fix[team],
                    "bio" : {
                        "number" : bio[0],
                        "pos" : bio[1],
                        "ht" : bio[2],
                        "wt" : bio[3],
                        "bday" : bio[4],
                        "country" : bio[5],
                        "exp" : bio[6],
                        "college" : bio[7]
                    },
                    "curr_season_avgs" : {
                        "games" : avgs[0],
                        "pts" : avgs[1],
                        "trb" : avgs[2],
                        "ast" : avgs[3],
                        "fg%" : avgs[4],
                        "fg3%" : avgs[5],
                        "ft%" : avgs[6],
                        "efg%" : avgs[7]
                    }
                }
                collection.update_one({"name" : name}, {"$set" : data}, upsert=True)

            time.sleep(random.uniform(1,3))

def addTeamsToDatabase():
    db['teams'].drop()
    collection = db['teams']
    for key in nba_teams:
        teamStats = getTeamStats(key)
        time.sleep(5)
        data = {
            "name" : nba_teams[key],
            "abbrev" : abbrev_fix[key],
            "roster" : getRosterFromTeam(key)[0],
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

        time.sleep(5)

        collection.update_one({"abbrev" : abbrev_fix[key]}, {"$set" : data}, upsert=True)

def addInjuryReportsToDatabase():
    db['injuries'].drop()
    injuries = getInjuryReports()
    if injuries:
        for injury in injuries:
            if len(injury) == 4:
                date = datetime.strptime(injury[2], "%a, %b %d, %Y")
                if (datetime.now() - date).days > 14:
                    continue
                status = injury[3].split('(')
                data = {
                    "name" : injury[0],
                    "team" : abbrev_fix[nba_teams_reversed[injury[1]]],
                    "date" : injury[2],
                    "status" : status[0].strip()
                }
                collection = db['injuries']
                collection.update_one({"name" : injury[0], "date" : injury[2]}, {"$set" : data}, upsert=True)

def addPlayerGameLogsToDatabase():
    db['gamelogs'].drop()
    game_logs = getPlayerGameStats()
    collection = db['gamelogs']
    all_game_logs = []
    
    if game_logs is not None:
        for _, game in game_logs.iterrows():
            matchup = game['MATCHUP']
            if 'vs.' in matchup:
                opp = matchup.split(' vs. ')
                home = 1  # Home game
            else:
                opp = matchup.split(' @ ')
                home = 0  # Away game

            data = {
                "player": game['PLAYER_NAME'],
                "team_abbreviation": game['TEAM_ABBREVIATION'],
                "game_id": game['GAME_ID'],
                "game_date": game['GAME_DATE'],
                "opponent": opp[1],
                "h/a": home,
                "w/l": 1 if game['WL'] == 'W' else 0,
                "min": game['MIN'],
                "fgm": game['FGM'],
                "fga": game['FGA'],
                "fg%": game['FG_PCT'],
                "3pm": game['FG3M'],
                "3pa": game['FG3A'],
                "3p%": game['FG3_PCT'],
                "ftm": game['FTM'],
                "fta": game['FTA'],
                "ft%": game['FT_PCT'],
                "oreb": game['OREB'],
                "dreb": game['DREB'],
                "reb": game['REB'],
                "ast": game['AST'],
                "tov": game['TOV'],
                "stl": game['STL'],
                "blk": game['BLK'],
                "blka": game['BLKA'],
                "pf": game['PF'],
                "pfd": game['PFD'],
                "pts": game['PTS'],
                "plus_minus": game['PLUS_MINUS'],
                "fp": game['NBA_FANTASY_PTS'],
            }

            all_game_logs.append(data)
            
        collection.insert_many(all_game_logs)

def addPlayerMatchupsToDatabase():
    db["matchups"].drop()
    collection = db['matchups']
    all_matchup_data = []

    matchups = getPlayerMatchups()

    for _, matchup in matchups.iterrows():
        data = {
            "off_player": fix_encoding(matchup['OFF_PLAYER_NAME']),
            "def_player": fix_encoding(matchup['DEF_PLAYER_NAME']),
            "stats": {
                "gp": matchup['GP'],
                "matchup_min": matchup['MATCHUP_MIN'],
                "partial_poss": matchup['PARTIAL_POSS'],
                "pts": matchup['PLAYER_PTS'],
                "team_pts": matchup['TEAM_PTS'],
                "ast": matchup['MATCHUP_AST'],
                "tov": matchup['MATCHUP_TOV'],
                "blk": matchup['MATCHUP_BLK'],
                "fgm": matchup['MATCHUP_FGM'],
                "fga": matchup['MATCHUP_FGA'],
                "fg%": matchup['MATCHUP_FG_PCT'],
                "3pm": matchup['MATCHUP_FG3M'],
                "3pa": matchup['MATCHUP_FG3A'],
                "3p%": matchup['MATCHUP_FG3_PCT'],
                "ftm": matchup['MATCHUP_FTM'],
                "fta": matchup['MATCHUP_FTA'],
                "sfl": matchup['SFL']
            }
        }

        all_matchup_data.append(data)

    if all_matchup_data:
        collection.insert_many(all_matchup_data)

def addPlayerUsageRatesToDatabase():
    db['usages'].drop()
    collection = db['usages']
    for month in range(1, 13):
        usages = getPlayerUsagePercentages(month)
        for usage in usages:
            data = {
                "name": usage[1],
                "team": usage[2],
                "gp": usage[4],
                "w": usage[5],
                "l": usage[6],
                "min": usage[7],
                "usg%": usage[8],
                "%fgm": usage[9],
                "%fga": usage[10],
                "%3pm": usage[11],
                "%3pa": usage[12],
                "%ftm": usage[13],
                "%fta": usage[14],
                "%oreb": usage[15],
                "%dreb": usage[16],
                "%reb": usage[17],
                "%ast": usage[18],
                "%tov": usage[19],
                "%stl": usage[20],
                "%blk": usage[21],
                "%blka": usage[22],
                "%pf": usage[23],
                "%pfd": usage[24],
                "%pts": usage[25],
                'month': month_dict[month]
            }

            collection.insert_one(data)


# addTeamsToDatabase()
# addPlayersToDatabase()
# addPlayerGameLogsToDatabase()
# addPlayerMatchupsToDatabase()
# addInjuryReportsToDatabase()
# addPlayerUsageRatesToDatabase()