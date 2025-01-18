import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry
from datetime import datetime, timedelta
from unidecode import unidecode
import pandas as pd
import numpy as np
import itertools as it
import mysql.connector 
import smtplib 
from email.message import EmailMessage 

"""
Scrape topsport, 7bet, betsafe and optibet. 
Create a single DataFrame with all matches.
Save to database
Send arbitraged matches as mail

Current runtime: ~5 seconds (without rate_limit)
"""

logging.basicConfig(
    filename="arbitrage.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
# Suppress mysql.connector logs
logging.getLogger("mysql.connector").setLevel(logging.WARNING)


# Limit requests speed to 1 calls per second
CALLS = 1
RATE_LIMIT = 1

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def check_limit():
    ''' Empty function just to check for calls to API '''
    return


def read_topsport(events):
    today = datetime.today()
    matches_topsport = []
    for event in events:
        try:
            date = event.find('span', {'class':'prelive-event-date'}).text.lower()
            teams = event.find_all('div', {'class': 'prelive-outcome-buttons-item-title-overflow'})
            odds = event.find_all('span', "prelive-list-league-rate ml-1 h-font-secondary h-fs17 h-fw500")

            # Convert "Today" and "Tomorrow" to actual dates
            if "šiandien" in date:
                date = today.strftime("%Y-%m-%d ") + date.split(" ")[1]
            elif "rytoj" in date:
                tomorrow = today + timedelta(days=1)
                date = tomorrow.strftime("%Y-%m-%d ") + date.split(" ")[1]

            # Ensure we have both teams and their respective odds
            if (len(teams), len(odds)) == (2, 2) or (len(teams), len(odds)) == (3, 3):
                team1, team2 = [unidecode(teams[i].text.strip()) for i in [0, -1]]
                odds = tuple(float(odds[i].text) for i in range(len(teams)))
                
                if team1 == "Taip" or team2 == "Taip": # Skip extra bets with "yes" "no" options
                    continue
                
                if len(teams) == 2:
                    matches_topsport.append((date, (team1, team2), odds))
                else:
                    matches_topsport.append((date, (team1, "Draw" ,team2), odds))
        except: # Skip invalid events
            continue
    
    return matches_topsport

# Helper function for 7bet
def remove_duplicates(events, only_duplicates = False):
    duplicates = []
    for idx, event in enumerate(events):
        # Skip first event
        if idx == 0: 
            continue      
        if only_duplicates:
            # Intented usage is for all events to use this. But it doesn't work for some, for now
            if event['name'] == events[idx - 1]['name']:
                duplicates.append(events[idx - 1])
        else:
            # If an event has the same name as the previous one, remove it.
            if event['name'] == events[idx - 1]['name']: 
                events.remove(event)
    
    return duplicates if only_duplicates else events

def read_7bet(data, number_of_odds):
    # Only first 2 bets of each match have False, others True. Can also be used 'typeID'==1 or 3
    filtered_events = [(idx, event) for idx, event in enumerate(data['odds']) if event['typeId'] in [1, 2, 3]]
    
    # Desired odds have 1, 2 or 3 as typeId. But also each event has extra 1 and 3. We filter second 1 and 3.
    if number_of_odds == 3:
        first_odds = [event for idx, event in filtered_events if event['typeId'] == 1]
        first_odds = remove_duplicates(first_odds)
        second_odds = [event for idx, event in filtered_events if event['typeId'] == 2]  
        third_odds = [event for idx, event in filtered_events if event['typeId'] == 3]
        third_odds = remove_duplicates(third_odds)
        events = [i for i in zip(first_odds, second_odds, third_odds)]

    elif number_of_odds == 2:
        first_odds = [event for idx, event in filtered_events if event['typeId'] == 1]
        second_odds = [event for idx, event in filtered_events if event['typeId'] == 3]  
        first_odds = remove_duplicates(first_odds) 
        second_odds = remove_duplicates(second_odds) 
        events = [i for i in zip(first_odds, second_odds)]

    matches_bet7 = []
    # Extract only price and name from each event
    for group in events:
        if len(group) == 2 or len(group) == 3:
            # Extracting the first and second team's names and odds
            team1, team2 = [unidecode(group[i]['name']).replace(',', '') for i in [0, -1]]
            odds = tuple(round(group[i]['price'], 2) for i in range(len(group)))

            if len(group) == 3:
                # Check for middle team, skip if not Draw
                middle = group[1]['name']
                if middle == "Lygiosios":
                    middle = "Draw"
                else: # Something went wrong
                    continue
                # Append the tuple to the matches list
                matches_bet7.append(((team1, middle, team2), odds))
            else:
                # Append the tuple to the matches list
                matches_bet7.append(((team1, team2), odds))

    return matches_bet7

def read_betsafe(data):
    matches_betsafe = []

    # Iterate through all events (matches)
    for event in data["events"]:
        try:
            # Get date
            date_text = event["date_start"]
            date =  datetime.strptime(date_text, "%Y-%m-%dT%H:%M:%S.%fZ")
            date = date + timedelta(hours=2)  # Adjust to local time (UTC+2)
            date = date.strftime("%Y-%m-%d %H:%M") # Remove seconds to match other dataframes

            # Get teams and odds
            match_data = event["main_odds"]["main"].items()
            teams = [unidecode(match["team_name"]) for _, match in match_data]
            odds = tuple(round(match["odd_value"], 2) for _, match in match_data)

            # Unpack teams and odds depending on number of bets
            if len(odds) == 2:
                team1, team2 = teams
                matches_betsafe.append((date, (team1, team2), odds))

            elif len(odds) == 3:
                team1, _, team2 = teams
                matches_betsafe.append((date, (team1, "Draw" ,team2), odds))
        except:
            continue
    return matches_betsafe

def read_optibet(data):
    matches = []
    for i in range(len(data)):
        try:
            timestamp = data[i]['time'] # Time is given as timestamp
            date = datetime.fromtimestamp(timestamp)
            date = date.strftime("%Y-%m-%d %H:%M") # Remove seconds to match other dataframes

            # Skip extra bets
            if data[i]['games'][0]['type'] != "match":
                continue

            team1, team2 = data[i]['player1']['name'], data[i]['player2']['name']
            number_of_odds = len(data[i]['games'][0]['odds'])
            odds = tuple(data[i]['games'][0]['odds'][j]['value'] for j in range(number_of_odds))
            matches.append((date, (team1, team2), odds))
        except KeyError: # There are 2 extra events that are not matches
            continue
    return matches

async def scrape_topsport(url, session):
    check_limit()
    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'lxml')

                # Find all events
                events = soup.find_all('div', {'class': 'js-prelive-event-row'})
                matches = read_topsport(events)
                return matches
            else:
                logging.warning(f"Failed to fetch {url}: HTTP {response.status}")
    except aiohttp.ClientError as e:
        logging.warning(f"Request failed for {url}: {e}")

async def scrape_bookmaker(url, session, bookmaker, league):
    check_limit()
    try:
        config = bookmaker_configs.get(bookmaker)
        headers = config["headers"]
        params = config["params"]
        read_bookmaker = config["reader"]

        # Perform the request
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()

                if bookmaker == "7bet": # 7bet needs number of odds as parameter
                    number_of_odds = league_list_7bet[league][0]
                    matches = read_7bet(data, number_of_odds)
                else:
                    matches = read_bookmaker(data)
                return matches
            else:
                logging.warning(f"Error: {response.status}")

    except aiohttp.ClientError as e:
        logging.warning(f"Request failed for {url}: {e}")

async def scrape_all_sites(urls, leagues, bookmaker, session):
    logging.info(f"Started scraping {bookmaker}")

    # Since topsport needs soup, needs different function for scraping
    if bookmaker == "topsport":
        events = [scrape_topsport(url, session) for url in urls]
    else:
        events = [scrape_bookmaker(url, session, bookmaker, league) for url, league in zip(urls, leagues)]

    results = await asyncio.gather(*events)
    all_matches = []
    for matches, league in zip(results, leagues):
        if matches:  # Handle cases where matches might be None

            if bookmaker == "7bet": # Only bookmaker without date
                all_matches.extend([(teams, odds, league) for teams, odds in matches])
            else:
                all_matches.extend([(date, teams, odds, league) for date, teams, odds in matches])
            
            logging.info(f"Scraped {len(matches)} {league} matches")
        else:
            logging.info(f"No matches found in {league}")
    return all_matches


async def main(urls, leagues, bookmaker):
    async with aiohttp.ClientSession() as session:
        all_matches = await scrape_all_sites(urls, leagues, bookmaker, session)

        if bookmaker == "topsport":
            df_topsport = pd.DataFrame(all_matches, columns=["Date", "Teams", "Odds", "League"])
            return df_topsport
        elif bookmaker == "7bet":
            df_7bet = pd.DataFrame(all_matches, columns=["Teams", "Odds", "League"])
            return df_7bet
        elif bookmaker == "betsafe":
            df_betsafe = pd.DataFrame(all_matches, columns=["Date", "Teams", "Odds", "League"])
            return df_betsafe
        elif bookmaker == "optibet":
            df_optibet = pd.DataFrame(all_matches, columns=["Date", "Teams", "Odds", "League"])
            return df_optibet


# Topsport scraping information
url_list_topsport = [
    "https://www.topsport.lt/krepsinis/nba",
    "https://www.topsport.lt/krepsinis/eurolyga",
    "https://www.topsport.lt/futbolas/uefa-europos-lyga",
    "https://www.topsport.lt/uefa-europos-konferenciju-lyga",
    "https://www.topsport.lt/amerikietiskas-futbolas/jav",
    "https://www.topsport.lt/ledo-ritulys/nhl",
    "https://www.topsport.lt/odds/all/4/6",  # Australian Tennis Open (M/F)
    "https://www.topsport.lt/tinklinis"  # Volleyball (All)
]
league_names_topsport = ['NBA', 'Eurolyga', 'UEFA', 'UEFA Konf.', "NFL", "NHL", "Tenisas", "Tinklinis"]

# 7bet scraping information
# Each url is url_body + (league_code)
url_body_7bet = ("https://sb2frontend-altenar2.biahosted.com/api/widget/"
            "GetEvents?culture=lt-LT&timezoneOffset=-120&integration=7bet&deviceType=1&"
            "numFormat=en-GB&countryCode=LT&eventCount=0&sportId=0&champIds=")

# Define leagues. Integers represent number of odds in each league
league_list_7bet = {
    "NBA": (2, "2980"),
    "Eurolyga": (2, "2995"),
    "UEFA": (3, "16809"),
    "UEFA Konf.": (3, "31608"),
    "NFL": (2, "3281"),
    "Tenisas": (2, "3013"), # ATP Males
    "Tenisas (F)": (2, "3036"), # ATP Females
    # NHL and AHL have 2 odds but order in the api is different, will be fixed later
    # "NHL": (2, "3232"), 
    # "AHL": (2, "3233"),
    "Tinklinis": (2,"1020"), # Europe
    }

league_names_7bet = {league: [inc, url_body_7bet + url_tail] 
                     for league, (inc, url_tail) in league_list_7bet.items()}

# Change exceptions
league_names_7bet['Tinklinis'][1] = league_names_7bet['Tinklinis'][1].replace("champIds", "catIds")

params_7bet = {
    "culture": "lt-LT",
    "timezoneOffset": "-120",
    "integration": "7bet",
    "deviceType": "1",
    "numFormat": "en-GB",
    "countryCode": "LT",
    "eventCount": "0",
    "sportId": "0",
    "champIds": "2980"
}

# Add headers copied from browser
headers_7bet = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://7bet.lt/", 
    "Origin": "https://7bet.lt",
    "Accept": "application/json, text/plain, */*"
}

# Betsafe scraping information
url_beginning_betsafe = "https://pre-5o-sp.websbkt.com/cache/5/lt/lt/"
# Hidenseek changes about once a week. update it from below line
params_betsafe = {"hidenseek": "0cff8888409d9b67762bcc09ac337184edbc3f2e9634"}
hidenseek_betsafe = "=".join(*params_betsafe.items())
url_ending_betsafe = "/prematch-by-tournaments.json?" + hidenseek_betsafe

leagues_betsafe = {
    # Basketball
    "NBA": "11624",
    #Football
    "Eurolyga": "4723:31611", 
    "UEFA": "35",
    # "UEFA Konf.": "https://www.betsafe.lt/en/betting/football/europe/uefa-europa-conference-league",
    # American Football
    "NFL": "1628",
    # Ice hockey
    "NHL": "96",
    "AHL": "95",
    # Tennis
    "Tenisas": "907", # ATP Males
    "Tenisas (F)": "910", # ATP Females
    # Volleyball (All)
    "Tinklinis": "732:752:782:858:3708:10426:10484:13801:14593:17615", 
    }

url_list_betsafe = {league : url_beginning_betsafe + url_body + url_ending_betsafe for league, url_body in leagues_betsafe.items()}

headers_betsafe = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Referer": "https://www.betsafe.lt/",
    "Origin": "https://www.betsafe.lt",
}

# Optibet scraping information
url_beginning_optibet = "https://ensb-trading.optibet.lt/lt/events/group/"

leagues_optibet = {
    "NBA": "466",
    "Eurolyga": "494",
    "UEFA": "1546",
    # "UEFA Konf.": "", # No matches yet
    "NFL": "464",
    "NHL": "2255",
    "AHL": "981",
    "Tinklinis": "16609,18139,16474,4652,3116,4653,7156,3696,19910,2664,3114,18171,3075,8919,18101", # All Male leagues
    "Tinklinis (F)": "3620,17958,17961,17962,6492,18036,7227,2669,4715", # All Female leagues
}
url_list_optibet = {league : url_beginning_optibet + url_body for league, url_body in leagues_optibet.items()}

params_optibet = {
    "gameTypes": -1,
}

headers_optibet = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
}

bookmaker_configs = {
    "7bet": 
    {
        "headers": headers_7bet,
        "params": params_7bet,
        "reader": read_7bet,
    },
    "betsafe": 
    {
        "headers": headers_betsafe,
        "params": params_betsafe,
        "reader": read_betsafe
    },
    "optibet": 
    {
        "headers": headers_optibet,
        "params": params_optibet,
        "reader": read_optibet,
    }
    }

# Helper Function for teams_match
def find_common_words(teams1, teams2):
    # Finds if there is any word in team2 that is also in team1
    
    match_bools = [False for _ in teams1]
    for idx, team in enumerate(teams2):
        team_split = team.split()
        for word in team_split:
            try:
                if word in teams1[idx]:
                    match_bools[idx] = True
                    continue
            except:
                print(teams1, teams2, idx)
    return match_bools

# Handle tuples to match names from different sites
def teams_match(teams1, teams2):
    # Skip if lengths don't match
    if len(teams1) != len(teams2):
        return False

    # Remove words like "FC" since it's very common and code would still match "FC A" with "FC B"
    teams1 = [team.lower().replace("fc", "") for team in teams1]
    teams2 = [team.lower().replace("fc", "") for team in teams2]

    bools1 = find_common_words(teams1, teams2) 
    bools2 = find_common_words(teams2, teams1)

    # Here bool1 or bool3 represents if the first team of teams1 and teams2 has a common word
    result = all([bool1 or bool2 for bool1, bool2 in zip(bools1, bools2)])

    return result

# Define the function to merge dataframes
def merge_dataframes(dataframes):
    # Log the start of merging
    logging.info("Started merging dataframes")

    merged_data = []
    matched_indices = {name: set() for name in dataframes.keys()}  # Track matched row indices for each dataframe

    # Iterate over each dataframe as the "source"
    for source_name, source_df in dataframes.items():
        for source_index, source_row in source_df.iterrows():
            if source_index in matched_indices[source_name]:  # Skip already matched rows
                continue

            # Store the source row and matches from other dataframes
            match_results = {source_name: source_row}
            matched_indices[source_name].add(source_index)  # Mark the source row as matched

            # Iterate over all other dataframes for matching
            for target_name, target_df in dataframes.items():
                if source_name != target_name:  # Avoid self-matching
                    match = target_df[
                        ~target_df.index.isin(matched_indices[target_name]) &  # Exclude already matched rows
                        target_df['Teams'].apply(lambda x: teams_match(source_row['Teams'], x)) & # Check team match
                        (target_df['League'] == source_row['League'])  # Compare leagues
                    ]
                    if not match.empty:
                        match_results[target_name] = match.iloc[0]  # Take the first match
                        matched_indices[target_name].add(match.index[0])  # Mark the matched row as used

            # Append the match results
            merged_data.append(match_results)


    # Convert merged results into a DataFrame
    df = pd.DataFrame([
        {
            'Date': next((match[frame]['Date'] for frame in match if 'Date' in match[frame]), None),
            'Teams': next((match[frame]['Teams'] for frame in match if 'Teams' in match[frame]), None),
            'League': next((match[frame]['League'] for frame in match if 'League' in match[frame]), None),
            'Topsport': match.get('topsport', {}).get('Odds', None),
            'Betsafe': match.get('betsafe', {}).get('Odds', None),
            '7bet': match.get('7bet', {}).get('Odds', None),
            'Optibet': match.get('optibet', {}).get('Odds', None),
        }
        for match in merged_data
    ])
    logging.info(f"Merged {len(df)} matches")
    return df

def get_stakes(odds: list, profit: float, favor: bool = False, idx: int = None) -> np.ndarray:
    # Given odds, calculate normal or favored bets with desired profit

    odds = np.array(odds)
    arbitrage = np.sum(1/odds)

    if favor: # Calculate favored bets, profit will be (0, 0,...,profit,..., 0) at index idx
        total_stake = (profit / odds[idx]) / (1 - arbitrage)
        stakes = total_stake / odds
        stakes[idx] += profit / odds[idx]

    else: # Calculate normal bets, profit will be (profit, profit, ..., profit)
        total_stake = (profit * arbitrage) / (1 - arbitrage) # changed
        stakes = (total_stake + profit) / odds
    
    return stakes

def round_stakes(odds: list, max_stake: int = 100) -> dict:
    # Given odds, calculate normal or favored bets with rounded numbers

    good_stakes = dict() # Final dict of good stakes
    unique_proportions = set() # Store unique proportions. Multiples of it will give the same profit
    for i in range(200):
        profit = 0.1 + i / 10 # Desired profit from that bet
        stakes_favored = [] # Favored bets
        stakes_normal = [] # Normal bets (equal payouts)
        for idx, _ in enumerate(odds):
            stakes_favored.append(get_stakes(odds, profit, favor= True, idx= idx)) # Get favored bets
            stakes_normal.append(get_stakes(odds, profit)) # Get normal bets 

        # Check if results are close to integers and not already tracked
        for stakes in [stakes_favored, stakes_normal]:
            for idx, stake in enumerate(stakes):
                if np.sum(stake) > max_stake: # Skip if stake is bigger than max_stake
                    continue

                proportion = tuple(np.round(stake / stake.sum(), 3)) # Normalise stake to 3 decimal places
                if np.isclose(stake, np.round(stake), atol=0.0001).all() and proportion not in unique_proportions:
                    payoff = odds * stake - np.sum(stake)
                    # Round payoff and replace -0.0 and 0.0 with 0 (-0.0 is caused from rounding)
                    round_payoff = round_payoff = tuple(0 if abs(x) == 0.0 else x for x in map(float, np.round(payoff, 2)))
                    round_stake = tuple(map(int, np.round(stake)))
                    good_stakes[round_stake] =  round_payoff
                    unique_proportions.add(proportion) # Add successfull proportion to set

    return good_stakes

    

def display_result(row_index, match, odds, bookmaker_odds):
    # For displaying bookmakers, define dictionary
    bookmakers = {0: 'Topsport', 1: 'Betsafe', 2: '7Bet'}

    possible_stakes_dict = round_stakes(odds)

    # Skip failed arbitrages (margin was so small that rounded stakes isn't profitable)
    if not possible_stakes_dict:
        return
    
    stakes = possible_stakes_dict.keys()
    profits = possible_stakes_dict.values()

    result_str = f"Match: {' - '.join(match)}.\n"

    # Find which bookmakers have the odds
    selected_bookmakers = []
    for idx, odd in enumerate(odds):
        for bm_idx, bm_tuple in enumerate(bookmaker_odds):
            if bm_tuple: # Skip empty (None) tuples
                if odd == bm_tuple[idx]:  # Check if the odd matches the coordinate in tuple
                    selected_bookmakers.append(bookmakers[bm_idx])
                    break
    
    # Display the bets with bookmakers
    for stake, profit in zip(stakes, profits):
        result_str += (
            f"Bet {stake}, odds {[float(odd) for odd in odds]} in ({'-'.join(selected_bookmakers)}). "
            f'Profit: {profit} ,\n'
        )

    arbitrage_list.append((row_index, result_str))
    return

# Function to check for arbitrage opportunities
def check_arbitrage(row):
    # Extract odds for each bookmaker
    odds_top = row['Topsport']
    odds_betsafe = row['Betsafe']
    odds_7bet = row['7bet']

    # Create a list of tuples (bookmaker, odds)
    odds_list = []
    if odds_top is not None:
        odds_list.append((odds_top))
    if odds_betsafe is not None:
        odds_list.append((odds_betsafe))
    if odds_7bet is not None:
        odds_list.append((odds_7bet))

    odds_array = np.array(odds_list) 
    odds_array_t = np.transpose(odds_array)

    min_sum = 2  # Any absurd starting value will work. At least should be 1

    # Generate combinations where each bookmaker provides one odds value
    for combination in it.product(*odds_array_t):
        # Skip rows that has 0 bets or empty combination
        if any(odd == 0 for odd in combination) or not combination:
            continue # Skip combinations containing zero

        # Check if the total sum indicates an arbitrage opportunity
        total_sum = round(sum(1 / odd for odd in combination), 5)
        min_sum = min(min_sum, total_sum)
        if total_sum < 1: # If the total sum indicates an arbitrage opportunity (should be less than 1)
            bookmaker_odds = row['Topsport'], row['Betsafe'], row['7bet']
            display_result(row.name, row['Teams'], combination, bookmaker_odds)

    # Return the results and minimum arbitrage sum, or False if no arbitrage is found
    return min_sum

def arbitrage(df):
    # Log the start of arbitrage checking
    logging.info("Started checking arbitrages")

    df['best arbitrage'] = df.apply(check_arbitrage, axis=1, result_type='expand')
    # Extract Team1 and Team2 from the Teams column
    df['team1'] = df['Teams'].apply(lambda x: x[0])  # First team
    df['team2'] = df['Teams'].apply(lambda x: x[-1])  # Last team

    # Add a 'draw_possible' column for 3 bet matches
    df['draw_possible'] = df['Teams'].apply(lambda x: len(x) == 3)

    # Log number of matches with arbitrage
    logging.info(f"Found arbitrages for {len(df[df['best arbitrage'] < 1])} matches")


def save_to_database(df):
    # Log the start of database connection
    logging.info("Starting database connection")

    # Read database credentials
    with open("database_info.txt") as f: 
        text = f.read()
        HOST, USER, PASSWORD = text.split(',')

    try:
        # Connect to the database
        db = mysql.connector.connect(
            host=HOST,
            user=USER,
            password=PASSWORD,
            database="ArbitrageAnalysis"
        )
    except:
        logging.exception("Database connection failed ")

    cursor = db.cursor()

    new_matches_count = 0
    arbitrage_matches_index = set()
    inserted_row_count = 0

    for idx, row in df.iterrows():
        # Check if the match exists. Set best_arbitrage to 1 if it's None
        check_query = """
        SELECT match_id, COALESCE(best_arbitrage, 1) AS best_arbitrage FROM matches 
        WHERE team1 = %s AND team2 = %s AND match_date = %s
        """
        cursor.execute(check_query, (row['team1'], row['team2'], row['Date']))
        result = cursor.fetchone()
        
        # Check if match already exists in the matches table. If exists and has arbitrage, check if it's better arbitrage
        if result:  
            match_id, existing_arbitrage = result
            existing_arbitrage = round(existing_arbitrage, 5) # Round to avoid rounding error mismatch
            if row['best arbitrage'] < 1 and row['best arbitrage'] < existing_arbitrage:
                # Update the best_arbitrage value
                update_query = """
                UPDATE matches 
                SET best_arbitrage = %s 
                WHERE match_id = %s
                """
                cursor.execute(update_query, (row['best arbitrage'], match_id))
                arbitrage_matches_index.add(idx) # Add index of the updated matches

        else:  
            # If the match has arbitrage, add it's index.
            # Note that it may not have rounded arbitrage odds. So, the match may not be in arbitrage_list
            if row['best arbitrage'] < 1:
                arbitrage_matches_index.add(idx)

            # Insert new match
            insert_match_query = """
            INSERT INTO matches (team1, team2, league, match_date, draw_possible, best_arbitrage)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_match_query, (
                row['team1'], 
                row['team2'], 
                row['League'], 
                row['Date'], 
                row['draw_possible'],
                row['best arbitrage']
            ))
            match_id = cursor.lastrowid
            new_matches_count += 1

        # Insert match data into scraped_data
        insert_scraped_query = """
        INSERT INTO scraped_data (
            match_id, 
            team1, team2, league, 
            best_arbitrage, 
            odd1_topsport, odd_draw_topsport, odd2_topsport, 
            odd1_betsafe, odd_draw_betsafe, odd2_betsafe, 
            odd1_7bet, odd_draw_7bet, odd2_7bet,
            odd1_optibet, odd_draw_optibet, odd2_optibet
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(
            insert_scraped_query, 
            (
                match_id, 
                row['team1'], row['team2'], row['League'],
                row['best arbitrage'], 

                # Topsport
                row['Topsport'][0] if row['Topsport'] else None,
                row['Topsport'][1] if (row['draw_possible'] and row['Topsport']) else None,
                row['Topsport'][-1] if row['Topsport'] else None,

                # Betsafe
                row['Betsafe'][0] if row['Betsafe'] else None,
                row['Betsafe'][1] if (row['draw_possible'] and row['Betsafe']) else None,
                row['Betsafe'][-1] if row['Betsafe'] else None,

                # 7bet
                row['7bet'][0] if row['7bet'] else None,
                row['7bet'][1] if (row['draw_possible'] and row['7bet']) else None,
                row['7bet'][-1] if row['7bet'] else None,

                # Optibet
                row['Optibet'][0] if row['Optibet'] else None,
                row['Optibet'][1] if (row['draw_possible'] and row['Optibet']) else None,
                row['Optibet'][-1] if row['Optibet'] else None,

            )
        )
        inserted_row_count += 1

    # Commit changes
    db.commit()

    cursor.close()
    db.close()

    logging.info(f"Inserted {new_matches_count} new matches to 'matches' table")
    logging.info(f"Inserted {inserted_row_count} new rows to 'scraped_data' table")
    logging.info(f"Found {len(arbitrage_matches_index)} new matches with arbitrage")

    return arbitrage_matches_index

def send_mail(arbitrage_matches_index):
    # Send mail with the arbitrage opportunities

    # Extract email and password from mail.txt. It should be in the format "email,password"
    with open("mail.txt") as f: 
        text = f.read()
        mail, password = text.split(',')

    FROM = mail
    PASSWORD = password
    SUBJECT = "Arbitrages"
    TO = FROM

    # Get all arbitrages with rounded bets and isn't better than existing arbitrage
    arbitrages = [arbitrage[1] for arbitrage in arbitrage_list if arbitrage[0] in arbitrage_matches_index]

    if not arbitrages:
        logging.info("No new arbitrages found.")
        return

    msg = EmailMessage()
    msg.set_content('\n'.join(arbitrages))
    msg['Subject'] = SUBJECT
    msg['From'] = FROM
    msg['To'] = TO

    # Send the message via an SMTP server
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()  # Start TLS encryption
            s.login(FROM, PASSWORD) 
            s.send_message(msg)
        logging.info(f"{len(arbitrages)} matches sent successfully!")
    except:
        logging.exception(f"Failed to send email ")


if __name__ == "__main__":
    # Scrape 4 websites
    # Topsport
    urls, leagues = url_list_topsport, league_names_topsport
    df_topsport = asyncio.run(main(urls, leagues, "topsport"))

    # 7bet
    urls = [league_names_7bet[league][1] for league in league_names_7bet]
    leagues = league_names_7bet.keys()
    df_7bet = asyncio.run(main(urls, leagues, "7bet"))

    # Betsafe
    leagues, urls = zip(*url_list_betsafe.items())
    df_betsafe = asyncio.run(main(urls, leagues, "betsafe"))

    # Optibet
    leagues, urls = zip(*url_list_optibet.items())
    df_optibet = asyncio.run(main(urls, leagues, "optibet"))

    # List of dataframes and their names
    dataframes = {
        'topsport': df_topsport,
        'betsafe': df_betsafe,
        '7bet': df_7bet,
        'optibet': df_optibet,
    }

    arbitrage_list = []
    # Merge all dataframes to one
    df = merge_dataframes(dataframes)
    # Sort the DataFrame by 'Date' and then by 'League' within each league
    df = df.sort_values(by=['Date', 'League'], ascending=[True, True]).reset_index(drop=True)

    # Do arbitrage analysis
    arbitrage(df)

    # Save to database
    arbitrage_matches_index = save_to_database(df)

    # Send results as mail
    if arbitrage_list:
        send_mail(arbitrage_matches_index)
    else:
        logging.info("No arbitrage found.")