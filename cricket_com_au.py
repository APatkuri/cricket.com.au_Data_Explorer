import pandas as pd
import requests
from datetime import date
import os
import ast
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def preprocess(data):
    if 'inning' not in data:
        return pd.DataFrame()
        
    df = pd.json_normalize(data['inning'], 'overs', 
                         ['id', 'fixtureId', 'inningNumber', 'battingTeamId', 'bowlingTeamId'], 
                         record_prefix='over_')
    
    if len(df) == 0:
        return pd.DataFrame()
        
    df = df.iloc[:-1]  # Drop last row
    df = df.explode('over_balls', ignore_index=True)
    df1 = pd.json_normalize(df['over_balls'])
    df = pd.concat([df.drop(columns=['over_balls']), df1], axis=1)
    df = df.iloc[::-1].reset_index(drop=True)
    return df.drop(columns=['comments'], errors='ignore')

def fetch_innings_data(matchid, inning):
    try:
        url = f"https://apiv2.cricket.com.au/web/views/comments?fixtureId={matchid}&inningNumber={inning}&commentType=&overLimit=499&jsconfig=eccn%3Atrue&format=json"
        response = requests.get(url, timeout=100)
        return preprocess(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error match{matchid} inning{inning}: {e}")
        return pd.DataFrame()

def main_func(matchid, player_dict, team_dict, comp_dict, match_comp_data):
    # Fetch both innings in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        dfs = list(executor.map(partial(fetch_innings_data, matchid), range(1,5)))
    
    comms_df = pd.concat(dfs, ignore_index=True)
    
    if comms_df.empty:
        return comms_df
        
    print(f"Processing match: {matchid}")
    
    # Add all mappings in one go
    name_mappings = {
        'bowlerPlayerId': ('bowlerPlayerName', player_dict),
        'battingPlayerId': ('battingPlayerName', player_dict), 
        'nonStrikeBattingPlayerId': ('nonStrikeBattingPlayerName', player_dict),
        'dismissalPlayerId': ('dismissalPlayerName', player_dict),
        'battingTeamId': ('battingTeamName', team_dict),
        'bowlingTeamId': ('bowlingTeamName', team_dict),
        'fixtureId': ('fixtureName', comp_dict),
        'id': ('matchName', match_comp_data)
    }
    
    for id_col, (name_col, mapping_dict) in name_mappings.items():
        if id_col in comms_df.columns:
            try:
                comms_df.insert(comms_df.columns.get_loc(id_col) + 1, 
                              name_col, 
                              comms_df[id_col].map(mapping_dict))
            except Exception as e:
                print(f"Error mapping {id_col} to {name_col}: {e}")
                comms_df[name_col] = None

    return comms_df

def expand_df_columns(col_names, df):
    df_final = df.copy()
    
    for col in col_names:
        if col not in df_final.columns:
            continue
            
        # Convert string representations to dicts
        df_final[col] = df_final[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().lower() != 'nan' and x.strip() != '' else x
        )
        
        valid_items = [item for item in df_final[col].dropna() if item is not None]
        if not valid_items or not isinstance(valid_items[0], dict):
            continue
            
        # Normalize dictionary column
        df_temp = pd.json_normalize(df_final[col].dropna()).add_prefix(f"{col}_")
        non_null_indices = df_final[col].notna()
        df_temp.index = df_final.index[non_null_indices]
        
        # Merge normalized data
        df_final = df_final.drop(col, axis=1)
        for new_col in df_temp.columns:
            df_final[new_col] = pd.NA
            df_final.loc[df_temp.index, new_col] = df_temp[new_col]
    
    return df_final

def fetch_year_data(year, is_completed):
    url = f"https://apiv2.cricket.com.au/web/fixtures/yearfilter?isCompleted={is_completed}&year={year}&limit=999&isInningInclude=true&jsconfig=eccn%3Atrue&format=json"
    try:
        response = requests.get(url, timeout=100)
        data = response.json()
        return data.get('fixtures', [])
    except Exception as e:
        print(f"Error fetching data for year {year}, isCompleted={is_completed}: {e}")
        return []

def sched_func():
    current_year = date.today().year
    file_name = f'./cricket.com.au_2009_{current_year}.csv'
    
    # Load existing data if available
    if os.path.exists(file_name):
        old_df = pd.read_csv(file_name, low_memory=False)
        old_df['year'] = pd.to_datetime(old_df['startDateTime']).dt.year
        old_df = old_df[old_df['year'] < current_year]  # Keep only historical data
    else:
        old_df = pd.DataFrame()

    # Only fetch current and next year's data
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for year in range(current_year, current_year+2):
            for is_completed in ["true", "false"]:
                futures.append(executor.submit(fetch_year_data, year, is_completed))
                
        year_data = []
        for future in futures:
            year_data.extend(future.result())

    if year_data:
        year_data.sort(key=lambda x: x.get("startDateTime", ""))
        year_df = pd.DataFrame(year_data)
        
        if 'startDateTime' in year_df.columns:
            year_df['year'] = pd.to_datetime(year_df['startDateTime']).dt.year
            year_df = expand_df_columns(['competition', 'homeTeam', 'awayTeam', 'venue', 'innings'], year_df)
            
            # Combine historical data with new data
            final_df = pd.concat([old_df, year_df], ignore_index=True)
            final_df.drop_duplicates(subset=['id'], keep='last', inplace=True)
            final_df.to_csv(file_name, index=False)
            return final_df
    
    return old_df

def player_data():
    try:
        response = requests.get("https://apiv2.cricket.com.au/web/players/list?isActive=&&jsconfig=eccn%3Atrue&format=json", timeout=100)
        # pd.DataFrame(response.json()['players']).to_csv("./players.csv", index=False)
        players_df = pd.DataFrame(response.json()['players'])
        players_df = players_df.sort_values(by='id')
        players_df.to_csv("./players.csv", index=False)
    except Exception as e:
        print(f"Error fetching player data: {e}")

def process_match_batch(matches, player_dict, team_dict, comp_dict, match_comp_dict, format_dir, live_matches_file, year_df):
    for match_id in matches:
        try:
            output_file = os.path.join(format_dir, f"{match_id}.csv")
            
            # Check if match is in live matches
            live_matches = set()
            if os.path.exists(live_matches_file):
                with open(live_matches_file, 'r') as f:
                    live_matches = set(json.load(f))
            
            # Process if match is live or not yet processed
            if match_id in live_matches or not os.path.exists(output_file):
                match_df = main_func(match_id, player_dict, team_dict, comp_dict, match_comp_dict)
                if not match_df.empty:
                    match_df.to_csv(output_file, index=False)
                    print(f"Processed and saved match {match_id} successfully")
                    
                    # Update live matches list
                    if match_id in live_matches:
                        # Check if match is completed
                        match_status = year_df[year_df['id'] == match_id]['isCompleted'].iloc[0]
                        if match_status:
                            live_matches.remove(match_id)
                            print(f"Match {match_id} completed, removed from live matches")
                    else:
                        # Add to live matches if match is live
                        match_status = year_df[year_df['id'] == match_id]['isLive'].iloc[0]
                        if match_status:
                            live_matches.add(match_id)
                            print(f"Match {match_id} is live, added to live matches")
                    
                    # Save updated live matches list
                    with open(live_matches_file, 'w') as f:
                        json.dump(list(live_matches), f)
            else:
                print(f"Match {match_id} already processed and not live, skipping")
                
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")

def main():
    player_data()
    year_df = sched_func()
    
    if year_df is None or year_df.empty:
        print("Failed to retrieve schedule data, attempting to load from file")
        file_name = f'./cricket.com.au_2009_{date.today().year}.csv'
        if os.path.exists(file_name):
            year_df = pd.read_csv(file_name, low_memory=False)
        else:
            print("Error: Could not load schedule data")
            return

    # Load player data
    try:
        player_dict = dict(zip(pd.read_csv('./players.csv', low_memory=False)['id'], 
                             pd.read_csv('./players.csv', low_memory=False)['displayName']))
    except Exception as e:
        print(f"Error loading player data: {e}")
        player_dict = {}
    
    # Setup team dictionaries
    temp_year_df = year_df.dropna(subset=['homeTeam_id', 'awayTeam_id'])
    team_dict = {**dict(zip(temp_year_df['homeTeam_id'], temp_year_df['homeTeam_name'])),
                 **dict(zip(temp_year_df['awayTeam_id'], temp_year_df['awayTeam_name']))}
    
    # Setup competition dictionaries
    comp_dict = dict(zip(year_df['id'], year_df['competition_name'])) if 'competition_name' in year_df.columns else {}
    match_comp_dict = dict(zip(year_df['id'], year_df['name'])) if 'name' in year_df.columns else {}

    base_dir = "./processed_matches"
    os.makedirs(base_dir, exist_ok=True)

    # Process matches in parallel batches
    for game_type_id, format_name in [(2, 'odis')]:
        for is_womens in [True, False]:
            gender = 'women' if is_womens else 'men'
            format_dir = os.path.join(base_dir, format_name, gender)
            os.makedirs(format_dir, exist_ok=True)
            
            # Create live matches file for this format/gender
            live_matches_file = os.path.join(format_dir, 'live_matches.json')
            
            print(f"Processing {format_name} for {gender}")
            
            matches_df = year_df[
                (year_df['gameTypeId'] == game_type_id) & 
                ((year_df['isCompleted'] == True) | (year_df['isLive'] == True)) & 
                (year_df['year'] >= 2018) & 
                (year_df['isWomensMatch'] == is_womens)
            ]
            
            if matches_df.empty:
                print(f"No matches found for {format_name} {gender}")
                continue
                
            print(f"Found {len(matches_df)} matches to process")
            
            # Process matches in parallel batches
            batch_size = 10
            match_batches = [matches_df['id'].iloc[i:i+batch_size].tolist() 
                           for i in range(0, len(matches_df), batch_size)]
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for batch in match_batches:
                    futures.append(executor.submit(process_match_batch, batch, 
                                                player_dict, team_dict, comp_dict, 
                                                match_comp_dict, format_dir, live_matches_file, year_df))
                
                # Wait for all batches to complete
                for future in futures:
                    future.result()
            
            print(f"Completed processing all matches for {format_name} {gender}")

if __name__ == "__main__":
    main()
