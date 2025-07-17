import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from cricket_com_au import main

st.title("cricket.com.au PlayGround")

match_list_df = pd.read_csv('./cricket.com.au_2009_2025.csv')
match_list_df = match_list_df[match_list_df['year']>2018]
match_list_df = match_list_df[(match_list_df['isCompleted'] | match_list_df['isLive'] | (match_list_df['gameStatusId'] == 'Stumps') )]

if 'format_type' not in st.session_state:
    st.session_state.format_type = None

if st.session_state.format_type:
    update_button = st.button('Update Data')
    if update_button:
        with st.spinner("Updating data... Please wait."):
            main(st.session_state.format_type.lower())
        st.success("Update completed successfully!")

format_type = st.selectbox(
    "Format",
    match_list_df['gameType'].dropna().unique(),
    index=None,
    placeholder="Select format type",
    key='format_type'
)

if(format_type):
    format_df = match_list_df[match_list_df['gameType'] == format_type]
    unique_competitions = format_df[['competition_name', 'year', 'competition_id',]].drop_duplicates()
    unique_competitions = unique_competitions.iloc[::-1]
    unique_competitions.loc[:, 'Display'] = unique_competitions.apply(lambda x: f"[{x['year']}] {x['competition_name']} ({x['competition_id']})", axis=1)
    comp_name = st.selectbox(
        "Series",
        unique_competitions['Display'],
        index=None,
        placeholder='Select series name'
    )

    if(comp_name):
        selected_competition_id = int(comp_name.split("(")[-1][:-1])
        series_df = format_df[format_df['competition_id'] == selected_competition_id]
        series_df = series_df.copy()
        series_df['name'] = series_df['name'].fillna('')
        series_df['Display'] = series_df.apply(lambda x: f"{x['name']} {x['homeTeam_name']} v {x['awayTeam_name']} ({x['id']})", axis=1)
        if(comp_name):
            match_name = st.selectbox(
                "Match",
                series_df['Display'].unique(),
                index=None,
                placeholder='Select match name'
            )


def false_shot_pitch_map(df):

    df = df[df['battingConnectionId'].isin(['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge', 'Gloved'])]
    heatmap_data = df.pivot_table(index="lengthTypeId", columns="lineTypeId", aggfunc="size", fill_value=0)
    length_order = ['FullToss', 'Yorker', 'HalfVolley', 'LengthBall', 'BackOfALength', 'Short' ]
    line_order = ['Wide', 'OutsideOff', 'OffStump', 'MiddleStump', 'LegStump', 'DownLeg', 'WideDownLeg']
    heatmap_data = heatmap_data.reindex(index=length_order, columns=line_order).fillna(0)
    heatmap_data = heatmap_data.astype(int)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.xaxis.tick_top()
    ax = sns.heatmap(heatmap_data, annot=True, fmt="d", cmap='coolwarm')
    plt.title('FalseShot HeatMap')
    st.pyplot(plt)
        
def pitch_map(df):
    line_rename_map = {
        'Wide': 'W', 'OutsideOff': 'OO', 'OffStump': 'Off',
        'MiddleStump': 'Mid', 'LegStump': 'Leg', 'DownLeg': 'DL', 'WideDownLeg': 'WDL'
    }

    length_order = ['FullToss', 'Yorker', 'HalfVolley', 'LengthBall', 'BackOfALength', 'Short']
    line_order = ['Wide', 'OutsideOff', 'OffStump', 'MiddleStump', 'LegStump', 'DownLeg', 'WideDownLeg']

    unique_bowlers = df['bowlerPlayerName'].dropna().unique()
    # unique_pairs = df[['inningNumber', 'bowlerPlayerName']].dropna().drop_duplicates()
    n_bowlers = len(unique_bowlers)
    n_cols = 3  
    n_rows = -(-n_bowlers // n_cols)  # Equivalent to math.ceil(n_bowlers / n_cols)

    if n_rows == 0:
        return

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    heatmap_data_list = []

    # **Single Pass to Compute Heatmaps & Find Global Min/Max**
    for bowler in unique_bowlers:
    # for inning, bowler in unique_pairs.itertuples(index=False):
        bowler_df = df[df['bowlerPlayerName'] == bowler]
        # bowler_df = df[(df['inningNumber'] == inning) & (df['bowlerPlayerName'] == bowler)]
        # **Optimized Counting Method (Faster than pivot_table)**

        # **Ensure Proper Ordering Without Multiple reindex() Calls**
        if(df['lineTypeId'] == 'Null').any():
            heatmap_data = bowler_df['lengthTypeId'].value_counts().reindex(length_order, fill_value=0).astype(int)
            heatmap_data = heatmap_data.to_frame()
        else:
            heatmap_data = bowler_df[['lengthTypeId', 'lineTypeId']].value_counts().unstack(fill_value=0)
            heatmap_data = heatmap_data.reindex(index=length_order, columns=line_order).fillna(0).astype(int)
            heatmap_data = heatmap_data.rename(columns=line_rename_map)

        heatmap_data_list.append(heatmap_data)

    # **Find Global Min/Max in One Go (Vectorized)**
    global_min = np.min([hm.values.min() for hm in heatmap_data_list])
    global_max = np.max([hm.values.max() for hm in heatmap_data_list])

    # **Plot Heatmaps**
    for i, (heatmap_data, bowler) in enumerate(zip(heatmap_data_list, unique_bowlers)):
    # for inning, bowler in unique_pairs.itertuples(index=False):
    # for i, (heatmap_data, row) in enumerate(zip(heatmap_data_list, unique_pairs.itertuples(index=False))):
        ax = axes[i]
        sns.heatmap(heatmap_data, fmt="d", cmap='coolwarm', vmin=global_min, vmax=global_max, ax=ax, annot=False)
        ax.set_title(f'{bowler}')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.tick_top()

    # **Hide Extra Subplots (If Any)**
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    bowling_teams = df['bowlingTeamName'].dropna().unique()
    innings = df['inningNumber'].dropna().unique()

    if bowling_teams.size > 0 and innings.size > 0:
        plt.suptitle(f"{bowling_teams[0]} Innings {innings[0]}", fontsize=16)
    elif innings.size > 0:
        plt.suptitle(f"Innings {innings[0]}", fontsize=16)
    elif bowling_teams.size > 0:
        plt.suptitle(f"{bowling_teams[0]}", fontsize=16)

    plt.tight_layout()
    st.pyplot(fig)


def wagon_wheel_graph(wagon_wheel_df):

    wagon_wheel_df['Zone'] = wagon_wheel_df.apply(
        lambda x: 1 if 0 <= x['shotAngle'] <= 45 and x['shotMagnitude'] > 0
        else 2 if 45 < x['shotAngle'] <= 90
        else 3 if 90 < x['shotAngle'] <= 135
        else 4 if 135 < x['shotAngle'] <= 180
        else 5 if 180 < x['shotAngle'] <= 225
        else 6 if 225 < x['shotAngle'] <= 270
        else 7 if 270 < x['shotAngle'] <= 315
        else 8 if 315 < x['shotAngle'] <= 360
        else 0, axis=1
    )
    
    temp_df = wagon_wheel_df[['shotAngle', 'shotMagnitude', 'runsScored', 'dismissalType']]
    temp_df['shotAngle'] = np.deg2rad(temp_df['shotAngle'])

    wicket_df = temp_df[temp_df['dismissalType'].notna()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'polar': True})

    # fig = plt.figure(figsize=(6,6))
    # ax = plt.subplot(111, polar=True)
    for angle, mag, run in zip(temp_df['shotAngle'], temp_df['shotMagnitude'], temp_df['runsScored']):
        if run == 1 or run == 2 or run == 3:
            ax1.plot([angle, angle], [0, mag], color='blue')
        elif run == 4:
            ax1.plot([angle, angle], [0, mag], color='green')
        elif run == 5:
            ax1.plot([angle, angle], [0, mag], color='orange')
        elif run == 6:
            ax1.plot([angle, angle], [0, mag], color='red')

    ax1.scatter(wicket_df['shotAngle'], wicket_df['shotMagnitude'], color='black', marker='.', zorder= 4)
    # Optional: Set the direction and location of 0 degrees
    ax1.set_theta_zero_location('E')  # 0 degrees at the top
    ax1.set_theta_direction(1)       # Clockwise

    blue_line, = ax1.plot([], [], color='blue', label='1s, 2s, 3s')
    green_line, = ax1.plot([], [], color='green', label='4s')
    orange_line, = ax1.plot([], [], color='orange', label='5s')
    red_line, = ax1.plot([], [], color='red', label='6s')
    black_dot = ax1.scatter([], [], color='black', marker='.', label='Wickets')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_facecolor('lightgreen')
    fig.patch.set_visible(False) 

    # plt.title("Polar Plot of Magnitude vs Angle")
    # plt.show()
    # st.pyplot(plt)



    total_runs = wagon_wheel_df['runsScored'].sum()
    wagon_wheel_df['isWide'] = wagon_wheel_df['isWide'].astype(bool)
    wagon_wheel_df['validDelivery'] = ~(wagon_wheel_df['isWide'].fillna(False))
    zone_perc_df = wagon_wheel_df[wagon_wheel_df['Zone'].between(1, 8)].groupby(['Zone']).agg(
        zone_runs_perc = ('runsScored', lambda x: (x.sum()/total_runs)*100),
        zone_outs = ('dismissalTypeId', lambda x: x.notna().sum()),
        zone_balls = ('validDelivery', 'sum'),
    )

    zone_perc_df['Avg'] = (zone_perc_df['zone_runs_perc']*total_runs*0.01)/zone_perc_df['zone_outs']
    zone_perc_df['S/R'] = np.where(zone_perc_df['zone_balls'] == 0, np.nan, ((zone_perc_df['zone_runs_perc']*total_runs*0.01)/zone_perc_df['zone_balls'])*100)


    zones = [1, 2, 3, 4, 5, 6, 7, 8]

    zone_perc_df = zone_perc_df.reindex(zones)
    # Fill only runs with 0; leave Avg/SR as NaN if missing
    zone_perc_df['zone_runs_perc'] = zone_perc_df['zone_runs_perc'].fillna(0)
    
    runs = zone_perc_df['zone_runs_perc'].to_list()
    avgs = zone_perc_df['Avg'].to_list()
    strikerates = zone_perc_df['S/R'].to_list()

    angles = np.linspace(np.pi/8, 17/8 * np.pi, 8, endpoint=False)

    # Matplotlib requires bars to loop around circle, so repeat the first value
    zones += [zones[0]]
    runs += [runs[0]]
    angles = np.append(angles, angles[0])
    avgs += [avgs[0]]
    strikerates += [strikerates[0]]

    # Plot
    # fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    bars = ax2.bar(angles, runs, width=np.pi/4, color='skyblue', edgecolor='black')

    label_radius = zone_perc_df['zone_runs_perc'].max() * 0.6
    for angle, run, zone, avg, sr in zip(angles, runs, zones, avgs, strikerates):
        # Radial position: a bit lower than the top of the bar
        # r = run * 0.5
        ax2.text(angle, label_radius, f'{run:.2f}%\n{avg:.2f} Avg\n{sr:.2f} S/R', ha='center', va='center', fontsize=10, rotation=0, rotation_mode='anchor', color='black')

        
    ax2.set_theta_zero_location('E')  # 0 degrees at the top
    ax2.set_theta_direction(1)

    ax2.set_xticklabels([])
    ax2.set_yticks([])  # Hide radial labels
    fig.patch.set_visible(False) 
    ax2.set_facecolor('lightgreen')

    st.pyplot(fig)

if(format_type and comp_name and match_name):
    selected_match_id = int(match_name.split("(")[-1][:-1])
    match_info_df = match_list_df[match_list_df['id'] == selected_match_id]
    
    file_name = "./processed_matches/" +  format_type.lower() + "/" + ("women/" if match_info_df['isWomensMatch'].iloc[0] else "men/") + str(selected_match_id) + ".csv"
    if os.path.exists(file_name):
        match_df = pd.read_csv(f'{file_name}', low_memory=False)

        selected_bowl_teams = st.multiselect(
            'Bowling Team',
            match_df['bowlingTeamName'].unique(),
            default=match_df['bowlingTeamName'].unique(),
            placeholder="Choose an option"
        )

        match_bowling_df = match_df[(match_df['bowlingTeamName'].isin(selected_bowl_teams))]

        if selected_bowl_teams:
            selected_innings = st.multiselect(
                'Innings',
                match_bowling_df['inningNumber'].unique(),
                default=match_bowling_df['inningNumber'].unique(),
                placeholder="Choose an option"
            )

            match_bowling_df = match_bowling_df[(match_bowling_df['inningNumber'].isin(selected_innings))]

            if selected_innings:
                max_overs = match_bowling_df['over_overNumber'].max()
                over_range = st.slider("Overs", 0, max_overs, (0, max_overs))
                match_bowling_df = match_bowling_df[match_bowling_df['over_overNumber'].between(over_range[0], over_range[1])]
        # else:
            # match_bowling_df = pd.DataFrame()

        if(len(match_bowling_df) > 0):

            false_shots_mask = match_bowling_df['battingConnectionId'].isin(['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge', 'Gloved'])
            
            false_shot_list = ['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge', 'Gloved']
            # false_shot_list = ['Missed', 'LeadingEdge', 'ThickEdge', 'BottomEdge', 'OutsideEdge', 'TopEdge', 'InsideEdge']
            
            incontrol_shot_list = ['WellTimed', 'Left']
            
            spin = ['Orthodox', 'Unorthodox', 'OffSpin', 'LegSpin']

            match_bowling_df = match_bowling_df.copy()
            # match_bowling_df['BowlingType'] = np.where(match_bowling_df['bowlingTypeId'].isin(spin), "Spin", "Pace")
            match_bowling_df['BowlingType'] = np.where(pd.isna(match_bowling_df['bowlingTypeId']), None, np.where(match_bowling_df['bowlingTypeId'].isin(spin), "Spin", "Pace"))
            # false_shot_df = match_df[false_shots_mask].groupby(["bowlerPlayerName", "bowlingTeamName"]).size().reset_index(name='FalseShots')
            # total_shots_df = match_df.groupby(["bowlerPlayerName", "bowlingTeamName"]).size().reset_index(name='TotalDeliveries')
            # false_shot_df['FalseShot%'] = (false_shot_df['FalseShots'] / false_shot_df['TotalDeliveries'])*100.0
            # false_shot_df = false_shot_df.sort_values(by="bowlingTeamName", ascending=False)
            def calculate_overs(df):
                legal_deliveries = df[~(df['isWide'] | df['isNoBall'])].shape[0]  # Exclude wides & no-balls
                full_overs = legal_deliveries // 6
                remaining_balls = legal_deliveries % 6
                return full_overs + (remaining_balls * 0.1)

            match_bowling_df['dismissalTypeId'] = match_bowling_df.get('dismissalTypeId', pd.NA)
            bowl_team_stats = match_bowling_df.groupby(["inningNumber", "BowlingType","bowlingTeamName"]).agg(
                Overs = ('ballNumber', lambda x: calculate_overs(match_bowling_df.loc[x.index])),
                TotalDeliveries=('ballNumber', 'count'),  # Total balls bowled
                RunsConceded=('runsConceded', 'sum'),  # Total Runs Given
                # Wickets=('isWicket', lambda x: x.sum()) , # Count of non-null wickets
                Wickets = ('dismissalTypeId', lambda x: x.notna().sum() - x[x =='RunOut'].count()),
                FalseShots=('battingConnectionId', lambda x: x.isin(false_shot_list).sum()),  # Count False Shots
            ).reset_index()

            bowl_team_stats['BallsPerFalseShot'] = (bowl_team_stats['TotalDeliveries'] / bowl_team_stats['FalseShots'])
            # bowl_team_stats['BallsPerFalseShot'] = bowl_team_stats['BallsPerFalseShot'].round(2).astype(str)
            bowl_team_stats['RunsPerFalseShot'] = (bowl_team_stats['RunsConceded'] / bowl_team_stats['FalseShots'])

            bowl_team_stats['FalseShotPerDismissial'] = np.where(bowl_team_stats['Wickets'] == 0, np.inf, (bowl_team_stats['FalseShots'] / bowl_team_stats['Wickets']))
            # bowl_team_stats['RunsPerFalseShot'] = bowl_team_stats['RunsPerFalseShot'].round(2).astype(str)
            bowl_team_stats['FalseShot%'] = (bowl_team_stats['FalseShots'] / bowl_team_stats['TotalDeliveries']) * 100
            bowl_team_stats['S/R'] = np.where(bowl_team_stats['Wickets'] == 0, np.inf, (bowl_team_stats['TotalDeliveries'] / bowl_team_stats['Wickets']))
            bowl_team_stats['Avg'] = np.where(bowl_team_stats['Wickets'] == 0, np.inf, (bowl_team_stats['RunsConceded'] / bowl_team_stats['Wickets']))
            bowl_team_stats['Eco'] = np.where(bowl_team_stats['TotalDeliveries'] == 0, 0, ((bowl_team_stats['RunsConceded'] / bowl_team_stats['Overs'])))
            # bowl_team_stats['FalseShot%'] = bowl_team_stats['FalseShot%'].round(2).astype(str)
            # bowl_team_stats['RunsConceded'] = bowl_team_stats['RunsConceded'].round(2).astype(str)
            bowl_team_stats = bowl_team_stats.sort_values(by="inningNumber", ascending=True)

            column_renames = {
                'FalseShots': 'FS',
                'FalseShot%': 'FS%',
                'FalseShotPerDismissial': 'FS/D',
                'BallsPerFalseShot': 'B/FS',
                'RunsPerFalseShot': 'R/FS',
                'RunsConceded': 'Runs',
                'bowlingTeamName': 'BowlTeam',
                'BowlingType': 'BowlType',
                'bowlerPlayerName': 'BowlerName',
                'battingPlayerName': 'BatterName',
                'battingTeamName': 'BatterTeam',
                'inningNumber': 'Inn',
                'TotalDeliveries': 'Balls',
                'FF': 'FF%',
                'BF': 'BF%',
                'FFCtrl' : 'FFCtrl%',
                'BFCtrl' : 'BFCtrl%',
                'SpinCtrl' : 'SpinCtrl%',
                'PaceCtrl' : 'PaceCtrl%',
                'Dot': 'Dot%',
                'Boundary': 'Boundary%'
                # Add other renames as needed
            }
            bowl_team_stats_copy = bowl_team_stats.copy()
            bowl_team_stats_copy = bowl_team_stats_copy.rename(columns=column_renames)



            bowler_stats = match_bowling_df.groupby(["inningNumber", "bowlerPlayerName", "BowlingType", "bowlingTeamName"]).agg(
                Overs = ('ballNumber', lambda x: calculate_overs(match_bowling_df.loc[x.index])),
                TotalDeliveries=('ballNumber', 'count'),  # Total balls bowled
                RunsConceded=('runsConceded', 'sum'),  # Total Runs Given
                # Wickets=('isWicket', lambda x: x.sum()),  # Count of non-null wickets
                Wickets = ('dismissalTypeId', lambda x: x.notna().sum() - x[x =='RunOut'].count()),
                FalseShots=('battingConnectionId', lambda x: x.isin(false_shot_list).sum())  # Count False Shots
            ).reset_index()

            bowler_stats['BallsPerFalseShot'] = (bowler_stats['TotalDeliveries'] / bowler_stats['FalseShots'])
            # bowler_stats['BallsPerFalseShot'] = bowler_stats['BallsPerFalseShot'].round(2).astype(str)
            bowler_stats['RunsPerFalseShot'] = (bowler_stats['RunsConceded'] / bowler_stats['FalseShots'])

            bowler_stats['FalseShotPerDismissial'] = np.where(bowler_stats['Wickets'] == 0, np.inf, (bowler_stats['FalseShots'] / bowler_stats['Wickets']))
            # bowler_stats['RunsPerFalseShot'] = bowler_stats['RunsPerFalseShot'].round(2).astype(str)
            bowler_stats['FalseShot%'] = (bowler_stats['FalseShots'] / bowler_stats['TotalDeliveries']) * 100
            bowler_stats['S/R'] = np.where(bowler_stats['Wickets'] == 0, np.inf, (bowler_stats['TotalDeliveries'] / bowler_stats['Wickets']))
            bowler_stats['Avg'] = np.where(bowler_stats['Wickets'] == 0, np.inf, (bowler_stats['RunsConceded'] / bowler_stats['Wickets']))
            bowler_stats['Eco'] = np.where(bowler_stats['TotalDeliveries'] == 0, 0, ((bowler_stats['RunsConceded'] / bowler_stats['Overs'])))
            # bowler_stats['FalseShot%'] = bowler_stats['FalseShot%'].round(2).astype(str)
            # bowler_stats['RunsConceded'] = bowler_stats['RunsConceded'].round(2).astype(str)
            bowler_stats = bowler_stats.sort_values(by=["inningNumber", "bowlingTeamName", "BowlingType"], inplace=False, ascending=False)

            bowl_stats_copy = bowler_stats.copy()
            bowl_stats_copy = bowl_stats_copy.rename(columns=column_renames)

            def calculate_foot_ctrl(df, foottype):

                false_shot_list = ['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge', 'Gloved']
                # false_shot_list = ['Missed', 'LeadingEdge', 'ThickEdge', 'BottomEdge', 'OutsideEdge', 'TopEdge', 'InsideEdge']
                
                new_df = df[(df['battingFeetId'] == foottype) & (df['validDelivery'] == True)]

                if(len(new_df) == 0):
                    return np.nan
                total_balls = new_df['ballNumber'].count()
                false_shots_ff = new_df['battingConnectionId'].isin(false_shot_list).sum()
                return 100 - ((false_shots_ff / total_balls)*100)

            def calculate_bowl_type_ctrl(df, bowltype):

                false_shot_list = ['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge', 'Gloved']
                # false_shot_list = ['Missed', 'LeadingEdge', 'ThickEdge', 'BottomEdge', 'OutsideEdge', 'TopEdge', 'InsideEdge']

                new_df = df[(df['BowlingType'] == bowltype) & (df['validDelivery'] == True)]

                if(len(new_df) == 0):
                    return np.nan
                total_balls = new_df['ballNumber'].count()
                false_shots_ff = new_df['battingConnectionId'].isin(false_shot_list).sum()
                return 100 - ((false_shots_ff / total_balls)*100)
            
            def calculate_best_shot(df):
                
                # most_common_shot = df['battingShotTypeId'].mode()[0]
                df = df[df['validDelivery'] == True]
                runs_per_shot = df.groupby(['battingShotTypeId'])['runsScored'].sum()
                max_shot = runs_per_shot.idxmax()

                return max_shot


            match_bowling_df_copy = match_bowling_df.copy()
            match_bowling_df_copy['isWide'] = match_bowling_df_copy['isWide'].astype(bool)
            match_bowling_df_copy['validDelivery'] = ~(match_bowling_df_copy['isWide'].fillna(False))
            # match_bowling_df_copy['validDelivery'] = np.where(~(match_bowling_df_copy['isWide'].fillna(False) | match_bowling_df_copy['isNoBall'].fillna(False)), True, False)
            batter_stats = match_bowling_df_copy.groupby(["inningNumber", "battingPlayerName", "battingTeamName"]).agg(
                Runs=('runsScored', 'sum'),
                TotalDeliveries=('validDelivery', 'sum'),  # Total balls faced
                FalseShots = ('battingConnectionId', lambda x: x.isin(false_shot_list).sum()),
                FF = ('battingFeetId', lambda x: (x[x == "FrontFoot"].count()/x.notna().count())*100),
                FFCtrl = ('ballNumber', lambda x: calculate_foot_ctrl(match_bowling_df_copy.loc[x.index], "FrontFoot")),
                BFCtrl = ('ballNumber', lambda x: calculate_foot_ctrl(match_bowling_df_copy.loc[x.index], "BackFoot")),
                SpinCtrl = ('ballNumber', lambda x: calculate_bowl_type_ctrl(match_bowling_df_copy.loc[x.index], "Spin")),
                PaceCtrl = ('ballNumber', lambda x: calculate_bowl_type_ctrl(match_bowling_df_copy.loc[x.index], "Pace")),
                Dot = ('runsScored', lambda x: ((x[(x == 0) & (match_bowling_df_copy.loc[x.index, 'validDelivery'] == True)].count()) / (x[match_bowling_df_copy.loc[x.index, 'validDelivery'] == True].notna().count())) * 100),
                Boundary = ('runsScored', lambda x: ((x[(x >= 4) & (match_bowling_df_copy.loc[x.index, 'validDelivery'] == True)].count()) / (x[match_bowling_df_copy.loc[x.index, 'validDelivery'] == True].notna().count())) * 100),
                ProductiveShot = ('ballNumber', lambda x: calculate_best_shot(match_bowling_df_copy.loc[x.index]))
            ).reset_index()
            batter_stats['Control%'] = 100 - ((batter_stats['FalseShots'] / batter_stats['TotalDeliveries']) * 100)
            batter_stats['S/R'] = np.where(batter_stats['TotalDeliveries'] == 0, 0, ((batter_stats['Runs'] / batter_stats['TotalDeliveries'])*100))
            # batter_stats = batter_stats.sort_values(by=["battingTeamName"], inplace=False, ascending=False)
            batter_stats = batter_stats.sort_values(by=["inningNumber", "battingTeamName", "Runs"], inplace=False, ascending=False)
            batter_stats_copy = batter_stats.copy()
            batter_stats_copy = batter_stats_copy.rename(columns=column_renames)

            team_met, bowl_met, bat_met, bowl_pitch_map, bowl_spell_met, wagon_wheel = st.tabs(["Team Metrics", "Bowler Metrics", "Batters Metrics", "Bowler Pitch Map", "Bowler Spell Metrics", "Wagon Wheel/Scoring Areas"])


            with team_met:
                st.subheader("Team Metrics")
                st.dataframe(bowl_team_stats_copy.style.highlight_max(color='green', axis=0, subset=['FS', 'FS%'])
                            .highlight_min(color='green', axis=0, subset=['FS/D', 'B/FS', 'R/FS', 'S/R', 'Avg', 'Eco'])
                            .format({'Overs': '{:.1f}', 'Eco': '{:.2f}', 'S/R': '{:.2f}', 'Avg': '{:.2f}', 'FS/D': '{:.2f}', 'B/FS': '{:.2f}', 'R/FS': '{:.2f}', 'Runs': '{:.0f}', 'FS%': '{:.2f}'}), 
                            hide_index=True)
                
            with bowl_met:    
                st.subheader("Bowler Metrics")
                st.dataframe(bowl_stats_copy.style.highlight_max(color='green', axis=0, subset=['FS', 'FS%'])
                            .highlight_min(color='green', axis=0, subset=['FS/D', 'B/FS', 'R/FS', 'S/R', 'Avg', 'Eco'])
                            .format({'Overs': '{:.1f}', 'Eco': '{:.2f}', 'S/R': '{:.2f}', 'Avg': '{:.2f}', 'FS/D': '{:.2f}', 'B/FS': '{:.2f}', 'R/FS': '{:.2f}', 'Runs': '{:.0f}', 'FS%': '{:.2f}'}),
                            hide_index=True)
            
            with bat_met:
                st.subheader("Batters Metrics")
                # columns_to_check = ['Control%', 'FFCtrl%', 'BFCtrl%', 'Runs', 'S/R', 'SpinCtrl%', 'PaceCtrl%', 'Boundary%']
                # batter_stats_copy[columns_to_check] = batter_stats_copy[columns_to_check].where(pd.notna(batter_stats_copy[columns_to_check]), 0)
                st.dataframe(batter_stats_copy.style.highlight_max(color='green', axis=0, subset=['Control%', 'FFCtrl%', 'BFCtrl%', 'Runs', 'S/R', 'SpinCtrl%', 'PaceCtrl%', 'Boundary%'])
                            .highlight_max(color='red', axis=0, subset=['FS', 'Dot%'])
                            .format({'Control%' : '{:.2f}', 'FF%' : '{:.2f}', 'FFCtrl%' : '{:.2f}', 'BFCtrl%' : '{:.2f}', 'Runs' : '{:.0f}', 'S/R' : '{:.2f}', 'SpinCtrl%' : '{:.2f}', 'PaceCtrl%' : '{:.2f}', 'Dot%': '{:.2f}', 'Boundary%': '{:.2f}'}),
                            hide_index=True)
                
            st.caption("FS: False Shot, FS%: False Shot %, S/R: Balls Per Dismissal, Avg: Runs Per Dismissal, Eco: Economy")
            st.caption("R/FS: Runs Conceded Per False Shot, B/FS: Balls Per False Shot, FS/D: False Shots Per Dismissal")
            st.caption("FF%: Front Foot %, FFCtrl% : Percentage of balls in control among all balls played on Front Foot by that batter")
            st.caption("BFCtrl%, SpinCtrl%, PaceCtrl% : Similar Control Percentage to FFCtrl% for BackFoot, Spin and Pace respectively")

            # column_renames = {
            #     'FalseShots': 'FS',
            #     'FalseShot%': 'FS%',
            #     'FalseShotPerDismissial': 'FS/D',
            #     'BallsPerFalseShot': 'B/FS',
            #     'RunsPerFalseShot': 'R/FS',
            #     'RunsConceded': 'Runs',
            #     'bowlingTeamName': 'BowlTeam',
            #     'BowlingType': 'BowlType',
            #     'bowlerPlayerName': 'BowlerName',
            #     'inningNumber': 'Inns',
            #     'TotalDeliveries': 'Balls'
            #     # Add other renames as needed
            # }
            # bowler_stats = bowler_stats.rename(columns=column_renames)

            # col_groups = {
            #     "Performance Metrics": ["BowlerName", "BowlType", "BowlTeam","Overs", "Runs", "Wickets", "Eco", "S/R", "Avg"],
            #     "False Shot Analysis": ["BowlerName", "BowlType", "BowlTeam", "Balls", "FS", "FS%", "B/FS", "R/FS", "FS/D"]
            # }

            # # Display each group with the same styling
            # for group_name, columns in col_groups.items():
            #     st.subheader(group_name)
            #     # Only include columns that exist in the DataFrame
            #     valid_columns = [col for col in columns if col in bowler_stats.columns]
                
            #     st.dataframe(
            #         bowler_stats[valid_columns].style
            #         .highlight_max(color='green', axis=0, subset=[col for col in ['FS', 'FS%'] if col in valid_columns])
            #         .highlight_min(color='green', axis=0, subset=[col for col in ['FS/D', 'B/FS', 
            #                                                                     'R/FS', 'S/R', 'Avg', 'Eco'] if col in valid_columns])
            #         .format({
            #             'Overs': '{:.1f}', 
            #             'Eco': '{:.2f}', 
            #             'S/R': '{:.2f}', 
            #             'Avg': '{:.2f}', 
            #             'FS/D': '{:.2f}', 
            #             'B/FS': '{:.2f}', 
            #             'R/FS': '{:.2f}', 
            #             'Runs': '{:.2f}', 
            #             'FS%': '{:.2f}'
            #         })
            #     )

            with bowl_pitch_map:
                st.subheader("Bowler Pitch Map")

                for inning in selected_innings:
                    match_bowling_inning_df = match_bowling_df[match_bowling_df['inningNumber'] == inning]

                    if not match_bowling_inning_df.empty:
                        pitch_map(match_bowling_inning_df)


            
            # Bowler Spell Wise Stats

            match_spell_bowling_df = match_bowling_df.copy()

            def calculate_spell_overs(legal_deliveries):
                full_overs = legal_deliveries // 6
                remaining_balls = legal_deliveries % 6
                return full_overs + (remaining_balls * 0.1)

            def assign_spells(group):
                overs = group[['over_overNumber']].drop_duplicates().sort_values('over_overNumber')
                overs['diff'] = overs['over_overNumber'].diff().fillna(1)
                overs['is_new_spell'] = overs['diff'] > 2
                overs['SpellNumber'] = overs['is_new_spell'].cumsum() + 1
                return group.merge(overs[['over_overNumber', 'SpellNumber']], on='over_overNumber', how='left')

            match_spell_bowling_df = match_spell_bowling_df.groupby(['inningNumber', 'bowlerPlayerName'], group_keys=False).apply(assign_spells)

            match_spell_bowling_df['isWide'] = match_spell_bowling_df['isWide'].astype(bool)
            match_spell_bowling_df['isNoBall'] = match_spell_bowling_df['isNoBall'].astype(bool)
            match_spell_bowling_df['validDelivery'] = ~(match_spell_bowling_df['isWide'].fillna(False) | match_spell_bowling_df['isNoBall'].fillna(False))
            bowler_stats = match_spell_bowling_df.groupby(["inningNumber", "bowlerPlayerName", "SpellNumber", "BowlingType", "bowlingTeamName"]).agg(
                OverPhase = ('over_overNumber', lambda x: f"{str(x.min())} - {str(x.max())}"),
                Overs = ('validDelivery', lambda x: calculate_spell_overs(x.sum())),
                TotalDeliveries=('ballNumber', 'count'),  # Total balls bowled
                RunsConceded=('runsConceded', 'sum'),  # Total Runs Given
                # Wickets=('isWicket', lambda x: x.sum()),  # Count of non-null wickets
                Wickets = ('dismissalTypeId', lambda x: x.notna().sum() - x[x =='RunOut'].count()),
                FalseShots=('battingConnectionId', lambda x: x.isin(false_shot_list).sum())  # Count False Shots
            ).reset_index()


            bowler_stats['BallsPerFalseShot'] = (bowler_stats['TotalDeliveries'] / bowler_stats['FalseShots'])
            # bowler_stats['BallsPerFalseShot'] = bowler_stats['BallsPerFalseShot'].round(2).astype(str)
            bowler_stats['RunsPerFalseShot'] = (bowler_stats['RunsConceded'] / bowler_stats['FalseShots'])

            bowler_stats['FalseShotPerDismissial'] = np.where(bowler_stats['Wickets'] == 0, np.inf, (bowler_stats['FalseShots'] / bowler_stats['Wickets']))
            # bowler_stats['RunsPerFalseShot'] = bowler_stats['RunsPerFalseShot'].round(2).astype(str)
            bowler_stats['FalseShot%'] = (bowler_stats['FalseShots'] / bowler_stats['TotalDeliveries']) * 100
            bowler_stats['S/R'] = np.where(bowler_stats['Wickets'] == 0, np.inf, (bowler_stats['TotalDeliveries'] / bowler_stats['Wickets']))
            bowler_stats['Avg'] = np.where(bowler_stats['Wickets'] == 0, np.inf, (bowler_stats['RunsConceded'] / bowler_stats['Wickets']))
            bowler_stats['Eco'] = np.where(bowler_stats['TotalDeliveries'] == 0, 0, ((bowler_stats['RunsConceded'] / bowler_stats['Overs'])))
            # bowler_stats['FalseShot%'] = bowler_stats['FalseShot%'].round(2).astype(str)
            # bowler_stats['RunsConceded'] = bowler_stats['RunsConceded'].round(2).astype(str)
            bowler_stats = bowler_stats.sort_values(by=["inningNumber", "bowlingTeamName", "BowlingType"], inplace=False, ascending=False)

            bowl_stats_copy = bowler_stats.copy()
            bowl_stats_copy = bowl_stats_copy.rename(columns=column_renames)


            with bowl_spell_met:
                st.subheader("Bowler Spell Metrics")
                st.dataframe(bowl_stats_copy.style.highlight_max(color='green', axis=0, subset=['FS', 'FS%'])
                            .highlight_min(color='green', axis=0, subset=['FS/D', 'B/FS', 'R/FS', 'S/R', 'Avg', 'Eco'])
                            .format({'Overs': '{:.1f}', 'Eco': '{:.2f}', 'S/R': '{:.2f}', 'Avg': '{:.2f}', 'FS/D': '{:.2f}', 'B/FS': '{:.2f}', 'R/FS': '{:.2f}', 'Runs': '{:.0f}', 'FS%': '{:.2f}'}),
                            hide_index=True)

            with wagon_wheel:
                st.subheader("Wagon Wheel")

                wagon_wheel_df = match_bowling_df.copy()

                spin = ['Orthodox', 'Unorthodox', 'OffSpin', 'LegSpin']
                wagon_wheel_df['BowlingType'] = np.where(pd.isna(wagon_wheel_df['bowlingTypeId']), None, np.where(wagon_wheel_df['bowlingTypeId'].isin(spin), "Spin", "Pace"))

                
                # if len(selected_innings) > 0:
                wagon_wheel_selected_innings = st.selectbox('Innings',
                                                            selected_innings,
                                                            index=0,
                                                            placeholder='Select an Inning')
                

            
                # for inning in selected_innings:
                wagon_wheel_inning_df = wagon_wheel_df[wagon_wheel_df['inningNumber'] == wagon_wheel_selected_innings]

                if not wagon_wheel_inning_df.empty:

                    selected_batter = st.selectbox('Batters',
                        ['All'] + list(wagon_wheel_inning_df['battingPlayerName'].sort_values().dropna().unique()),
                        index=0,
                        placeholder= 'Select a Batter')
                    
                    if selected_batter != 'All':
                        wagon_wheel_inning_df = wagon_wheel_inning_df[wagon_wheel_inning_df['battingPlayerName'] == selected_batter]
                    
                    bowl_type = st.radio(
                            'Bowling Type',
                            ['All', 'Pace', 'Spin'],
                            horizontal=True
                        )
                    # selected_bowler = st.selectbox('Bowlers',
                    #     ['All'] + list(wagon_wheel_inning_df['bowlerPlayerName'].sort_values().dropna().unique()),
                    #     index=0,
                    #     placeholder= 'Select a Bowler')
                    
                    # if selected_bowler != 'All':
                    #     wagon_wheel_inning_df = wagon_wheel_inning_df[wagon_wheel_inning_df['bowlerPlayerName'] == selected_bowler]
                    

                    if bowl_type != 'All':
                        wagon_wheel_inning_df = wagon_wheel_inning_df[wagon_wheel_inning_df['BowlingType'] == bowl_type]

                    wagon_wheel_graph(wagon_wheel_inning_df)

                

        else:
            st.warning("Select a Team")

    else:
        st.warning("Match Data Not Found")