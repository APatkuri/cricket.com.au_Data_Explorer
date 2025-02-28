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

update_button = st.button('Update Data')
if update_button:
    with st.spinner("Updating data... Please wait."):
        main()
    st.success("Update completed successfully!")

match_list_df = pd.read_csv('./cricket.com.au_2009_2025.csv')
match_list_df = match_list_df[match_list_df['year']>2018]
match_list_df = match_list_df[(match_list_df['isCompleted'] | match_list_df['isLive'])]

format_type = st.selectbox(
    "Format",
    match_list_df['gameType'].unique(),
    index=None,
    placeholder="Select format type"
)

if(format_type):
    format_df = match_list_df[match_list_df['gameType'] == format_type]
    unique_competitions = format_df[['competition_name', 'competition_id']].drop_duplicates()
    unique_competitions.loc[:, 'Display'] = unique_competitions.apply(lambda x: f"{x['competition_name']} ({x['competition_id']})", axis=1)
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
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge'])]
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

    # df = df[df['battingConnectionId'].isin(['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
    #                                     'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge'])]
    heatmap_data = df.pivot_table(index="lengthTypeId", columns="lineTypeId", aggfunc="size", fill_value=0)
    length_order = ['FullToss', 'Yorker', 'HalfVolley', 'LengthBall', 'BackOfALength', 'Short' ]
    line_order = ['Wide', 'OutsideOff', 'OffStump', 'MiddleStump', 'LegStump', 'DownLeg', 'WideDownLeg']
    heatmap_data = heatmap_data.reindex(index=length_order, columns=line_order).fillna(0)
    heatmap_data = heatmap_data.astype(int)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.xaxis.tick_top()
    ax = sns.heatmap(heatmap_data, annot=True, fmt="d", cmap='coolwarm')
    plt.title('HeatMap')
    st.pyplot(plt)

if(format_type and comp_name and match_name):
    selected_match_id = int(match_name.split("(")[-1][:-1])
    match_info_df = match_list_df[match_list_df['id'] == selected_match_id]
    
    file_name = "./processed_matches/" +  format_type.lower() + "s/" + ("women/" if match_info_df['isWomensMatch'].iloc[0] else "men/") + str(selected_match_id) + ".csv"
    if os.path.exists(file_name):
        match_df = pd.read_csv(f'{file_name}', low_memory=False)

        selected_bowl_teams = st.multiselect(
            'Bowling Team',
            match_df['bowlingTeamName'].unique(),
            default=match_df['bowlingTeamName'].unique(),
            placeholder="Choose an option"
        )

        match_bowling_df = match_df[match_df['bowlingTeamName'].isin(selected_bowl_teams)]

        if(len(match_bowling_df) > 0):

            false_shots_mask = match_bowling_df['battingConnectionId'].isin(['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge'])
            
            false_shot_list = ['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge']
            
            spin = ['Orthodox', 'Unorthodox', 'OffSpin', 'LegSpin']

            match_bowling_df['BowlingType'] = np.where(match_bowling_df['bowlingTypeId'].isin(spin), "Spin", "Pace")
            # false_shot_df = match_df[false_shots_mask].groupby(["bowlerPlayerName", "bowlingTeamName"]).size().reset_index(name='FalseShots')
            # total_shots_df = match_df.groupby(["bowlerPlayerName", "bowlingTeamName"]).size().reset_index(name='TotalDeliveries')
            # false_shot_df['FalseShot%'] = (false_shot_df['FalseShots'] / false_shot_df['TotalDeliveries'])*100.0
            # false_shot_df = false_shot_df.sort_values(by="bowlingTeamName", ascending=False)
            def calculate_overs(df):
                legal_deliveries = df[~(df['isWide'] | df['isNoBall'])].shape[0]  # Exclude wides & no-balls
                full_overs = legal_deliveries // 6
                remaining_balls = legal_deliveries % 6
                return full_overs + (remaining_balls / 10)

            bowl_team_stats = match_bowling_df.groupby(["inningNumber", "BowlingType","bowlingTeamName"]).agg(
                Overs = ('ballNumber', lambda x: calculate_overs(match_bowling_df.loc[x.index])),
                TotalDeliveries=('ballNumber', 'count'),  # Total balls bowled
                RunsConceded=('runsConceded', 'sum'),  # Total Runs Given
                Wickets=('isWicket', lambda x: x.sum()) , # Count of non-null wickets
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

            bowler_stats = match_bowling_df.groupby(["bowlerPlayerName", "BowlingType", "bowlingTeamName"]).agg(
                Overs = ('ballNumber', lambda x: calculate_overs(match_bowling_df.loc[x.index])),
                TotalDeliveries=('ballNumber', 'count'),  # Total balls bowled
                RunsConceded=('runsConceded', 'sum'),  # Total Runs Given
                Wickets=('isWicket', lambda x: x.sum()),  # Count of non-null wickets
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
            bowler_stats = bowler_stats.sort_values(by=["bowlingTeamName", "BowlingType"], ascending=False)

            st.dataframe(bowl_team_stats.style.highlight_max(color='green', axis=0, subset=['FalseShots', 'FalseShot%'])
                         .highlight_min(color='green', axis=0, subset=['FalseShotPerDismissial', 'BallsPerFalseShot', 'RunsPerFalseShot', 'S/R', 'Avg', 'Eco'])
                        .format({'Overs': '{:.1f}', 'Eco': '{:.2f}', 'S/R': '{:.2f}', 'Avg': '{:.2f}', 'FalseShotPerDismissial': '{:.2f}', 'BallsPerFalseShot': '{:.2f}', 'RunsPerFalseShot': '{:.2f}', 'RunsConceded': '{:.2f}', 'FalseShot%': '{:.2f}'}))
            st.dataframe(bowler_stats.style.highlight_max(color='green', axis=0, subset=['FalseShots', 'FalseShot%'])
                         .highlight_min(color='green', axis=0, subset=['FalseShotPerDismissial', 'BallsPerFalseShot', 'RunsPerFalseShot', 'S/R', 'Avg', 'Eco'])
                        .format({'Overs': '{:.1f}', 'Eco': '{:.2f}', 'S/R': '{:.2f}', 'Avg': '{:.2f}', 'FalseShotPerDismissial': '{:.2f}', 'BallsPerFalseShot': '{:.2f}', 'RunsPerFalseShot': '{:.2f}', 'RunsConceded': '{:.2f}', 'FalseShot%': '{:.2f}'}))
            false_shot_pitch_map(match_bowling_df)
        else:
            st.warning("Select a Team")

    else:
        st.warning("Match Data Not Found")