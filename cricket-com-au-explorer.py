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
match_list_df = match_list_df[(match_list_df['isCompleted'] | match_list_df['isLive'])]

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
    match_list_df['gameType'].unique(),
    index=None,
    placeholder="Select format type",
    key='format_type'
)

if(format_type):
    format_df = match_list_df[match_list_df['gameType'] == format_type]
    unique_competitions = format_df[['competition_name', 'year', 'competition_id',]].drop_duplicates()
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
    line_rename_map = {
        'Wide': 'W', 'OutsideOff': 'OO', 'OffStump': 'Off',
        'MiddleStump': 'Mid', 'LegStump': 'Leg', 'DownLeg': 'DL', 'WideDownLeg': 'WDL'
    }

    length_order = ['FullToss', 'Yorker', 'HalfVolley', 'LengthBall', 'BackOfALength', 'Short']
    line_order = ['Wide', 'OutsideOff', 'OffStump', 'MiddleStump', 'LegStump', 'DownLeg', 'WideDownLeg']

    unique_bowlers = df['bowlerPlayerName'].dropna().unique()
    n_bowlers = len(unique_bowlers)
    n_cols = 3  
    n_rows = -(-n_bowlers // n_cols)  # Equivalent to math.ceil(n_bowlers / n_cols)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    heatmap_data_list = []

    # **Single Pass to Compute Heatmaps & Find Global Min/Max**
    for bowler in unique_bowlers:
        bowler_df = df[df['bowlerPlayerName'] == bowler]

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
        ax = axes[i]
        sns.heatmap(heatmap_data, fmt="d", cmap='coolwarm', vmin=global_min, vmax=global_max, ax=ax, annot=False)
        ax.set_title(f'{bowler}')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.tick_top()

    # **Hide Extra Subplots (If Any)**
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
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
            max_overs = match_df[match_df['bowlingTeamName'].isin(selected_bowl_teams)]['over_overNumber'].max()
            over_range = st.slider("Overs", 0, max_overs, (0, max_overs))
            match_bowling_df = match_bowling_df[match_bowling_df['over_overNumber'].between(over_range[0], over_range[1])]
        # else:
            # match_bowling_df = pd.DataFrame()

        if(len(match_bowling_df) > 0):

            false_shots_mask = match_bowling_df['battingConnectionId'].isin(['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge'])
            
            false_shot_list = ['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge']
            
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
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge']
                
                new_df = df[(df['battingFeetId'] == foottype) & (df['validDelivery'] == True)]

                if(len(new_df) == 0):
                    return np.nan
                total_balls = new_df['ballNumber'].count()
                false_shots_ff = new_df['battingConnectionId'].isin(false_shot_list).sum()
                return 100 - ((false_shots_ff / total_balls)*100)

            def calculate_bowl_type_ctrl(df, bowltype):

                false_shot_list = ['Missed', 'MisTimed', 'HitPad', 'PlayAndMissLegSide', 'LeadingEdge', 'ThickEdge', 
                                        'BottomEdge', 'OutsideEdge', 'TopEdge', 'Spliced', 'InsideEdge']

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
                Dot = ('runsScored', lambda x: (x[x == 0].count()/x.notna().count())*100),
                Boundary = ('runsScored', lambda x: (x[x >= 4].count()/x.notna().count())*100),
                ProductiveShot = ('ballNumber', lambda x: calculate_best_shot(match_bowling_df_copy.loc[x.index]))
            ).reset_index()
            batter_stats['Control%'] = 100 - ((batter_stats['FalseShots'] / batter_stats['TotalDeliveries']) * 100)
            batter_stats['S/R'] = np.where(batter_stats['TotalDeliveries'] == 0, 0, ((batter_stats['Runs'] / batter_stats['TotalDeliveries'])*100))
            # batter_stats = batter_stats.sort_values(by=["battingTeamName"], inplace=False, ascending=False)
            batter_stats = batter_stats.sort_values(by=["inningNumber", "battingTeamName", "Runs"], inplace=False, ascending=False)
            batter_stats_copy = batter_stats.copy()
            batter_stats_copy = batter_stats_copy.rename(columns=column_renames)


            st.subheader("Team Metrics")
            st.dataframe(bowl_team_stats_copy.style.highlight_max(color='green', axis=0, subset=['FS', 'FS%'])
                         .highlight_min(color='green', axis=0, subset=['FS/D', 'B/FS', 'R/FS', 'S/R', 'Avg', 'Eco'])
                        .format({'Overs': '{:.1f}', 'Eco': '{:.2f}', 'S/R': '{:.2f}', 'Avg': '{:.2f}', 'FS/D': '{:.2f}', 'B/FS': '{:.2f}', 'R/FS': '{:.2f}', 'Runs': '{:.2f}', 'FS%': '{:.2f}'}), 
                        hide_index=True)
            st.subheader("Bowler Metrics")
            st.dataframe(bowl_stats_copy.style.highlight_max(color='green', axis=0, subset=['FS', 'FS%'])
                         .highlight_min(color='green', axis=0, subset=['FS/D', 'B/FS', 'R/FS', 'S/R', 'Avg', 'Eco'])
                        .format({'Overs': '{:.1f}', 'Eco': '{:.2f}', 'S/R': '{:.2f}', 'Avg': '{:.2f}', 'FS/D': '{:.2f}', 'B/FS': '{:.2f}', 'R/FS': '{:.2f}', 'Runs': '{:.2f}', 'FS%': '{:.2f}'}),
                        hide_index=True)
            

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

            st.subheader("Bowler Pitch Map")
            pitch_map(match_bowling_df)
        else:
            st.warning("Select a Team")

    else:
        st.warning("Match Data Not Found")