from itertools import groupby
from re import T
from click import group
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from scipy.stats import linregress, ttest_ind, f_oneway

class TournamentVisualizer:
    _instance = None

    @staticmethod
    def instance():
        """
        Static method to get the singleton instance.
        """
        if TournamentVisualizer._instance is None:
            TournamentVisualizer._instance = TournamentVisualizer()
        return TournamentVisualizer._instance

    @staticmethod
    def plot_average_turn_time(file_paths):
        data_frames = []

        for file_path in file_paths:
            df = pd.read_csv(file_path)
            df.columns = ['Turn Number','Turn Time']
            # Ensure that turn number is correctly labeled and is the same in all files
            df.rename(columns={df.columns[0]: 'Turn Number'}, inplace=True)
            data_frames.append(df)

        # Combine all data frames
        combined_df = pd.concat(data_frames)

        # Calculate the average turn time for each turn number
        avg_turn_times = combined_df.groupby('Turn Number').mean().reset_index()

        # Plotting
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=avg_turn_times, x='Turn Number', y='Turn Time', marker='o', linewidth=2.5, markersize=10)
        
        plt.title("Average Turn Time per Turn", fontsize=20)
        plt.xlabel("Turn Number", fontsize=16)
        plt.ylabel("Average Turn Time (seconds)", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.show()

    @staticmethod
    def analyze_depth_breadth_combinations(file_path):
        """
        Analyze different combinations of depth and breadth parameters to find the most effective one.
        """
        df = pd.read_csv(file_path)

        # Extract depth and breadth from the player names
        df['p1_depth'] = df['p1'].str.extract(r'Dpth_(\d+)').astype(int)
        df['p1_breadth'] = df['p1'].str.extract(r'Brth_(\d+)').astype(int)
        df['p2_depth'] = df['p2'].str.extract(r'Dpth_(\d+)').astype(int)
        df['p2_breadth'] = df['p2'].str.extract(r'Brth_(\d+)').astype(int)

        # Calculate average depth and breadth
        df['avg_depth'] = (df['p1_depth'] + df['p2_depth']) / 2
        df['avg_breadth'] = (df['p1_breadth'] + df['p2_breadth']) / 2

        # Calculate the average win percentage and other metrics for each depth-breadth combination
        performance_metrics = df.groupby(['avg_depth', 'avg_breadth']).agg({
            'p1_win_pct': 'mean',
            'p2_win_pct': 'mean',
            'p1_max_time': 'mean',
            'p2_max_time': 'mean',
            'p1_median_time': 'mean',
            'p2_median_time': 'mean'
        }).reset_index()

        # Visualize the data
        sns.pairplot(performance_metrics, kind="scatter", diag_kind="kde", markers="+",
                     plot_kws={'alpha': 0.5}, diag_kws={'shade': True})
        plt.show()

        return performance_metrics


    @staticmethod
    def plot_depth_vs_win_pct(file_path):
        """
        Determine whether win percentage is correlated with depth, including a regression line.
        """
        # Read the data
        df = pd.read_csv(file_path)
        

        # Extracting depth from player names
        df['depth'] = df['p1'].str.extract(r'Dpth_(\d+)').astype(float)

        # Calculating the average win percentage
        df['avg_win_pct'] = df['p1_win_pct']

        # Calculate the linear regression
        slope, intercept, r_value, p_value, std_err = linregress(df['depth'], df['avg_win_pct'])
        # Plotting with regression
        # plt.figure(figsize=(10, 6))
        # sns.regplot(x='depth', y='avg_win_pct', data=df, scatter_kws={'alpha':0.5})
        # plt.xlabel('Depth')
        # plt.ylabel('Average Win Percentage')
        # plt.title('Average Win Percentage vs. Depth with Regression Line')
        # plt.grid(True)
        # plt.show()
                # Plotting with regression
        plt.figure(figsize=(10, 6))
        sns.regplot(x='depth', y='avg_win_pct', data=df, scatter_kws={'alpha':0.5})
        plt.xlabel('Depth')
        plt.ylabel('Average Win Percentage')
        plt.title(f'Average Win Percentage vs. Depth with Regression Line\nR-Squared: {r_value**2:.2f}')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_breadth_vs_win_pct(file_path):
        """
        Determine whether win percentage is correlated with breadth, including a regression line.
        """
        # Read the data
        df = pd.read_csv(file_path)

        # Extracting breadth from player names
        df['breadth'] = df['p1'].str.extract(r'Brth_(\d+)').astype(float)

        # Calculating the average win percentage
        df['avg_win_pct'] = df['p1_win_pct']

        # Calculate the linear regression
        slope, intercept, r_value, p_value, std_err = linregress(df['breadth'], df['avg_win_pct'])
        # Plotting with regression
        # plt.figure(figsize=(10, 6))
        # sns.regplot(x='breadth', y='avg_win_pct', data=df, scatter_kws={'alpha':0.5})
        # plt.xlabel('Breadth')
        # plt.ylabel('Average Win Percentage')
        # plt.title('Average Win Percentage vs. Breadth with Regression Line')
        # plt.grid(True)
        # plt.show()
                # Plotting with regression
        plt.figure(figsize=(10, 6))
        sns.regplot(x='breadth', y='avg_win_pct', data=df, scatter_kws={'alpha':0.5})
        plt.xlabel('Breadth')
        plt.ylabel('Average Win Percentage')
        plt.title(f'Average Win Percentage vs. Breadth with Regression Line\nR-Squared: {r_value**2:.2f}')
        plt.grid(True)
        plt.show()

    @staticmethod
    def conduct_ttest(csv_file, breadth1, breadth2):
        '''
        Conduct a t-test between the win percentage of two players with different breadths.
        '''

        # Load the data
        df = pd.read_csv(csv_file)

        # Extracting depth and breadth from player names
        df['p1_breadth'] = df['p1'].str.extract(r'Brth_(\d+)').astype(int)
        df['p2_breadth'] = df['p2'].str.extract(r'Brth_(\d+)').astype(int)

        # Calculate win percentage for each player
        df['p1_win_pct'] = df['p1_wins'] / (df['p1_wins'] + df['p2_wins'])
        df['p2_win_pct'] = df['p2_wins'] / (df['p1_wins'] + df['p2_wins'])

        # Filter data for the two specific breadths
        group1 = df[(df['p1_breadth'] == breadth1) | (df['p2_breadth'] == breadth1)]
        group2 = df[(df['p1_breadth'] == breadth2) | (df['p2_breadth'] == breadth2)]

        # Choose the appropriate win percentage based on breadth
        win_pct1 = group1['p1_win_pct'] if group1['p1_breadth'].iloc[0] == breadth1 else group1['p2_win_pct']
        win_pct2 = group2['p1_win_pct'] if group2['p1_breadth'].iloc[0] == breadth2 else group2['p2_win_pct']

        # Conduct T-test
        t_stat, p_val = ttest_ind(win_pct1, win_pct2, nan_policy='omit')

        print(f"T-Test between breadths {breadth1} and {breadth2}: Statistic = {t_stat}, P-value = {p_val}")

    @staticmethod
    def visualize_score_heatmap(csv_file):
        """
        Static method to visualize the scores as a heatmap.
        """
        
        # Load the data
        tournament_data = pd.read_csv(csv_file)

        # tournament_data['p1 depth'] = tournament_data['p1'].str.extract(r'D_(\d+)').astype(int)
        # tournament_data['p1 breadth'] = tournament_data['p1'].str.extract(r'B_(\d+)').astype(int)
        # tournament_data['p2 depth'] = tournament_data['p2'].str.extract(r'D_(\d+)').astype(int)
        # tournament_data['p2 breadth'] = tournament_data['p2'].str.extract(r'B_(\d+)').astype(int)

        # # Rename each value in the p1 and p2 columns to be their 'Breadth_Depth' value
        # tournament_data['p1'] = tournament_data['p1 breadth'].astype(str) + '_' + tournament_data['p1 depth'].astype(str)
        # tournament_data['p2'] = tournament_data['p2 breadth'].astype(str) + '_' + tournament_data['p2 depth'].astype(str)


        # Creating a pivot table for the scores
        pivot_table_player_1 = tournament_data.pivot_table(index='p1', columns='p2', values='p1_wins')#, aggfunc=np.sum)
        pivot_table_player_2 = tournament_data.pivot_table(index='p2', columns='p1', values='p2_wins')#, aggfunc=np.sum)

        # print the pivot tables
        print(pivot_table_player_1)
        print(pivot_table_player_2)
        
        # Combine the two pivot tables
        combined_pivot = pivot_table_player_1.combine_first(pivot_table_player_2)

        print(combined_pivot)

        total_wins = combined_pivot.sum(axis=1).sort_values(ascending=False)
        combined_pivot= combined_pivot.reindex(total_wins.index)
        combined_pivot = combined_pivot[total_wins.index]

        print(combined_pivot)

        # make a mask so nans don't show up in the heatmap
        # 0 values still should show up
        mask = combined_pivot.isnull()

        # Rename the columns and rows to just display their

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(combined_pivot, annot=True, cmap='coolwarm', fmt="g", mask=mask)
        plt.title('Head-to-Head Tournament Scores')
        # put oppoenent label on top
        ax.xaxis.tick_top()
        plt.xlabel('Opponent')
        plt.xticks(rotation=45, ha='left')  # Rotate x-axis labels by 45 degrees
        plt.ylabel('Player')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_total_wins(csv_file):
        """
        Static method to visualize the total scores for each player.
        """
        # Load the data
        tournament_data = pd.read_csv(csv_file)

        # Calculate total scores for each player
        player_1_scores = tournament_data.groupby('p1')['p1_wins'].sum()
        player_2_scores = tournament_data.groupby('p2')['p2_wins'].sum()
        total_scores = player_1_scores.add(player_2_scores, fill_value=0)

        # Plot the bar chart
        total_scores.sort_values().plot(kind='barh')
        plt.title('Total wins for Each Player')
        plt.xlabel('Total Score')
        plt.ylabel('Player')
        plt.show()

    @staticmethod
    def visualize_max_match_duration(csv_file):
        """
        Static method to visualize the average match duration for each player.
        """
        # Load the data
        tournament_data = pd.read_csv(csv_file)

        # # Convert time strings to lists of floats
        # tournament_data['p1_maxtime'] = tournament_data['player_1_time'].apply(lambda x: eval(x))
        # tournament_data['player_2_time'] = tournament_data['player_2_time'].apply(lambda x: eval(x))

        # # Calculate average time for each player
        # tournament_data['player_1_avg_time'] = tournament_data['player_1_time'].apply(np.mean)
        # tournament_data['player_2_avg_time'] = tournament_data['player_2_time'].apply(np.mean)

        max_time_player_1 = tournament_data.groupby('p1')['p1_max_time'].max()
        max_time_player_2 = tournament_data.groupby('p2')['p2_max_time'].max()

        # Combine and max the times for players regardless of their position
        max_times = max_time_player_1.combine(max_time_player_2, max, fill_value=0)

        # Plot the bar chart
        max_times.sort_values().plot(kind='barh')
        plt.title('Max Turn Duration for Each Player')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Player')
        plt.show()

    @staticmethod
    def analyze_hyperparameter_impact(file_path):
        """
        Analyze the impact of board size, depth, and breadth on win percentage.
        """
        df = pd.read_csv(file_path)

        # Extract depth, breadth, and board size from player names
        df['depth'] = df['p1'].str.extract(r'Dpth_(\d+)').astype(int)
        df['breadth'] = df['p1'].str.extract(r'Brth_(\d+)').astype(int)

        # Calculate the win percentage for player 1
        df['win_pct'] = df['p1_wins'] / (df['p1_wins'] + df['p2_wins'])

        # Create a new DataFrame focusing on hyperparameters and their impact on win percentage
        hyperparam_df = df[['board_size', 'depth', 'breadth', 'win_pct']]

        # Visualize the data
        sns.pairplot(hyperparam_df, kind="scatter", diag_kind="kde", markers="+",
                     plot_kws={'alpha': 0.5}, diag_kws={'shade': True})
        plt.show()

        return hyperparam_df

    @staticmethod
    def unify_breadth_data(df):
        """
        Unify the breadth data from both p1 and p2.
        """
        df['p1_breadth'] = df['p1'].str.extract(r'Brth_(\d+)').astype(int)
        df['p2_breadth'] = df['p2'].str.extract(r'Brth_(\d+)').astype(int)

        p1_data = df[['p1', 'p1_breadth', 'p1_wins', 'p1_win_pct']].rename(columns={'p1': 'player', 'p1_breadth': 'breadth', 'p1_wins': 'wins', 'p1_win_pct': 'win_pct'})
        p2_data = df[['p2', 'p2_breadth', 'p2_wins', 'p2_win_pct']].rename(columns={'p2': 'player', 'p2_breadth': 'breadth', 'p2_wins': 'wins', 'p2_win_pct': 'win_pct'})

        combined_data = pd.concat([p1_data, p2_data])
        return combined_data

    @staticmethod
    def analyze_breadth_impact(csv_file):
        '''
        Conduct an ANOVA test to compare win percentages across different breadths.
        '''
        # Load the data
        df = pd.read_csv(csv_file)

        # Unify breadth data
        unified_data = TournamentVisualizer.unify_breadth_data(df)

        # Group data by breadth and collect win percentages
        breadth_groups = unified_data['breadth'].unique()
        win_pct_groups = [unified_data[unified_data['breadth'] == breadth]['win_pct'].dropna() for breadth in breadth_groups]

        # Conduct ANOVA test
        f_stat, p_val = f_oneway(*win_pct_groups)

        print(f"ANOVA test result: F-statistic = {f_stat}, P-value = {p_val}")

if __name__ == "__main__":
    csv_file = 'simulation_results.csv'  # Replace with the path to your CSV file
    TournamentVisualizer.visualize_score_heatmap(csv_file)
    TournamentVisualizer.visualize_total_wins(csv_file)
    TournamentVisualizer.visualize_max_match_duration(csv_file)
    #TournamentVisualizer.analyze_breadth_impact(csv_file)
