from re import T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

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
    def plot_turn_data(file_path):
        turn_numbers = []
        turn_times = []

        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header
            for row in reader:
                turn_numbers.append(int(row[0]))
                turn_times.append(float(row[1]))

        plt.figure(figsize=(10, 6))
        plt.plot(turn_numbers, turn_times, marker='o')
        plt.title("Turn Times per Turn")
        plt.xlabel("Turn Number")
        plt.ylabel("Turn Time (seconds)")
        plt.grid(True)
        plt.show()

    @staticmethod
    def visualize_score_heatmap(csv_file):
        """
        Static method to visualize the scores as a heatmap.
        """
        # Load the data
        tournament_data = pd.read_csv(csv_file)

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

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(combined_pivot, annot=True, cmap='coolwarm', fmt="g", mask=mask)
        plt.title('Head-to-Head Tournament Scores')
        # put oppoenent label on top
        ax.xaxis.tick_top()
        plt.xlabel('Opponent')
        plt.ylabel('Player')
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

if __name__ == "__main__":
    # csv_file = 'simulator_results.csv'  # Replace with the path to your CSV file
    # TournamentVisualizer.visualize_score_heatmap(csv_file)
    # TournamentVisualizer.visualize_total_wins(csv_file)
    # TournamentVisualizer.visualize_max_match_duration(csv_file)
    #TournamentVisualizer.plot_turn_data('game_not_auto_played_A_vsB_turn_data.csv')
    game_data = [f"turn_data/game_{i}_student_agent_vs_alpha_agent_turn_data.csv" for i in range(12)]
    TournamentVisualizer.plot_average_turn_time(game_data)
