from re import T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    def visualize_score_heatmap(csv_file):
        """
        Static method to visualize the scores as a heatmap.
        """
        # Load the data
        tournament_data = pd.read_csv(csv_file)

        # Creating a pivot table for the scores
        pivot_table_player_1 = tournament_data.pivot_table(index='player_1', columns='player_2', values='player_1_score', aggfunc=np.sum)
        pivot_table_player_2 = tournament_data.pivot_table(index='player_2', columns='player_1', values='player_2_score', aggfunc=np.sum)
        
        # Combine the two pivot tables
        combined_pivot = pivot_table_player_1.add(pivot_table_player_2, fill_value=0)

        # Replace NaN values with 0
        combined_pivot = combined_pivot.fillna(0)

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(combined_pivot, annot=True, cmap='coolwarm', fmt="g")
        plt.title('Head-to-Head Tournament Scores')
        plt.xlabel('Opponent')
        plt.ylabel('Player')
        plt.show()

    @staticmethod
    def visualize_total_scores(csv_file):
        """
        Static method to visualize the total scores for each player.
        """
        # Load the data
        tournament_data = pd.read_csv(csv_file)

        # Calculate total scores for each player
        player_1_scores = tournament_data.groupby('player_1')['player_1_score'].sum()
        player_2_scores = tournament_data.groupby('player_2')['player_2_score'].sum()
        total_scores = player_1_scores.add(player_2_scores, fill_value=0)

        # Plot the bar chart
        total_scores.sort_values().plot(kind='barh')
        plt.title('Total Scores for Each Player')
        plt.xlabel('Total Score')
        plt.ylabel('Player')
        plt.show()

    @staticmethod
    def visualize_average_match_duration(csv_file):
        """
        Static method to visualize the average match duration for each player.
        """
        # Load the data
        tournament_data = pd.read_csv(csv_file)

        # Convert time strings to lists of floats
        tournament_data['player_1_time'] = tournament_data['player_1_time'].apply(lambda x: eval(x))
        tournament_data['player_2_time'] = tournament_data['player_2_time'].apply(lambda x: eval(x))

        # Calculate average time for each player
        tournament_data['player_1_avg_time'] = tournament_data['player_1_time'].apply(np.mean)
        tournament_data['player_2_avg_time'] = tournament_data['player_2_time'].apply(np.mean)

        avg_time_player_1 = tournament_data.groupby('player_1')['player_1_avg_time'].mean()
        avg_time_player_2 = tournament_data.groupby('player_2')['player_2_avg_time'].mean()

        # Combine and average the times for players regardless of their position
        avg_times = avg_time_player_1.add(avg_time_player_2, fill_value=0) / 2

        # Plot the bar chart
        avg_times.sort_values().plot(kind='barh')
        plt.title('Average Match Duration for Each Player')
        plt.xlabel('Average Duration (seconds)')
        plt.ylabel('Player')
        plt.show()


# Usage example
csv_file = 'simulator_results.csv'  # Replace with the path to your CSV file
TournamentVisualizer.visualize_score_heatmap(csv_file)
TournamentVisualizer.visualize_total_scores(csv_file)
TournamentVisualizer.visualize_average_match_duration(csv_file)
