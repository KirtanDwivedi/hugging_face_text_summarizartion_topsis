import pandas as pd
import numpy as np

def topsis(data, weights, impacts):

    normalized_data = data / np.sqrt(np.sum(np.square(data), axis=0))
    weighted_data = normalized_data * weights

    ideal_solution = np.where(np.array(impacts) == '+', weighted_data.max(), weighted_data.min())
    negative_ideal_solution = np.where(np.array(impacts) == '+', weighted_data.min(), weighted_data.max())

    distance_positive = np.sqrt(np.sum(np.square(weighted_data - ideal_solution), axis=1))
    distance_negative = np.sqrt(np.sum(np.square(weighted_data - negative_ideal_solution), axis=1))
    topsis_scores = distance_negative / (distance_positive + distance_negative)

    return topsis_scores

file_names = ['politics.csv', 'technology.csv', 'health.csv', 'entertainment.csv']

weights = [1, 1, 1, 1, 1, 1]
impacts = ['+', '+', '+', '+', '-', '-']
best_alternatives = {}

for file_name in file_names:
    data = pd.read_csv(file_name, index_col='Alternative')
    topsis_scores = topsis(data, weights, impacts)
    rank = topsis_scores.rank(ascending=False)
    best_alternative = rank[rank == 1].index[0]
    best_alternatives[file_name] = best_alternative

most_frequent_best = max(set(best_alternatives.values()), key=list(best_alternatives.values()).count)

def get_model_name(model_code):
  model_map = {
      'M1': 'BART',
      'M2': 'PEGASUS',
      'M3': 'T5',
      'M4': 'LongT5',
      'M5': 'LED'
  }
  return model_map.get(model_code, 'Unknown Model')

most_frequent_best_model = get_model_name(most_frequent_best)

print(f"The best model for cnn_dailymail dataset is : {most_frequent_best_model}")