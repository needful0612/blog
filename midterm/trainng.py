import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pickle

df = pd.read_csv('data/Online_Dating_Behavior_Dataset.csv')

min_matches = df['Matches'].min()
max_matches = df['Matches'].max()
df['Matches_Percentage'] = ((df['Matches'] - min_matches) / (max_matches - min_matches)) * 99 + 1

df = df.drop('Matches', axis=1)

scaler = MinMaxScaler()
df[['Income', 'Age', 'Attractiveness']] = scaler.fit_transform(df[['Income', 'Age', 'Attractiveness']])

y = df['Matches_Percentage']
X = df.drop('Matches_Percentage', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

knn_model = KNeighborsRegressor(n_neighbors=3, weights='distance', metric='manhattan')
knn_model.fit(X_train, y_train)


with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

print("done")
