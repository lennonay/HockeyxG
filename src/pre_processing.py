import pandas as pd
import zipfile
import numpy as np
import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
  filename = "data/shots_2021.zip"
  shots_2021 = pd.read_csv(zipfile.ZipFile(filename).open("shots_2021.csv"))

  # list of columns to include
  lst_1 = ["shotType","shotAngle","arenaAdjustedShotDistance", "shooterLeftRight","xCordAdjusted",
          "yCordAdjusted","awayEmptyNet","awayTeamGoals",
        "defendingTeamDefencemenOnIce","defendingTeamForwardsOnIce", "distanceFromLastEvent", 
        "homeEmptyNet", "homeTeamGoals", "isPlayoffGame", "lastEventCategory",
        "shootingTeamDefencemenOnIce", "shootingTeamForwardsOnIce", "team","goal"]

  shots_2021 = shots_2021[lst_1]

  shots_2021['shootingTeamEmptyNet'] = np.where(shots_2021['team'] == 'HOME',shots_2021["homeEmptyNet"],shots_2021["awayEmptyNet"])
  shots_2021['shootingTeamGoals'] = np.where(shots_2021['team'] == 'HOME',shots_2021["homeTeamGoals"],shots_2021["awayTeamGoals"])
  shots_2021['defendingTeamEmptyNet'] = np.where(shots_2021['team'] == 'HOME',shots_2021["awayEmptyNet"],shots_2021["homeEmptyNet"])
  shots_2021['defendingTeamGoals'] = np.where(shots_2021['team'] == 'HOME',shots_2021["awayTeamGoals"],shots_2021["homeTeamGoals"])

  shots_2021 = shots_2021.drop(['homeEmptyNet','homeTeamGoals','awayEmptyNet','awayTeamGoals' ],axis = 1)

  X = shots_2021.drop('goal', axis = 1 )
  y = shots_2021['goal']

  X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.5, random_state = 123)

  output = 'data/processed/dummy.csv'
  os.makedirs(os.path.dirname(output),exist_ok=True)
  