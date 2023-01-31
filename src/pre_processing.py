import pandas as pd
import zipfile
import numpy as np
import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
  filename = "data/raw/shots_2021.zip"
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

  goal_df = shots_2021.pop('goal')
  shots_2021['goal'] = goal_df

  train_df, test_df  = train_test_split(shots_2021,test_size = 0.5, random_state = 123)

  output_location='data/pre_processed/'
  
  try:
      train_df.to_csv(output_location+'train_df.csv', index = False)
      test_df.to_csv(output_location+'test_df.csv', index = False)
  except:
      os.makedirs(os.path.dirname(output_location+'train_df.csv'),exist_ok=True)
      train_df.to_csv(output_location+'train_df.csv', index = False)
      test_df.to_csv(output_location+'test_df.csv', index = False)