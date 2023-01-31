import pandas as pd

filename = "data/shots_2021.zip"
shots_2021 = pd.read_csv(zipfile.ZipFile(filename).open("shots_2021.csv"))

pd.DataFrame(shots_2021['goal'].value_counts()).T