import pandas as pd

from Size.female import size_constant_female, size_dataloader_female
X_train = pd.read_csv("./dutch/X_train.csv", skipinitialspace=True, usecols=size_constant_female.demographic)

print(X_train)