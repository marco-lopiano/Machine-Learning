import pandas as pd

sex_dict = {
    'male' : 1,
    'female' : 0
}

all_data = pd.read_csv('full_data.csv')
all_data['Sex'] = all_data['Sex'].apply(lambda x: sex_dict['male'] if x=='male' else sex_dict['female'])
all_data.drop('Name', axis=1, inplace=True)


all_data.to_csv('clean.csv', index=False)
