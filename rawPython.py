import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

matches = pd.read_csv("Prem Match Predictor/matches.csv", index_col = 0) 
matches.head() 

matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes 
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+","", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek 
matches["target"] = (matches["result"] == "W").astype("int")
matches.head()

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = matches[matches["date"] < '2022-01-01']

test = matches[matches["date"] > '2022-01-01'] 

predictors = ["venue_code", "opp_code", "hour", "day_code"] 

rf.fit(train[predictors],train["target"])

pred = rf.predict(test[predictors])

acc = accuracy_score(test["target"], pred)
preScore = precision_score(test["target"], pred)

# acc
# preScore

groupedByTeam = matches.groupby("team") 

def currentForm(team, cols, new_cols) : 
    team = team.sort_values("date")
    form = team[cols].rolling(5, closed='left').mean() 
    team[new_cols] = form 
    team = team.dropna(subset=new_cols)
    return team 

cols = ["gf","ga","sh","sot","dist","fk","pk","pkatt"] 
new_cols = [f"{c}_rolling" for c in cols]

currentForm(team, cols, new_cols)

calculateForm = matches.groupby("team").apply(lambda x: currentForm(x, cols, new_cols))
calculateForm                                  

calculateForm = calculateForm.droplevel('team')

calculateForm 

calculateForm.index = range(calculateForm.shape[0])
calculateForm

def makePredictions(data,predictors) : 
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01'] 
    rf.fit(train[predictors],train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted = preds), index=test.index)
    precision = precision_score(test["target"],preds)
    return combined, precision 

combined, precision = makePredictions(calculateForm, predictors + new_cols)

# precision 

combined = combined.merge(calculateForm[["date","team","opponent","result"]], left_index=True, right_index=True)

# combined

class MissingDict(dict) : 
    __missing__ = lambda self, key:key 

map_values = {
    "Brighton and Hove Albion" : "Brighton",
    "Manchester United" : "Manchester Utd", 
    "Newcastle United" : "Newcastle Utd", 
    "Tottenham Hotspur" : "Tottenham", 
    "West Ham United" : "West Ham", 
    "Wolverhampton Wanderers" : "Wolves" 
} 

mapping = MissingDict(**map_values)

# mapping["Brighton and Hove Albion"]

combined["normalizedName"] = combined["team"].map(mapping)
# combined 

merged = combined.merge(combined, left_on = ["date", "normalizedName"], right_on = ["date", "opponent"])

# merged

merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()

# finalPrecisionScore = (27/40)

# finalPrecisionScore





