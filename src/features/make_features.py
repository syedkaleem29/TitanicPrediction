import pandas as pd

train = pd.read_csv('../../data/raw/train.csv', header=0, index_col=0)
test = pd.read_csv('../../data/raw/test.csv', header=0, index_col=0)
data = train.append(test)

print("Data structure:")
print("***************")
print(data.columns)
print(data.dtypes)
print("\nExample:")
print("**********")
print(data.head())
print("\nStatistics:")
print("*************")
print(data.describe())
print("Correlations:")
print(data.corr())
print("*************")

for i in data.columns:

    catdat = pd.Categorical(data[i])
    if(len(catdat.categories) > 9):
        continue
    print(i, " ", pd.Categorical(data[i]))

print("*************")

# clean up the mess
print("All empty and NA values:")
for i in data.columns:
    nas = sum(data[i].isnull())
    if nas>0:
        print(i," ", nas)

print("*************")


print("Data cleaning")

data.Age.fillna(value=data.Age.mean(), inplace=True)
data.Fare.fillna(value=data.Fare.mean(), inplace=True)
data.Embarked.fillna(value=data.Embarked.value_counts().idxmax(), inplace=True)
data.Survived.fillna(value=-1, inplace=True)

print("*************")
print(data.head())
print("Making features")
print("Extracting titles and adding column...")
titles = pd.DataFrame(data.apply(lambda x: x.Name.split(",")[1].split(".")[0], axis=1), columns=["Title"])
print(pd.Categorical(titles.Title))
data = data.join(titles)


print("Calculating family size and adding column...")
fsize = pd.DataFrame(data.apply(lambda x: x.SibSp + x.Parch, axis=1), columns=["FSize"])
data = data.join(fsize)

# drop useless columns
data.drop('Name', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)

# no need for the following as the sum is used
data.drop('Parch', axis=1, inplace=True)
data.drop('SibSp', axis=1, inplace=True)

print("Converting non number columns to numeric categories")
for col in data.select_dtypes(exclude=["number"]).columns:
    print("converting " + col + ".....")
    data[col] = data[col].astype('category')
    print(data[col].cat.categories)
    data[col] = data[col].cat.codes

train = data[data['Survived']!=-1]
train.to_csv('../../data/processed/train-clean.csv')

test = data[data['Survived']==-1]
data.drop('Survived', axis=1, inplace=True)
test.to_csv('../../data/processed/test-clean.csv')










