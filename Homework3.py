# %%
#https://www.datacamp.com/tutorial/association-rule-mining-python
    #It explains lift in a easy to understand way
#NOTE: Cannot install pycaret if your python version is not right. Use 3.8 or lower if you still want it.
    #Or you can do this. 
        #pip install -U --pre pycaret --user
    #Make sure you have at least 2015 visual studio dist package
#https://medium.com/@mervetorkan/association-rules-with-python-9158974e761a

#https://www.codeforests.com/2020/08/30/pandas-split-data-into-buckets/
#%%
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
import csv
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
print("Finished loading")


# %%
df = pd.read_csv("csv_files/bankdata_csv_all.csv")
        #Need to path to the folder containing the csv file.
df["age_group"] = pd.cut(df["age"], [0, 17, 25, 45, 64, 200], precision=0, labels=["youth", "young_adult", "adult", "middle_age", "older_adult"])
    #Configured age into five differnt bins
df["income_level"] = pd.cut(df['income'], [0, 32048, 53413, 106827, 373827, float("inf")], precision=0, labels=["low_income", "lower_middle_income", "middle_income", "upper_middle_income", "upper_income"])
    #Configured income level into five diffrent income levels
    #Based on this website's data
    #https://money.usnews.com/money/personal-finance/family-finance/articles/where-do-i-fall-in-the-american-economic-class-system

df = df.replace({'married': {'NO': 'Single', 'YES' : 'Married'}})
    #Cleaning up the Yes/No in the data. Will help with reading rules later. 
df = df.replace({'car': {'NO': 'No_Car', 'YES' : 'Has_Car'}})
df = df.replace({'save_act': {'NO': 'No_Save_Act', 'YES' : 'Has_Save_Act'}})
df = df.replace({'current_act': {'NO': 'No_Current_Act', 'YES' : 'Has_Current_Act'}})
df = df.replace({'mortgage': {'NO': 'No_Mortgage', 'YES' : 'Has_Mortgage'}})
df = df.replace({'pep': {'NO': 'No_Pep', 'YES' : 'Has_Pep'}})
df = df.drop('id', axis = 1)
df = df.drop('age', axis = 1)
df = df.drop('income', axis = 1)
    #axis 1 is for columns, axis 0 is for rows

df.astype({'children':'category'}).dtypes
df


transactions_from_df = [tuple(row) for row in pd_df.values.tolist()]

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(df)

# %%
#Transforming dataset into transcational matrix
#array = TransactionEncoder.fit(bankCSV).transform(bankCSV)
te = TransactionEncoder()
te_ary = te.fit(df).transform(df)
te_df = pd.DataFrame(te_ary, columns=te.columns_)
print(te_df)

# %%
frequent_itemsets = apriori(te_df, min_support = .01, use_colnames = True)
    #This finds the frequently occuring itemsets using Apriori
rules = association_rules(frequent_itemsets, metric= "confidence", min_threshold = 0.8)
print(rules)
# %%


#TODO:
    #Fix The rest of the cells
    #Find the intresting rules
    #Clean up the file
    #SAVE THE INTRESTING LINKS SOMEWHERE




