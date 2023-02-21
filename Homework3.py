#%%
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
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
df = df.replace({'children' : {0 : '0 children', 1 : '1 children', 2 : '2 children', 3 : '3 children', 4 : '4 children', 5 : '5 children'}})
df = df.drop('id', axis = 1)
df = df.drop('age', axis = 1)
df = df.drop('income', axis = 1)
    #axis 1 is for columns, axis 0 is for rows
df = df.astype('category')
pd_df = df
transactions_from_df = [tuple(row) for row in pd_df.values.tolist()]
#test_df = [str(x) for x in transactions_from_df]
    #unneeded
# %%
#Transforming dataset into transcational matrix
te = TransactionEncoder()
te_ary = te.fit(transactions_from_df).transform(transactions_from_df)
te_df = pd.DataFrame(te_ary, columns=te.columns_)
# %%
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
    #Allows you to choose how many lines will print in pandas dataframe
frequent_itemsets = apriori(te_df, min_support = .2, use_colnames = True)
    #This finds the frequently occuring itemsets using Apriori
rules = association_rules(frequent_itemsets, metric= "confidence", min_threshold = 0.6)
pd_rules = pd.DataFrame(data = rules)
selected_rules = pd_rules[pd_rules['consequents'] == {'No_Pep'}]
print(selected_rules)
