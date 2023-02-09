#Outline
    #This is how I plan to create this algorthim
#Input
    #csv File
#output
    #Info that tells a story about the data
        #Histogram 
            #Could aggregate all of t



#devide the data up tino each school

#Then devide the dat aup into each section for those schools


#Websites Used:
#https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_boxplot.html 
#https://www.w3schools.com/python/pandas/ref_df_sum.asp#:~:text=The%20sum()%20method%20adds,the%20sum%20of%20each%20row.
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html

# %%
    #This is how you create a cell
    #For importing all needed libraries

import pandas as pd
#import numpy as np
import csv
import plotnine as p9
print("Loading Successful")
# %%

df = pd.read_csv(r'csv_files\data-storyteller (1).csv')
print(df)
# %%
#test: Creating a histogram graph
#plot = plotnine(data=surveys_complete, mapping = plotnine.aes(x='Section'))

plot = ggplot(df) + geom_boxplot(aes(x='School', y = ' Completed'))
plot
#Another Test
# %%
#Bargraph
#gdf = df.groupby('School')
#aplot = ggplot(gdf) + geom_boxplot(aes(x='School', y = ' Completed'))
ggplot(df) + geom_bar(aes(x='School')) 

# %%

#group the schools and put the sections together. Find out how many poeple have completed course in each school. 
#Find out how many are very behind
#This sums the entire row, drops school and section rows to not mess with sum of students
Testdf = df
Sdf = df.drop(['School', 'Section'], axis = 1).sum(axis=1)
Testdf['Sdf'] = Sdf
Sdf

# %%
basePlot = (p9.ggplot(data=df,
           mapping=p9.aes(x='School', y=' Very Ahead +5')))

basePlot + p9.geom_point()


#find way to divde total count of row with number of sections that go into it. it is mean
# %%
