#Outline
    #This is how I plan to create this algorthim
#Input
    #csv File
#output
    #Info that tells a story about the data
        #Histogram 

# %%
    #This is how you create a cell
    #For importing all needed libraries
import plotnine
import pandas as pd
import csv
print("Loading Successful")
# %%
df = pd.read_csv(r'csv_files\data-storyteller (1).csv')
print(df)
# %%
#test: Creating a scatterplot graph
plot = plotnine(data=surveys_complete, mapping = plotnine.aes(x='Section'))



#Another Test