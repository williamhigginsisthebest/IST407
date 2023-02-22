#https://scikit-learn.org/stable/modules/clustering.html

#Info about the data
    #HM stands for Hamilton and Madison writting the paper together
        #For papers 18, 19, 20
        #Four authors, Hamilton, Madison, Jay, and Hamilton&Madison
    #There is a column for author and filename
        #Could prob drop file name
    #Rest of the columns are simple words and what % they make up the paper. 


#TODO:
###GOAL###
#find out who wrote the disputed essays, Hamilton vs Madison vs Jay
    #Done useing k-Means and HAC
#Need to provide doumention on anlyiss process
#Provide evidence for each method used to deterimne result
#use visulization
#For K-Means, anaylze the centroids to expain which attriubets are most usefufl for clustering
    #Hint: Centroid values on these dimesnions should be far apart from eachother to be able to distungish the clusters. 






#%%
#packages needed
import pandas as pd
import sklearn.cluster as slc
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import kmeans_plusplus
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


#%%
#Functions
Test_DF = pd.read_csv("csv_files\HW4-data-fedPapers85.csv")
def csv_int_to_string(csv_file):
    Conversion_df = csv_file.reset_index()
    for index, row in Conversion_df.iterrows():
        print(row[''])
    
print(csv_int_to_string(Test_DF))

#%%
#TODO:
    #Issue: Having issues with transcation_encoder, str vs int confliction
        #Need to create a function to convert all feilds in csv to a string
            #Make is so it works will all csv files!!!!

df = pd.read_csv("csv_files\HW4-data-fedPapers85.csv")
df = df.drop('filename', axis = 1)
    #Droping filename, will not need it for clustering

#df = df.astype(str)
    #converts datatypes to string

transactions_from_df = [tuple(row) for row in df.values.tolist()]
transactions_from_df
# %%
#Converting dataframe into a sparse matrix
#le = preprocessing.LabelEncoder()
#le.fit(['dispt', 'Hamilton', 'HM', 'Jay', 'Madison'])
#list(le.classes_)
#le.transform([])
def CustFun(train_data):
    str_to_num_dic = {}
    le = preprocessing.LabelEncoder()
    for column_name in train_data.columns:
        if train_data[column_name].dtype == object:
            train_data[column_name] = le.fit_transform(train_data[column_name])
            str_to_num_dic[column_name] = le.fit_transform(train_data[column_name])
        else:
            pass
    return train_data, str_to_num_dic

tDF, dic = CustFun(df)
tDF = tDF.astype(int)
input_df = tDF.to_numpy()

#input_df = pd.DataFrame(tDF)
#TODO:
#Make a dictionary which tells me which number represnts which author, for automation. 
# %%
centers, indices = kmeans_plusplus(input_df, n_clusters = 4, random_state=0)

# %%
#Matplotlib
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
for k, col in enumerate(colors):
    cluster_data = indices == k
    plt.scatter(tDF[cluster_data, 0], tDF[cluster_data, 1], c = col, marker = ".", s = 10)
plt.scatter(centers[:, 0], centers[:, 1], c="b", s=50)
plt.title("K-Means++ Initialization")
plt.xticks([])
plt.yticks([])
plt.show()
# %%
#Test Run to see is Matplotlib is working
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
n_samples = 4000
n_components = 4

X, y_true = make_blobs(
    n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0
)

X = X[:, ::-1]

# Calculate seeds from kmeans++
centers_init, indices = kmeans_plusplus(X, n_clusters=4, random_state=0)

# Plot init seeds along side sample data
plt.figure(1)
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]

for k, col in enumerate(colors):
    cluster_data = y_true == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)

plt.scatter(centers_init[:, 0], centers_init[:, 1], c="b", s=50)
plt.title("K-Means++ Initialization")
plt.xticks([])
plt.yticks([])
plt.show()
# %%
