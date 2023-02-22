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
from sklearn.cluster import KMeans
from sklearn import preprocessing
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#%matplotlib inline
print("Done")
#https://www.kaggle.com/code/tobyanderson/federalist-papers-clustering/notebook 

#%%
df = pd.read_csv("csv_files\HW4-data-fedPapers85.csv")
#df = df.drop('filename', axis = 1)
    #Droping filename, will not need it for clustering

#df = df.astype(str)
    #converts datatypes to string
scaler = MinMaxScaler()
#slace = scaler.fit_transform(df[['']])

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
# %%
#km=KMeans(n_clusters=4)
#y_predicted = km.fit_predict(df[['author','also']])
#y_predicted

scaler = MinMaxScaler()
c = df.drop(['author', 'filename'], axis = 1).columns
df[c] = scaler.fit_transform(df[c])

vals = df.drop(['author', 'filename'], axis = 1).values
doc_cluster = KMeans(n_clusters=4, n_init=10, random_state=0)
doc_cluster.fit(vals)
labs = doc_cluster.labels_
centroids = doc_cluster.cluster_centers_

# %%
df['labels'] = labs
df.head(2)

#find out how to put this on a scatterplot
# %%
#df['Clusters'] = km.labels_
#sns.scatterplot(x="author", y="also", hue = "Clusters", data = df, palette = "viridis")

# %%
test_df = df[df.author == 'dispt']
train_df = df[df.author != 'dispt']
X_test = test_df.drop(['author', 'filename'], axis = 1)
X_train = train_df.drop(['author', 'filename'], axis = 1)
y_test = test_df.author
y_train = train_df.author

paper_tree = DecisionTreeClassifier()
paper_tree.fit(X_train, y_train)


# %%
fig = plt.figure(figsize = (25, 20))
_ = tree.plot_tree(paper_tree,
                   feature_names = X_train.columns,
                   class_names = np.sort(y_train.unique()))
# %%
y_pred = paper_tree.predict(X_test)
list(y_pred)
# %%
