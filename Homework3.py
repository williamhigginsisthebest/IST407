# %%
#https://www.datacamp.com/tutorial/association-rule-mining-python
    #It explains lift in a easy to understand way
#NOTE: Cannot install pycaret if your python version is not right. Use 3.8 or lower if you still want it.
    #Or you can do this. 
        #pip install -U --pre pycaret --user
    #Make sure you have at least 2015 visual studio dist package
#https://medium.com/@mervetorkan/association-rules-with-python-9158974e761a
#%%
import pandas as pd
import numpy as np
import pycaret
# %%
from pycaret.datasets import get_data
data = get_data("france")

# %%
#https://analyticsindiamag.com/beginners-guide-to-understanding-apriori-algorithm-with-implementation-in-python/ 
from pycaret.utils import *
s = setup(data = data, transaction_id = 'InvoiceNo', item_id = "Description")
s

# %%
