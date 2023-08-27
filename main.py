import pandas as pd 
import csv
from itertools import zip_longest
dataset = []
#dataset=pd.read_csv("Market_Basket_Optimisation.csv")

with open("Market_Basket_Optimisation.csv") as csvfile:
    rows = csv.reader(csvfile)
    res = list(zip_longest(*rows))
    res2 = [list(filter(None.__ne__, l)) for l in res]
    dataset = res2

from mlxtend.preprocessing import TransactionEncoder
te=TransactionEncoder()
te_ary=te.fit(dataset).transform(dataset)    #Apply one-hot-encoding on our dataset
df=pd.DataFrame(te_ary, columns=te.columns_)  #Creating a new DataFrame from our Numpy array
from mlxtend.frequent_patterns import apriori

frequent_itemsets=apriori(df, min_support=0.75, use_colnames=True) #Instead of column indices we can use column names.

from mlxtend.frequent_patterns import association_rules 
confidence = association_rules(frequent_itemsets,metric="confidence",min_threshold=0.7)
lift = association_rules(frequent_itemsets,metric="lift",min_threshold=1.25)


print(frequent_itemsets)
print(confidence)
print(lift)