import numpy as np
#import matplotlib.pyplot as plt
from math import sqrt
#from matplotlib import style
import warnings
from collections import Counter
import random
import pandas as pd
#style.use('fivethirtyeight')

#dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
#new_features = [5,7]

#for i in dataset:
#    for ii in dataset[i]:
#        plt.scatter(ii[0],ii[1])

#plt.show()

def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warning.warn("wrong input")
    distances = []
    for groups in data:
        for features in data[groups]:
            euclid_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclid_distance,groups])

    #print(distances)
    #print(sorted(distances))
    #print(sorted(distances)[:k])

    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes))
    vote_result = Counter(votes).most_common(1)[0][0]
    
    return vote_result

#result = k_nearest_neighbors(dataset,new_features,k=3)

#print(result)

df = pd.read_csv("breast-cancer-wisconsin.data.txt")
df.drop(['id'],1,inplace=True)
df.replace('?',1,inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

#print(train_set)

total = 0
favour = 0

for groups in test_set:
    for data in test_set[groups]:
        vote = k_nearest_neighbors(train_set,data,k=5)
        if vote==groups:
            favour += 1
        total+=1

accuracy = favour/total

print(accuracy)















