import numpy as np
import math
import time
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
def load_gene_dataset(filename):
  data=''
  with open(filename,'r') as TP:
    data=TP.read()
    data=data.split()
    return data
def get_gene_lis(gene_chains):
  total_lis=np.array([g_u for g_u in ''.join(gene_chains)])
  return total_lis
def get_relative_pos(gene_chain):
  return np.array([math.sin(i) if i%2==0 else math.cos(i) for i in range(1,len(gene_chain)+1)])
def extra_combine_embedding(gene_lis,gene_chain):
  com_lis=[]
  for i in range(len(gene_lis)):
    if gene_lis[i] not in com_lis:
      com_lis.append(gene_lis[i])
    for j in range(len(gene_lis)):
      if gene_lis[i]+gene_lis[j] not in com_lis:
        com_lis.append(gene_lis[i]+gene_lis[j])
      for k in range(len(gene_lis)):
        if gene_lis[i]+gene_lis[j]+gene_lis[k] not in com_lis:
          com_lis.append(gene_lis[i]+gene_lis[j]+gene_lis[k])
        for z in range(len(gene_lis)):
          if gene_lis[i]+gene_lis[j]+gene_lis[k]+gene_lis[z] not in com_lis:
            com_lis.append(gene_lis[i]+gene_lis[j]+gene_lis[k]+gene_lis[z])
  com_lis=np.unique(np.array(com_lis))
  count_lis=[''.join(gene_chain).count(c)+10e-7 for c in com_lis]
  return count_lis+get_relative_pos(count_lis)
if __name__ == '__main__':
  start=time.time()
  x_train_pos=load_gene_dataset('train_positives.txt')
  y_train_pos=[1 for i in range(len(x_train_pos))]

  x_train_neg=load_gene_dataset('train_negatives.txt')[:round(1160*1.1)]
  y_train_neg=[0 for i in range(len(x_train_neg))]

  x_train=[]
  x_train.extend(x_train_pos)
  x_train.extend(x_train_neg)

  y_train=[]
  y_train.extend(y_train_pos)
  y_train.extend(y_train_neg)

  gene_lis=np.unique(get_gene_lis(x_train))

  train_data=[]
  max_it=1000
  lr=10e-6
  hidden_shape=(52,156,780,780,156,52)
  estimator_stack_MA=[MLPClassifier(hidden_layer_sizes=hidden_shape,max_iter=max_it,alpha=lr,solver='sgd'),AdaBoostClassifier()]
  cls=StackingClassifier(estimators=[(str(i),estimator_stack_MA[i])for i in range(len(estimator_stack_MA))])
  single_MLP=MLPClassifier(hidden_layer_sizes=hidden_shape,max_iter=max_it,alpha=lr,solver='sgd')
  knn=KNeighborsClassifier()
  ada=AdaBoostClassifier()
  for n in range(len(x_train)):
    index_list=extra_combine_embedding(gene_lis,x_train[n])
    train_data.append(np.array(index_list))

  print('positive num:',len(x_train_pos))
  print('negative num:',len(y_train_neg))
  print('original data shape:',np.array(x_train).shape)
  print('input shape:',np.array(train_data).shape)
  print('output shape:',np.array(y_train).shape)
  print('.....testing_start.....')
  cls.fit(np.reshape(np.array(train_data),(len(x_train),780)),y_train)
  single_MLP.fit(np.reshape(np.array(train_data),(len(x_train),780)),y_train)
  knn.fit(np.reshape(np.array(train_data),(len(x_train),780)),y_train)
  ada.fit(np.reshape(np.array(train_data),(len(x_train),780)),y_train)

  x_test_pos=load_gene_dataset('test_positives.txt')
  y_test_pos=[1 for i in range(len(x_test_pos))]
  print('positive num:',len(x_test_pos))
  x_test_neg=load_gene_dataset('test_negatives.txt')
  y_test_neg=[0 for i in range(len(x_test_neg))]
  print('negative num:',len(y_test_neg))
  x_test=[]
  x_test.extend(x_test_pos)
  x_test.extend(x_test_neg)

  y_test=[]
  y_test.extend(y_test_pos)
  y_test.extend(y_test_neg)
  test_data=[]
  for n in range(len(x_test)):
    index_list=extra_combine_embedding(gene_lis,x_test[n])
    test_data.append(np.array(index_list))
  print('input shape:',np.reshape(np.array(test_data),(len(x_test),780)).shape)
  print('sample:',np.reshape(np.array(test_data),(len(x_test),780))[0])
  print('output shape:',np.array(y_test).shape)
  result=cls.predict(np.reshape(np.array(test_data),(len(x_test),780)))
  score=cls.score(np.reshape(np.array(test_data),(len(x_test),780)),y_test)
  score_mlp=single_MLP.score(np.reshape(np.array(test_data),(len(x_test),780)),y_test)
  score_knn=knn.score(np.reshape(np.array(test_data),(len(x_test),780)),y_test)
  score_ada=ada.score(np.reshape(np.array(test_data),(len(x_test),780)),y_test)
  output_data=pd.DataFrame()
  output_data['class(1 pos, 0 neg)']=result
  output_data['true_value']=y_test
  output_data.to_csv('classified_result.csv',index=False)
  joblib.dump(cls,'Concise_DRL.pkl')
  print(' model        R2-score            Loss           ACC')
  print('score(Ada-boost)',score_ada,(1-sum(ada.predict(np.reshape(np.array(test_data),(len(x_test),780)))[:469])/len(result[:469]))*
        sum(ada.predict(np.reshape(np.array(test_data),(len(x_test),780)))[469:])/len(result[469:]),
        sum(np.array(ada.predict(np.reshape(np.array(test_data),(len(x_test),780))))==result)/len(result)
        )
  print('score(KNN)',score_knn,(1-sum(knn.predict(np.reshape(np.array(test_data),(len(x_test),780)))[:469])/len(result[:469]))*
        sum(knn.predict(np.reshape(np.array(test_data),(len(x_test),780)))[469:])/len(result[469:]),
        sum(np.array(knn.predict(np.reshape(np.array(test_data),(len(x_test),780))))==result)/len(result)
        )
  print('score(concise-DRL)',score,(1-sum(result[:469])/len(result[:469]))*sum(result[469:])/len(result[469:]),
        sum(np.array(cls.predict(np.reshape(np.array(test_data),(len(x_test),780))))==result)/len(result)
        )
  print('score(MLP)',score_mlp,(1-sum(single_MLP.predict(np.reshape(np.array(test_data),(len(x_test),780)))[:469])/len(result[:469]))*
        sum(single_MLP.predict(np.reshape(np.array(test_data),(len(x_test),780)))[469:])/len(result[469:]),
        sum(np.array(single_MLP.predict(np.reshape(np.array(test_data),(len(x_test),780))))==result)/len(result)
        )
  print('using time:',time.time()-start)