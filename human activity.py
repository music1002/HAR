import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,accuracy_score
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression

df_train=pd.read_csv("/Users/shreyashrivastava/Desktop/AI COURSE/PROJECTS/Human activity recognition/archive/train.csv")
df_test=pd.read_csv("/Users/shreyashrivastava/Desktop/AI COURSE/PROJECTS/Human activity recognition/archive/test.csv")
data=df_train.iloc[:,:-1].values
print(df_train.head())
print("*"*40)
cov=np.cov(data,rowvar=False)
df_train.groupby("Activity").groups
df_train.groupby("Activity").count()
pca=PCA(100)
cov_pca=pca.fit(df_train.iloc[:,:-1].values)
data_train=cov_pca.transform(df_train.iloc[:,:-1].values)
data_test=cov_pca.transform(df_test.iloc[:,:-1].values)
df_train_red=pd.DataFrame(data_train)
df_train_red["Activity"]=df_train["Activity"]
print(df_train_red.head())

df_train_red_STAND = df_train_red[df_train_red["Activity"]=="STANDING"]
df_train_red_SITTING = df_train_red[df_train_red["Activity"]=="SITTING"]
df_train_red_LAYING = df_train_red[df_train_red["Activity"]=="LAYING"]
df_train_red_WALKING = df_train_red[df_train_red["Activity"]=="WALKING"]
df_train_red_WALKING_DOWNSTAIRS = df_train_red[df_train_red["Activity"]=="WALKING_DOWNSTAIRS"]
df_train_red_WALKING_UPSTAIRS = df_train_red[df_train_red["Activity"]=="WALKING_UPSTAIRS"]

df_test.dropna(inplace=True)
df_test_red=pd.DataFrame(data_test)
df_test_red["Activity"]=df_test["Activity"]

labels_act=[]
for i in range(len(df_test_red)):
    if (df_test_red["Activity"].iloc[i]=="STANDING"):
        labels_act.append(0)
    if (df_test_red["Activity"].iloc[i]=="SITTING"):
        labels_act.append(1)
    if (df_test_red["Activity"].iloc[i]=="LAYING"):
        labels_act.append(2)
    if (df_test_red["Activity"].iloc[i]=="WALKING"):
        labels_act.append(3)
    if (df_test_red["Activity"].iloc[i]=="WALKING_UPSTAIRS"):
        labels_act.append(4)
    if (df_test_red["Activity"].iloc[i]=="WALKING_DOWNSTAIRS"):
        labels_act.append(5)
labels_act=np.array(labels_act)

def hmm_f1_acc(N,M,labels_act):
    hmm_stand=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type="diag")
    hmm_sit=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type="diag")
    hmm_lay=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type="diag")
    hmm_walk=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type="diag")
    hmm_walk_d=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type="diag")
    hmm_walk_u=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type="diag")
        
    hmm_stand.fit(df_train_red_STAND.iloc[:,0:100].values)
    hmm_sit.fit(df_train_red_SITTING.iloc[:,0:100].values)
    hmm_lay.fit(df_train_red_LAYING.iloc[:,0:100].values)
    hmm_walk.fit(df_train_red_WALKING.iloc[:,0:100].values)
    hmm_walk_u.fit(df_train_red_WALKING_UPSTAIRS.iloc[:,0:100].values)
    hmm_walk_d.fit(df_train_red_WALKING_DOWNSTAIRS.iloc[:,0:100].values)
    
    
    y_pred = []
    for i in range(len(df_test_red)):
        log_like=np.array([hmm_stand.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),
                           hmm_sit.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),hmm_lay.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),
                           hmm_walk_u.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),hmm_walk_d.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),
                           hmm_stand.score(df_test_red.iloc[i,0:100].values.reshape((1,100)))])
        y_pred.append(np.argmax(log_like))
    y_pred=np.array(y_pred)
    
    f_1= f1_score(labels_act,y_pred,average="micro")
    acc=accuracy_score(labels_act,y_pred)
    return f_1,acc
    
states= np.arange(1,36,1)
f1_val_state = []
acc_val_state= []
for i in states:
    print("HMM has been trained for {} states".format(i))
    f1,acc=hmm_f1_acc(i,1,labels_act)
    f1_val_state.append(f1)
    acc_val_state.append(acc)
fig,ax= plt.subplots(2,1)
ax[0].plot(f1_val_state)
ax[1].plot(acc_val_state)
plt.show()

clf = LogisticRegression(solver='lbfgs', max_iter=1000)
labels_tr=[]
for i in range(len(df_train_red)):
    if (df_train_red["Activity"].iloc[i]=="STANDING"):
        labels_tr.append(0)
    if (df_train_red["Activity"].iloc[i]=="SITTING"):
        labels_tr.append(1)
    if (df_train_red["Activity"].iloc[i]=="LAYING"):
        labels_tr.append(2)
    if (df_train_red["Activity"].iloc[i]=="WALKING"):
        labels_tr.append(3)
    if (df_train_red["Activity"].iloc[i]=="WALKING_UPSTAIRS"):
        labels_tr.append(4)
    if (df_train_red["Activity"].iloc[i]=="WALKING_DOWNSTAIRS"):
        labels_tr.append(5)
labels_tr=np.array(labels_tr)
labels_tr.shape

clf.fit(df_train_red.iloc[:,0:100].values,labels_tr)
predictions = clf.predict(df_test_red.iloc[:,0:100].values)
f1_2= f1_score(labels_act,predictions,average="micro")
acc2=accuracy_score(labels_act,predictions)

print("F1 SCORE using HMM:",np.round(f1,decimals=2))
print("Accuracy using HMM:",np.round(acc,decimals=2))

print("F1 SCORE using Logistic Regression:",np.round(f1_2,2))
print("Accuracy using Logistic Regression:",np.round(acc2,2))
