from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import numpy as np
# 查找关联(后面清洗数据的时候也要经常用的，用来比较效果)
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('excel/df_train_washed.csv', index_col=0)
df_test = pd.read_csv('excel/df_test_washed.csv', index_col=0)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

x = df_train.iloc[:, 1:]
x['constant'] = 1
vif = [variance_inflation_factor(x.values, x.columns.get_loc(i)) for i in x.columns]
column = [i for i in x.columns]
VIF = pd.DataFrame({'feature': column, 'vif': vif})
VIF.to_excel('excel/vif.xlsx')

df_train_inputs = df_train.loc[:, df_train.columns.values[1:]]
df_test_inputs = df_test.loc[:, df_train.columns.values[1:]]
df_train_target = df_train.loc[:, df_train.columns.values[0]].to_frame()

X_train, X_valid, y_train, y_valid = train_test_split(np.array(df_train_inputs), np.array(df_train_target),
                                                      test_size=0.2, random_state=42,
                                                      stratify=np.array(df_train_target))
lr = LogisticRegression(max_iter=300, solver='liblinear')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_valid)
y_pred_proba = lr.predict_proba(X_valid)
y_pred_proba = y_pred_proba[:][:, 1]

# cm = metrics.confusion_matrix(y_valid, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues_r');
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# all_sample_title = 'Confusion Matrix'
# plt.title(all_sample_title, size=15)
# plt.savefig('Figs/CM.jpg')


def plot_roc(y_valid, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr, linestyle='--', color='k')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig('Figs/ROC.jpg')


print(classification_report(y_valid, y_pred))
plot_roc(y_valid, y_pred_proba)
print(roc_auc_score(y_valid, y_pred_proba))
