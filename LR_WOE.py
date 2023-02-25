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
import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc(y_valid, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr, linestyle='--', color='k')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig('Figs/LR_WOE_ROC_7F.jpg')


X_train = pd.read_csv('excel/df_train_inputs.csv', index_col=0)
X_test = pd.read_csv('excel/df_test_inputs.csv', index_col=0)
X_test.fillna(method='pad', axis=0)
y_train = pd.read_csv('excel/df_train_target.csv', index_col=0)
y_test = pd.read_csv('excel/df_test_target.csv', index_col=0)

original_features = ['RevolvingUtilizationOfUnsecuredLines', 'age',
                     'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                     'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                     'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                     'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

woe_train_inputs = X_train.copy()
woe_test_inputs = X_test.copy()

for col in X_train:
    if col not in original_features:
        X_train.drop(col, axis=1, inplace=True)

for col in X_test:
    if col not in original_features:
        X_test.drop(col, axis=1, inplace=True)

lr_woe = LogisticRegression(max_iter=300, solver='liblinear')
lr_woe.fit(X_train, y_train)

X_test.replace(np.nan, 0, inplace=True)
y_pred = lr_woe.predict(X_test)

y_pred_proba = lr_woe.predict_proba(X_test)
y_pred_proba = y_pred_proba[:, 1]

# cm = metrics.confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues_r');
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# all_sample_title = 'Confusion Matrix'
# plt.title(all_sample_title, size=15)
# plt.savefig('Figs/LR_WOE_CM_7F.jpg')

plot_roc(y_test, y_pred_proba)
print(roc_auc_score(y_test, y_pred_proba))

print(lr_woe.coef_[0])
print(lr_woe.intercept_)


def plot_CAP(y, y_hat):
    # 画CAP曲线，与计算AR值

    if type(y) == pd.core.frame.DataFrame:
        y = y.iloc[:, 0]

    total_count = len(y)
    pos_count = int(np.sum(y))

    a = auc([0, total_count], [0, pos_count])  # random model
    ap = auc([0, pos_count, total_count], [0, pos_count, pos_count]) - a

    model_y = [y_ for _, y_ in sorted(zip(y_hat, y), reverse=True)]
    y_values = np.append([0], np.cumsum(model_y))
    x_values = np.arange(0, total_count + 1)

    ar = auc(x_values, y_values) - a

    AR = ar / float(ap)

    lw = 2
    plt.figure(figsize=(8, 8))

    plt.plot([0, pos_count, total_count], [0, pos_count, pos_count], color='darkblue',
             lw=lw, label='Perfect Model')
    plt.plot(x_values, y_values, color='darkgreen',
             lw=lw, label='Actual Model')
    plt.plot([0, total_count], [0, pos_count], color='darkorange',
             lw=lw, label='Random Model', linestyle='--')

    plt.xlim([0.0, total_count])
    plt.ylim([0.0, pos_count + 1])
    plt.xlabel('Total Observations')
    plt.ylabel('Positive Observations')
    plt.title('Cumulative Accuracy Profile, AR = %.2f' % AR)
    plt.legend(loc="lower right")
    plt.savefig('Figs/CAP.jpg')


def AR_cal(y, y_hat):
    # 计算AR值

    if type(y) == pd.core.frame.DataFrame:
        y = y.iloc[:, 0]

    total_count = len(y)
    pos_count = int(np.sum(y))

    a = auc([0, total_count], [0, pos_count])
    ap = auc([0, pos_count, total_count], [0, pos_count, pos_count]) - a

    model_y = [y_ for _, y_ in sorted(zip(y_hat, y), reverse=True)]
    y_values = np.append([0], np.cumsum(model_y))
    x_values = np.arange(0, total_count + 1)

    ar = auc(x_values, y_values) - a

    AR = ar / float(ap)
    return AR


print(AR_cal(y_test, y_pred_proba))
plot_CAP(y_test, y_pred_proba)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def KS_cal(y, y_hat_proba):
    # 计算ks值

    fpr, tpr, _ = roc_curve(y, y_hat_proba)
    diff = np.subtract(tpr, fpr)
    ks = diff.max()

    return ks


def plot_ks(y, y_hat_proba):
    # 画ks曲线

    fpr, tpr, thresholds = roc_curve(y, y_hat_proba)
    diff = np.subtract(tpr, fpr)
    ks = diff.max()

    y_len = len(y)

    # 计算比例，这样计算比较快
    # 也可以自己划分样本的比例，自己计算fpr，tpr
    y_hat_proba_sort = sorted(y_hat_proba, reverse=True)
    cnt_list = []
    cnt = 0
    for t in thresholds:
        for p in y_hat_proba_sort[cnt:]:
            if p >= t:
                cnt += 1
            else:
                cnt_list.append(cnt)
                break
    percentage = [c / float(y_len) for c in cnt_list]

    if min(thresholds) <= min(y_hat_proba_sort):
        percentage.append(1)

    # 以下为画图部分
    best_thresholds = thresholds[np.argmax(diff)]
    best_percentage = percentage[np.argmax(diff)]
    best_fpr = fpr[np.argmax(diff)]

    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(percentage, tpr, color='darkorange',
             lw=lw, label='True Positive Rate')
    plt.plot(percentage, fpr, color='darkblue',
             lw=lw, label='False Positive Rate')
    plt.plot(percentage, diff, color='darkgreen',
             lw=lw, label='diff')
    plt.plot([best_percentage, best_percentage], [best_fpr, best_fpr + ks],
             color='navy', lw=lw, linestyle='--', label='ks = %.2f, thresholds = %.2f' % (ks, best_thresholds))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('percentage')
    plt.title('Kolmogorov-Smirnov')
    plt.legend(loc="lower right")
    plt.savefig('Figs/KS.jpg')


# plot_ks(y_test, y_pred_proba)
print(X_train.columns)


def getScore(X_train, model):
    p = model.predict_proba(X_train)
    odds = p[:, 1] / (p[:, 0])
    B = 50 / np.log(2)
    A = 500
    score = A - B * np.log(odds + 0.0001)
    return score


exception = getScore(X_train, lr_woe)
actual = getScore(X_test, lr_woe)


def plot_hist(exception, actual):
    plt.close()
    plt.hist([actual, exception], 10, weights=[np.zeros_like(actual) + 1 / len(actual), np.zeros_like(exception) + 1 / len(exception)], label=['actual', 'expected'])
    plt.xlabel('score')
    plt.ylabel('ratio')
    plt.legend(loc='upper left')
    plt.title('distribution')
    plt.savefig('Figs/distribution.jpg')


plot_hist(exception, actual)


def PSI_cal(score_actual, score_except, bins=10):
    actual_min = score_actual.min()
    actual_max = score_actual.max()

    binlen = (actual_max - actual_min) / bins
    cuts = [actual_min + i * binlen for i in range(1, bins)]
    cuts.insert(0, -float("inf"))
    cuts.append(float("inf"))

    actual_cuts = np.histogram(score_actual, bins=cuts)
    except_cuts = np.histogram(score_except, bins=cuts)

    actual_df = pd.DataFrame(actual_cuts[0], columns=['actual'])
    predict_df = pd.DataFrame(except_cuts[0], columns=['predict'])
    psi_df = pd.merge(actual_df, predict_df, right_index=True, left_index=True)

    psi_df['actual_rate'] = (psi_df['actual'] + 1) / psi_df['actual'].sum()  # 计算占比，分子加1，防止计算PSI时分子分母为0
    psi_df['predict_rate'] = (psi_df['predict'] + 1) / psi_df['predict'].sum()

    psi_df['psi'] = (psi_df['actual_rate'] - psi_df['predict_rate']) * np.log(
        psi_df['actual_rate'] / psi_df['predict_rate'])
    psi = psi_df['psi'].sum()
    return psi


print(PSI_cal(actual, exception))
