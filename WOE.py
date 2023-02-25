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


def woe_discrete(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_bad']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_bad'] = df['prop_bad'] * df['n_obs']
    df['n_good'] = (1 - df['prop_bad']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['diff_prop_good'] = (1 - df['prop_bad']).diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    # df['IV'] = df['IV'].replace([np.inf, -np.inf], np.nan).sum()
    return df


def woe_continuous(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_bad']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_bad'] = df['prop_bad'] * df['n_obs']
    df['n_good'] = (1 - df['prop_bad']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    # df = df.sort_values(['WoE'])
    # df = df.reset_index(drop = True)
    df['diff_prop_good'] = (1 - df['prop_bad']).diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    # df['IV'] = df['IV'].replace([np.inf, -np.inf], np.nan).sum()
    return df


def plot_by_woe(df_WoE, rotation_of_x_axis_labels=0, name='0'):
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    # Turns the values of the column with index 0 to strings, makes an array from these strings, and passes it to variable x.
    y = df_WoE['WoE']
    # Selects a column with label 'WoE' and passes it to variable y.
    plt.figure(figsize=(18, 6))
    # Sets the graph size to width 18 x height 6.
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    # Plots the datapoints with coordiantes variable x on the x-axis and variable y on the y-axis.
    # Sets the marker for each datapoint to a circle, the style line between the points to dashed, and the color to black.
    plt.xlabel(df_WoE.columns[0])
    # Names the x-axis with the name of the column with index 0.
    plt.ylabel('Weight of Evidence')
    # Names the y-axis 'Weight of Evidence'.
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    # Names the grapth 'Weight of Evidence by ' the name of the column with index 0.
    plt.xticks(rotation=rotation_of_x_axis_labels)
    # Rotates the labels of the x-axis a predefined number of degrees.
    plt.savefig('woe/' + name + '.jpg')


df_train = pd.read_csv('excel/df_train_washed.csv', index_col=0)
df_test = pd.read_csv('excel/df_test_washed.csv', index_col=0)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

df_train_inputs = df_train.loc[:, df_train.columns.values[1:]]
df_test_inputs = df_test.loc[:, df_train.columns.values[1:]]

df_train_target = df_train.loc[:, df_train.columns.values[0]].to_frame()
df_test_target = df_test.loc[:, df_test.columns.values[0]].to_frame()

# WOE
df_temp = woe_discrete(df_train_inputs, 'NumberOfTime30-59DaysPastDueNotWorse', df_train_target)
df_temp.to_excel(excel_writer='woe/NumberOfTime30-59DaysPastDueNotWorse.xlsx', sheet_name='NumberOfTime30'
                                                                                          '-59DaysPastDueNotWorse')
plot_by_woe(df_temp, 0, 'NumberOfTime30-59DaysPastDueNotWorse')


def woe_encoding(woe_df, df_train_inputs, df_train, feature_name):
    d = dict()
    for i in range(len(df_temp)):
        if np.isposinf(df_temp.iloc[i, 8]):
            d[df_temp.iloc[i, 0]] = 1
        elif np.isneginf(df_temp.iloc[i, 8]):
            d[df_temp.iloc[i, 0]] = -1
        else:
            d[df_temp.iloc[i, 0]] = df_temp.iloc[i, 8]
    for i in d.keys():
        df_train_inputs[feature_name] = np.where(
            df_train[feature_name].isin([i]), d[i],
            df_train_inputs[feature_name])
    return df_train_inputs


df_train_inputs = woe_encoding(df_temp, df_train_inputs, df_train, 'NumberOfTime30-59DaysPastDueNotWorse')
df_test_inputs = woe_encoding(df_temp, df_test_inputs, df_test, 'NumberOfTime30-59DaysPastDueNotWorse')

df_temp = woe_discrete(df_train_inputs, 'NumberOfTime60-89DaysPastDueNotWorse', df_train_target)
plot_by_woe(df_temp, 0, 'NumberOfTime60-89DaysPastDueNotWorse')
df_temp.to_excel(excel_writer='woe/NumberOfTime60-89DaysPastDueNotWorse.xlsx', sheet_name='NumberOfTime60'
                                                                                          '-89DaysPastDueNotWorse')

df_train_inputs = woe_encoding(df_temp, df_train_inputs, df_train, 'NumberOfTime60-89DaysPastDueNotWorse')
df_test_inputs = woe_encoding(df_temp, df_test_inputs, df_test, 'NumberOfTime60-89DaysPastDueNotWorse')

df_temp = woe_discrete(df_train_inputs, 'NumberOfTimes90DaysLate', df_train_target)
plot_by_woe(df_temp, 0, 'NumberOfTimes90DaysLate')
df_temp.to_excel(excel_writer='woe/NumberOfTimes90DaysLate.xlsx', sheet_name='NumberOfTimes90DaysLate')

df_train_inputs = woe_encoding(df_temp, df_train_inputs, df_train, 'NumberOfTimes90DaysLate')
df_test_inputs = woe_encoding(df_temp, df_test_inputs, df_test, 'NumberOfTimes90DaysLate')

df_temp = woe_discrete(df_train_inputs, 'NumberOfDependents', df_train_target)
plot_by_woe(df_temp, 0, 'NumberOfDependents')
df_train_inputs = woe_encoding(df_temp, df_train_inputs, df_train, 'NumberOfDependents')
df_test_inputs = woe_encoding(df_temp, df_test_inputs, df_test, 'NumberOfDependents')
df_temp.to_excel(excel_writer='woe/NumberOfDependents.xlsx', sheet_name='NumberOfDependents')

fig, axes = plt.subplots(2, 2, figsize=(18, 6))
sns.histplot(x=df_train[df_train['MonthlyIncome'] < 1000]['MonthlyIncome'], ax=axes[0, 0])
sns.histplot(x=df_train[(df_train['MonthlyIncome'] > 1000) &
                        (df_train['MonthlyIncome'] <= 10000)]['MonthlyIncome'], ax=axes[0, 1])
sns.histplot(x=df_train[(df_train['MonthlyIncome'] > 10000) &
                        (df_train['MonthlyIncome'] <= 20000)]['MonthlyIncome'], ax=axes[1, 0])
sns.histplot(x=df_train[(df_train['MonthlyIncome'] > 20000) &
                        (df_train['MonthlyIncome'] <= 50000)]['MonthlyIncome'], ax=axes[1, 1])

plt.savefig('woe/monthlyIncome_bin0')

fig, axes = plt.subplots(2, 2, figsize=(18, 6))
sns.histplot(x=df_train[(df_train['MonthlyIncome'] > 50000) &
                        (df_train['MonthlyIncome'] <= 100000)]['MonthlyIncome'], ax=axes[0, 0])
sns.histplot(x=df_train[(df_train['MonthlyIncome'] > 100000) &
                        (df_train['MonthlyIncome'] <= 200000)]['MonthlyIncome'], ax=axes[0, 1])
sns.histplot(x=df_train[(df_train['MonthlyIncome'] > 200000) &
                        (df_train['MonthlyIncome'] <= 500000)]['MonthlyIncome'], ax=axes[1, 0])
sns.histplot(x=df_train[df_train['MonthlyIncome'] > 500000]['MonthlyIncome'], ax=axes[1, 1])
plt.savefig('woe/monthlyIncome_bin1')

bins = pd.IntervalIndex.from_tuples([(0, 1000)])
bins3 = pd.IntervalIndex.from_tuples([(10000, 12000), (12000, 14000), (14000, 16000), (16000, 20000)])
bins4 = pd.IntervalIndex.from_tuples([(20000, 30000), (30000, 50000)])
box1 = pd.cut(df_train[df_train['MonthlyIncome'] <= 1000]['MonthlyIncome'], bins)
box2 = pd.qcut(df_train[(df_train['MonthlyIncome'] > 1000) &
                        (df_train['MonthlyIncome'] <= 10000)]['MonthlyIncome'], 4)
box3 = pd.cut(df_train[(df_train['MonthlyIncome'] > 10000) &
                       (df_train['MonthlyIncome'] <= 20000)]['MonthlyIncome'], bins3)
box4 = pd.cut(df_train[(df_train['MonthlyIncome'] > 20000) &
                       (df_train['MonthlyIncome'] <= 50000)]['MonthlyIncome'], bins4)

bins5 = pd.IntervalIndex.from_tuples(
    [(50000, 70000), (70000, 100000), (100000, 140000), (140000, 200000), (200000, 500000),
     (500000, 3500000)])
box5 = pd.cut(df_train[df_train['MonthlyIncome'] > 50000]['MonthlyIncome'], bins5)
df_train_inputs['MonthlyIncome_x'] = df_train_inputs['MonthlyIncome'].values

df_train_inputs.loc[box1.index.values, 'MonthlyIncome_x'] = box1.values
df_train_inputs.loc[box2.index.values, 'MonthlyIncome_x'] = box2.values
df_train_inputs.loc[box3.index.values, 'MonthlyIncome_x'] = box3.values
df_train_inputs.loc[box4.index.values, 'MonthlyIncome_x'] = box4.values
df_train_inputs.loc[box5.index.values, 'MonthlyIncome_x'] = box5.values

df_temp = woe_continuous(df_train_inputs, 'MonthlyIncome_x', df_train_target)


def woe_encoding_continues(woe_df, df_train_inputs, df_train, feature_name):
    d = dict()
    for i in range(len(df_temp)):
        if np.isposinf(df_temp.iloc[i, 8]):
            d[df_temp.iloc[i, 0]] = 1
        elif np.isneginf(df_temp.iloc[i, 8]):
            d[df_temp.iloc[i, 0]] = -1
        else:
            d[df_temp.iloc[i, 0]] = df_temp.iloc[i, 8]
    for i in d.keys():
        pos_min = df_train[feature_name] >= i.left
        pos_max = df_train[feature_name] < i.right
        pose = pos_min & pos_max
        df_train_inputs[feature_name] = np.where(
            pose == True, d[i],
            df_train_inputs[feature_name])
    return df_train_inputs


df_train_inputs = woe_encoding_continues(df_temp, df_train_inputs, df_train, 'MonthlyIncome')
df_test_inputs = woe_encoding_continues(df_temp, df_test_inputs, df_test, 'MonthlyIncome')

df_temp.to_excel(excel_writer='woe/MonthlyIncome.xlsx', sheet_name='MonthlyIncome')

# Train_Dataset Boxplot
fig, axes = plt.subplots(2, 2, figsize=(18, 6))
sns.histplot(x=df_train[df_train['DebtRatio'] < 1]['DebtRatio'],
             ax=axes[0, 0])
sns.histplot(x=df_train[(df_train['DebtRatio'] > 1) &
                        (df_train['DebtRatio'] <= 10)]['DebtRatio'],
             ax=axes[0, 1])
sns.histplot(x=df_train[(df_train['DebtRatio'] > 10) &
                        (df_train['DebtRatio'] <= 100)]['DebtRatio'],
             ax=axes[1, 0])
sns.histplot(x=df_train[(df_train['DebtRatio'] > 100) &
                        (df_train['DebtRatio'] <= 1000)]['DebtRatio'],
             ax=axes[1, 1])
plt.savefig('woe/DebtRatio_bin')

bins = pd.IntervalIndex.from_tuples([(1, 10), (10, 100), (100, 1000), (1000, int(df_train_inputs['DebtRatio'].max()))])
box1 = pd.qcut(df_train[df_train['DebtRatio'] <= 1]['DebtRatio'], 10)
box2 = pd.cut(df_train[df_train['DebtRatio'] > 1]['DebtRatio'], bins)

df_train_inputs['DebtRatio_x'] = df_train_inputs['DebtRatio'].values
df_train_inputs.loc[box1.index.values, 'DebtRatio_x'] = box1.values
df_train_inputs.loc[box2.index.values, 'DebtRatio_x'] = box2.values

df_temp = woe_continuous(df_train_inputs, 'DebtRatio_x', df_train_target)
df_temp.to_excel(excel_writer='woe/DebtRatio.xlsx', sheet_name='DebtRatio')

plot_by_woe(df_temp, 90, 'DebtRatio')

df_train_inputs = woe_encoding_continues(df_temp, df_train_inputs, df_train, 'DebtRatio')
df_test_inputs = woe_encoding_continues(df_temp, df_test_inputs, df_test, 'DebtRatio')

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.histplot(x=df_train[df_train['RevolvingUtilizationOfUnsecuredLines'] < 1]['RevolvingUtilizationOfUnsecuredLines'],
             ax=axes[0])
sns.histplot(x=df_train[(df_train['RevolvingUtilizationOfUnsecuredLines'] > 1) &
                        (df_train['RevolvingUtilizationOfUnsecuredLines'] <= 10)][
    'RevolvingUtilizationOfUnsecuredLines'],
             ax=axes[1])
plt.savefig('woe/RevolvingUtilizationOfUnsecuredLinesbin.jpg')

box1 = pd.cut(df_train[df_train['RevolvingUtilizationOfUnsecuredLines'] <= 1]['RevolvingUtilizationOfUnsecuredLines'],
              50)
box2 = pd.cut(df_train[df_train['RevolvingUtilizationOfUnsecuredLines'] > 1]['RevolvingUtilizationOfUnsecuredLines'],
              10)
df_train_inputs['RevolvingUtilizationOfUnsecuredLines_x'] = df_train_inputs[
    'RevolvingUtilizationOfUnsecuredLines'].values

df_train_inputs.loc[box1.index.values, 'RevolvingUtilizationOfUnsecuredLines_x'] = box1.values
df_train_inputs.loc[box2.index.values, 'RevolvingUtilizationOfUnsecuredLines_x'] = box2.values

df_temp = woe_continuous(df_train_inputs, 'RevolvingUtilizationOfUnsecuredLines_x', df_train_target)

df_temp.to_excel('woe/RevolvingUtilizationOfUnsecuredLines.xlsx', 'RevolvingUtilizationOfUnsecuredLines')
df_train_inputs = woe_encoding_continues(df_temp, df_train_inputs, df_train, 'RevolvingUtilizationOfUnsecuredLines')
df_test_inputs = woe_encoding_continues(df_temp, df_test_inputs, df_test, 'RevolvingUtilizationOfUnsecuredLines')
plot_by_woe(df_temp, 90, 'RevolvingUtilizationOfUnsecuredLines')

df_train_inputs['RevolvingUtilizationOfUnsecuredLines:<0.0004'] = np.where(
    (df_train_inputs['RevolvingUtilizationOfUnsecuredLines'] < 0.0004), 1, 0)

df_temp = woe_continuous(df_train_inputs, 'NumberRealEstateLoansOrLines', df_train_target)
df_train_inputs = woe_encoding(df_temp, df_train_inputs, df_train, 'NumberRealEstateLoansOrLines')
df_test_inputs = woe_encoding(df_temp, df_test_inputs, df_test, 'NumberRealEstateLoansOrLines')

df_temp.to_excel('woe/NumberRealEstateLoansOrLines.xlsx', 'NumberRealEstateLoansOrLines')
plot_by_woe(df_temp, 0, 'NumberRealEstateLoansOrLines')

# df_train_inputs['NumberRealEstateLoansOrLines:>7_REF'] = np.where(df_train_inputs['NumberRealEstateLoansOrLines'].isin(range(8, int(df_test_inputs['NumberRealEstateLoansOrLines'].max()))), 1, 0)



bins = np.linspace(df_train_inputs['age'].min(), df_train_inputs['age'].max() + 1, 30)
df_train_inputs['age_x'] = pd.cut(df_train_inputs['age'], bins=bins, include_lowest=True, precision=0)
df_temp = woe_continuous(df_train_inputs, 'age_x', df_train_target)
df_temp.to_excel('woe/age.xlsx', 'age')
plot_by_woe(df_temp, 90, 'age')

df_train_inputs = woe_encoding_continues(df_temp, df_train_inputs, df_train, 'age')
df_test_inputs = woe_encoding_continues(df_temp, df_test_inputs, df_test, 'age')

df_temp = woe_continuous(df_train_inputs, 'NumberOfOpenCreditLinesAndLoans', df_train_target)
plot_by_woe(df_temp, 0, 'NumberOfOpenCreditLinesAndLoans')

df_train_inputs = woe_encoding(df_temp, df_train_inputs, df_train, 'NumberOfOpenCreditLinesAndLoans')
df_test_inputs = woe_encoding(df_temp, df_test_inputs, df_test, 'NumberOfOpenCreditLinesAndLoans')

df_train_inputs.to_csv('excel/df_train_inputs.csv')
df_test_inputs.to_csv('excel/df_test_inputs.csv')
df_train_target.to_csv('excel/df_train_target.csv')
df_test_target.to_csv('excel/df_test_target.csv')