import pandas as pd
import numpy as np
# 查找关联(后面清洗数据的时候也要经常用的，用来比较效果)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('EconomicData/cs-training.csv', index_col=0)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

e = pd.DataFrame({'count':df_train.isnull().sum().values, 'ratio': df_train.isnull().mean() * 100})
e.to_excel(excel_writer='excel/train_missing_values.xlsx', sheet_name='train_missing_values')

e = pd.DataFrame({'count':df_test.isnull().sum().values, 'ratio': df_test.isnull().mean() * 100})
e.to_excel(excel_writer='excel/test_missing_values.xlsx', sheet_name='test_missing_values')

# MonthlyIncome为空时
e = df_train[df_train['MonthlyIncome'].isnull()][['NumberOfDependents', 'DebtRatio']].describe()
e.to_excel(excel_writer='excel/MonthlyIncomeIsNull.xlsx', sheet_name='MonthlyIncomeIsNull')

# NumberOfDependent为空
e = df_train[df_train['NumberOfDependents'].isnull()][['MonthlyIncome', 'DebtRatio']].describe()
e.to_excel(excel_writer='excel/NumberOfDependentIsNull.xlsx', sheet_name='NumberOfDependentIsNull')

df_train['MonthlyIncome'].replace(np.nan, 0, inplace=True)
df_test['MonthlyIncome'].replace(np.nan, 0, inplace=True)
df_train['NumberOfDependents'].replace(np.nan, 0, inplace=True)
df_test['NumberOfDependents'].replace(np.nan, 0, inplace=True)

#单因素分析
# RevolvingUtilizationOfUnsecuredLines
e = df_train['RevolvingUtilizationOfUnsecuredLines'].describe().to_frame()
e.to_excel(excel_writer='excel/RevolvingUtilizationOfUnsecuredLines.xlsx', sheet_name='RevolvingUtilizationOfUnsecuredLines')

fig, axes = plt.subplots(1, 2, figsize=(18,6))
sns.distplot(x = np.array(df_train['RevolvingUtilizationOfUnsecuredLines']),
             ax = axes[0])
axes[0].set_title('Histogram Plot of RevolvingUtilizationOfUnsecuredLines')
sns.boxplot(x = df_train['RevolvingUtilizationOfUnsecuredLines'], ax = axes[1])
axes[1].set_title('Box Plot of RevolvingUtilizationOfUnsecuredLines')
plt.savefig("Figs/RevolvingUtilizationOfUnsecuredLines0.jpg")

below_1 = df_train[df_train['RevolvingUtilizationOfUnsecuredLines'] < 1]['RevolvingUtilizationOfUnsecuredLines'].count()*100/len(df_train)
bet_1_10 = df_train[(df_train['RevolvingUtilizationOfUnsecuredLines'] > 1) &
        (df_train['RevolvingUtilizationOfUnsecuredLines'] < 10)]['RevolvingUtilizationOfUnsecuredLines'].count() * 100/len(df_train)
beyond_10 = df_train[df_train['RevolvingUtilizationOfUnsecuredLines'] > 10]['RevolvingUtilizationOfUnsecuredLines'].count()*100/len(df_train)
fig, axes = plt.subplots(1, 2, figsize=(18,6))
sns.boxplot(x = df_train[df_train['RevolvingUtilizationOfUnsecuredLines'] < 1]['RevolvingUtilizationOfUnsecuredLines'],
            ax = axes[0])
axes[0].set_title('{}% of Train_Dataset'.format(round(below_1, 0)))
sns.boxplot(x = df_train[(df_train['RevolvingUtilizationOfUnsecuredLines'] > 1) &
                        (df_train['RevolvingUtilizationOfUnsecuredLines'] < 10)]['RevolvingUtilizationOfUnsecuredLines'],
            ax = axes[1])
axes[1].set_title('{}% of Train_Dataset'.format(round(bet_1_10, 0)))
plt.savefig("Figs/RevolvingUtilizationOfUnsecuredLines1.jpg")

to_drop_train = df_train[df_train['RevolvingUtilizationOfUnsecuredLines'] > 10].index.values
to_drop_test = df_test[df_test['RevolvingUtilizationOfUnsecuredLines'] > 10].index.values
df_train.drop(to_drop_train, axis = 0, inplace = True)
df_test.drop(to_drop_test, axis = 0, inplace = True)

# DEBT RATIO
e = df_train['DebtRatio'].describe().to_frame()
e.to_excel(excel_writer='excel/DebtRatio.xlsx', sheet_name='DebtRatio')
fig, axes = plt.subplots(1, 2, figsize=(18,6))
sns.distplot(x = np.array(df_train['DebtRatio']),
             ax = axes[0])
axes[0].set_title('Histogram Plot of Debt Ratio')
sns.boxplot(x = df_train['DebtRatio'], ax = axes[1])
axes[1].set_title('Box Plot of Debt Ratio')
plt.savefig("Figs/DebtRatio.jpg")

e = pd.DataFrame({'below 1': df_train[df_train['DebtRatio'] <= 1]['DebtRatio'].count()*100/len(df_train),
             'between 1 - 10': df_train[(df_train['DebtRatio'] > 1) &
                                        (df_train['DebtRatio'] <=10)]['DebtRatio'].count()*100/len(df_train),
             'beyond 10': df_train[df_train['DebtRatio'] > 10]['DebtRatio'].count()*100/len(df_train)}, index = [1])
e.to_excel(excel_writer='excel/DebtRatioDist.xlsx', sheet_name='DebtRatioDist')
e = df_train[(df_train['DebtRatio'] > 1) & (df_train['DebtRatio'] <=10)]['DebtRatio'].describe().to_frame()
e.to_excel(excel_writer='excel/DebtRatio1_10Dist.xlsx', sheet_name='DebtRatio1_10Dist')

e = df_train[df_train['DebtRatio'] > 10]['DebtRatio'].describe().describe().to_frame()
e.to_excel(excel_writer='excel/DebtRatio_ge10Dist.xlsx', sheet_name='DebtRatio_ge10Dist')

fig, axes = plt.subplots(1, 2, figsize=(18,6))
sns.boxplot(x= df_train['age'], ax = axes[0])
axes[0].set_title('Train_Dataset')
sns.boxplot(x= df_test['age'], ax = axes[1])
axes[1].set_title('Test_Dataset')
plt.savefig("Figs/Age.jpg")

df_train['age'].replace(0, 21, inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(18,6))
sns.histplot(x = df_train['NumberOfOpenCreditLinesAndLoans'], binwidth=1, ax = axes[0])
sns.histplot(x = df_test['NumberOfOpenCreditLinesAndLoans'], binwidth=1, ax = axes[1])
plt.savefig("Figs/NumberOfOpenCreditLinesAndLoans.jpg")

fig, axes = plt.subplots(1, 2, figsize=(18,6))
sns.histplot(x = df_train['NumberRealEstateLoansOrLines'], binwidth=1, ax = axes[0])
sns.histplot(x = df_test['NumberRealEstateLoansOrLines'], binwidth=1, ax = axes[1])
plt.savefig("Figs/NumberRealEstateLoansOrLines.jpg")

fig, axes = plt.subplots(1, 2, figsize=(18,6))
sns.histplot(x = df_train['NumberOfDependents'], binwidth=1, ax = axes[0])
sns.histplot(x = df_test['NumberOfDependents'], binwidth=1, ax = axes[1])
plt.savefig("Figs/NumberOfDependents.jpg")

#PAST DUE
due_30_59 = pd.DataFrame(df_train['NumberOfTime30-59DaysPastDueNotWorse'].value_counts()).rename(columns = {'NumberOfTime30-59DaysPastDueNotWorse':'30-59days'})
due_60_89 =  pd.DataFrame(df_train['NumberOfTime60-89DaysPastDueNotWorse'].value_counts()).rename(columns = {'NumberOfTime60-89DaysPastDueNotWorse':'60-89days'})
due_90 = pd.DataFrame(df_train['NumberOfTimes90DaysLate'].value_counts()).rename(columns = {'NumberOfTimes90DaysLate':'90days'})
e = pd.concat([due_30_59, due_60_89, due_90], axis = 1)
e.to_excel(excel_writer='excel/NumberOfDaysPastDueNotWorse.xlsx', sheet_name='NumberOfDaysPastDueNotWorse')

e = df_train[df_train['NumberOfTime30-59DaysPastDueNotWorse'] > 17][['NumberOfTime30-59DaysPastDueNotWorse',
                                                                'NumberOfTime60-89DaysPastDueNotWorse',
                                                                'NumberOfTimes90DaysLate']]
e.to_excel(excel_writer='excel/NumberOfDaysGe17PastDueNotWorse.xlsx', sheet_name='NumberOfDaysGe17PastDueNotWorse')

df_train.to_csv('excel/df_train_washed.csv')
df_test.to_csv('excel/df_test_washed.csv')

