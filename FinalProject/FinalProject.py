#!/usr/bin/env python
# coding: utf-8

# # Описание проекта

# Оператор связи «Ниединогоразрыва.ком» хочет научиться прогнозировать отток клиентов. Если выяснится, что пользователь планирует уйти, ему будут предложены промокоды и специальные условия. Команда оператора собрала персональные данные о некоторых клиентах, информацию об их тарифах и договорах.

# ### Описание услуг

# Оператор предоставляет два основных типа услуг:   
# 
# 1. Стационарную телефонную связь. Возможно подключение телефонного аппарата к нескольким линиям одновременно.  
# 2. Интернет. Подключение может быть двух типов: через телефонную линию (DSL*,* от англ. *digital subscriber line*, «цифровая абонентская линия») или оптоволоконный кабель (*Fiber optic*).  
# 
# Также доступны такие услуги:  
# 
# - Интернет-безопасность: антивирус (*DeviceProtection*) и блокировка небезопасных сайтов (*OnlineSecurity*);  
# - Выделенная линия технической поддержки (*TechSupport*);  
# - Облачное хранилище файлов для резервного копирования данных (*OnlineBackup*);
# - Стриминговое телевидение (*StreamingTV*) и каталог фильмов (*StreamingMovies*).  
# 
# За услуги клиенты могут платить каждый месяц или заключить договор на 1–2 года.   Доступны различные способы расчёта и возможность получения электронного чека.

# ### Описание данных

# Данные состоят из файлов, полученных из разных источников:  
# 
# - `contract_new.csv` — информация о договоре;  
# - `personal_new.csv` — персональные данные клиента;  
# - `internet_new.csv` — информация об интернет-услугах;  
# - `phone_new.csv` — информация об услугах телефонии.
#  
# Во всех файлах столбец `customerID` содержит код клиента.
# 
# Информация о договорах актуальна на 1 февраля 2020.

# # План работы

# 1. Знакомство с данными. Необходимо выгрузить и проверить типы данных и наличие пропусков. 
# 2. Предобработка данных. На этом этапе нужно подготовить данные. В данном случае требуется обьединить датафреймы по ключу customerID, избавиться от пропусков в последствии, изменить типы данных и привести названия столбцов к нижнему регистру.
# 3. Исследовательский анализ данных. Необходимо визуализировать данные, исследовать корреляцию между признаками и избавиться от выбросов. 
# 4. Подготовить данные для обучения модели. Удалить,если есть, ненужные столбцы, разделить данные на выборки, закодировать категориальные признаки с помощью OneHotEncoder, для линейных моделей масштабировать числовые признаки. 
# 5. Подбор гиперпараметров и обучение 2-3 моделей. 
# 6. Тестирование наилучшей модели. 
# 7. Вывод. 

# # Цель работы

# Прогноз оттока клиентов для предложения им ряда дополнительных услуг или бонусов для оптимизации и регулировки количества пользователей данного оператора связи, следствием чего является увеличение/сохранение выручки компании.

# # Знакомство с данными

# Импорт библиотек
%pip install phik%pip install xgboost%pip install lightgbm 
# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.stats import f_oneway
from scipy.stats import spearmanr

import seaborn as sns

import phik
from phik.report import plot_correlation_matrix
from phik import report

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings

warnings.filterwarnings('ignore')


# Чтение файлов с датасетами

# In[2]:


try:
    contract = pd.read_csv('contract_new.csv')
    personal = pd.read_csv('personal_new.csv')
    internet = pd.read_csv('internet_new.csv')
    phone = pd.read_csv('phone_new.csv')
except:
    contract = pd.read_csv('/datasets/contract_new.csv')
    personal = pd.read_csv('/datasets/personal_new.csv')
    internet = pd.read_csv('/datasets/internet_new.csv')
    phone = pd.read_csv('/datasets/phone_new.csv')


# Напишем функцию для оптимизации просмотра данных

# In[3]:


def info(df):
    display(df.info())
    display(df.head(10))
    display(df.describe())


# Познакомимся с фреймом contract

# In[4]:


info(contract)


# Познакомимся с фреймом personal

# In[5]:


info(personal)


# Познакомимся с фреймом internet 

# In[6]:


info(internet)


# Познакомимся с фреймом phone

# In[7]:


info(phone)


# ### Вывод  
# 
# 1. Contract
#  - В столбце BeginDate необходимо изменить изменить тип данных на datetime64;  
#  - В столбце TotalCharges необходимо изменить изменить тип данных на float64;
#  - EndDate является целевым признаком. Тут необходимо заменить No на дату сбора данных (1 февраля 2020);
#  -  PaperlessBilling, Type, PaymentMethod являются категориальными признаками и нуждаются в кодировке.
# 2. Personal  
#  - Gender, Partner и Dependents являются категориальными признаками и нуждаются в кодировке.
# 3. Internet
#  - Всем столбцам кроме customerID требуется кодировка.
# 4. Phone
#  - MultipleLines требуется кодировка. 
# 5. Общий вывод по данным
#  - В данных нет пропусков, но во фреймах Internet и Phone есть данные не обо всех клиентах, что может привести к потере данных при обьединении фреймов. 

# # Предобработка данных

# Для начала добавим признак ухода клиента

# In[8]:


contract['tag'] = (contract['EndDate'] != 'No').astype(int)
contract.head(10)


# Заменим в EndDate значения "No" датой выгрузки датасета

# In[9]:


contract['EndDate'] = contract['EndDate'].replace(['No'], ['2020-02-01'])


# Заменим тип данных в BeginDate и EndDate

# In[10]:


contract['BeginDate'] = pd.to_datetime(contract['BeginDate'], format='%Y-%m-%d')
contract['EndDate'] = pd.to_datetime(contract['EndDate'], format='%Y-%m-%d')
contract.info()


# Заменим тип данных в TotalCharges

# In[11]:


contract['TotalCharges'] = pd.to_numeric(contract['TotalCharges'], errors ='coerce')
contract['TotalCharges'].isnull().sum()


# Взглянем, что в чем проблема с пропусками

# In[12]:


contract[contract['TotalCharges'].isnull()]


# Вероятно это новые пользователи. Заменим пропуски в этом стлолбце на 0

# In[13]:


contract['TotalCharges'] = contract['TotalCharges'].fillna(0)


# Обьеденим все датафреймы в один и проиндексируем фрейм по customerID

# In[14]:


df = contract     .merge(personal, how='left', on='customerID')     .merge(internet, how='left', on='customerID')     .merge(phone, how='left', on='customerID') 

df = df.set_index('customerID')


# In[15]:


df.head()


# Пропуски в пунктах 12-19 вероятнее всего означают отстутствие этого сервиса у клиента. Заменим этим пропуски "No"

# In[16]:


columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
for i in columns:
    df[i] = df[i].fillna('No')


# In[17]:


df.info()


# ### Вывод  
# 
# - Добавлен целевой признак ухода/неухода клиента
# - Заменили в EndDate значения "No" датой выгрузки датасета
# - Заменили тип данных в BeginDate, EndDate и TotalCharges и избавились от попусков
# - Обьеденили датасеты и устранили пропуски  
#  
# Теперь в данных порядок. Приводить названия столбцов к нижненму регистру не стала, так как в целом в этом нет необходимости

# # Исследовательский анализ данных

# Построим гистограммы и диаграммы размаха для числовых данных

# In[18]:


col = 2
row = 3
columns = ['MonthlyCharges', 'TotalCharges']

plt.figure(figsize=(20, 20))

j = 0

for i in columns:
    j += 1 
    plt.subplot(row, col, j)
    plt.hist([df[i].loc[df['tag'] == 1], df[i].loc[df['tag'] == 0]], label=['Ушедшие', 'Оставшиеся'], color=['lightblue', 'pink'])
    plt.title(i)
    plt.legend()
    
for i in columns:
    j += 1 

    plt.subplot(row, col, j)
    plt.boxplot(df[i])
    plt.title(i)

plt.show()


# Значения в MonthlyCharges распредедяются неравномерно. Есть пики на 20 и 80. Кажется такое распределение значений может говорить о наличии тарифов у оператора или каких то других "групп" клиентов. Гистограмма TotalCharges равномерно убывает. Аномальных выбросов я не увидела. 

# In[19]:


df[columns].describe()


# Попробуем провести статистический тест. Для определения зависимости категориальный признак - числовой признак воспользуемся ANOVA.

# Нулевая гипотеза(H0): Признаки не коррелируют между собой.  
# Гипотиза подтверждается только тогда, когда P-Value > 0,05

# In[20]:


for i in columns:
    print(i)
    CategoryGroupLists=df.groupby(i)['tag'].apply(list)
    AnovaResults = f_oneway(*CategoryGroupLists)
    print('P-Value for Anova is: ', AnovaResults[1])


# Гипотеза отвергается. Признаки коррелируют с нашим целевым признаком.

# Взглянем на корреляцию этих признаков между собой

# In[21]:


df[columns].corr()


# In[22]:


rho, p = spearmanr(df['MonthlyCharges'], df['TotalCharges'])
print('rho=',rho, 'p=', p)


# Признаки коррелируют, что логично. Чем больше месячный платеж, тем больше общая сумма покупок. Один из них нужно убрать, но пока не понятно какой именно.

# Далее посмотрим на категориальные данные

# In[23]:


cat_columns = ['Type','PaperlessBilling','PaymentMethod','gender','SeniorCitizen','Partner','Dependents',
                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'MultipleLines']
for j in cat_columns:
    sns.catplot(data=df, x='tag', hue=j, kind='count', palette=sns.color_palette("vlag"))


# - Оплата по месяцам в два раза популярнее, чем оплата сразу за один/два года, а вот у ушедших клиентов показатели примерно на одном уровне. 
# - У ушедших клиентов в MultipleLines соотношение подключивших и неподключивших эту услугу противоположно оставшимся клиентам. Та же ситуация у StreamingMovies, StreamingTV, Partner.
# - Так же отличаются распределения в колонках OnlineBackup, PaymentMethod (по одному пункту) и DeviceProtection.
# - Столбец gender никак не скажется на результатах, так как выборка по мужчинам и женщинам равномерная. Позже удалим его.

# Посмотрим корреляцию категориальных признаков

# In[24]:


phik_overview = df[cat_columns].phik_matrix()
plot_correlation_matrix(phik_overview.values, 
                        x_labels=phik_overview.columns, 
                        y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map='PuBu', 
                        title=r"correlation $\phi_K$", 
                        fontsize_factor=1.5, 
                        figsize=(20, 20))
plt.tight_layout()


# Далее взглянем на корреляюцию всех от всех

# In[25]:


phik_overview = df.phik_matrix()
plot_correlation_matrix(phik_overview.values, 
                        x_labels=phik_overview.columns, 
                        y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map='PuBu', 
                        title=r"correlation $\phi_K$", 
                        fontsize_factor=1.5, 
                        figsize=(20, 20))
plt.tight_layout()


# Корреляция между StreamingTV и StreamingMovies 0.74.  
# Тоже касается Partner и Dependents. Удалять мы их не будем пока. 
# Так же наблюдается сильная корреляция в столбцах с датами. Для обучения они нам не нужны, а испортить модель могут. Удалим оба столбца, предварительно создав столбец с общим количеством дней пользования.

# In[26]:


df['Days'] = (df['EndDate'] - df['BeginDate']).dt.days


# In[27]:


del df['BeginDate'], df['EndDate'], df['gender']


# Теперь вернемся к столбцам MonthlyCharges и TotalCharges. Посмотрим на корреляцию со столбцом Days.

# In[28]:


columns = ['MonthlyCharges', 'TotalCharges', 'Days']
df[columns].corr()


# TotalCharges проиграл эту битву. Возьмем на заметку, но удалять пока не будем.

# ### Вывод  
# 
# - Значения в MonthlyCharges распредедяются неравномерно. Есть пики на 20 и 80. Кажется такое распределение значений может говорить о наличии тарифов у оператора или каких то других "групп" клиентов. Гистограмма TotalCharges равномерно убывает. Аномальных выбросов я не увидела.
# - Обнаружили корреляцию у MonthlyCharges и TotalCharges.
# - Оплата по месяцам в два раза популярнее, чем оплата сразу за один/два года, а вот у ушедших клиентов показатели примерно на одном уровне. 
# - У ушедших клиентов в MultipleLines соотношение подключивших и неподключивших эту услугу противоположно оставшимся клиентам. Та же ситуация у StreamingMovies, StreamingTV, Partner.
# - Так же отличаются распределения в колонках OnlineBackup, PaymentMethod (по одному пункту) и DeviceProtection.
# - Столбец gender никак не скажется на результатах, так как выборка по мужчинам и женщинам равномерная.
# - Обнаружена корреляция между StreamingTV и StreamingMovies в 0.74, а также Partner и Dependents. 
# - Наблюдается также корреляция целевого признака и столбцов с датами. Удалили их, предварительно создав новый столбец со временем пользования даннным оперетором.

# # Подготовка данных для обучения

# Разделим датафрейм на обучающую и тестовую выбороки

# In[29]:


RANDOM_STATE = 50623
target = df['tag'] 
features = df.drop(columns=['tag'])
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=RANDOM_STATE)


# Закодируем категориальные данные с помощью One-Hot Encoding

# In[30]:


features_train_not_ohe = features_train
features_test_not_ohe = features_test


# In[31]:


cat_columns = features_train.select_dtypes(include='object').columns.to_list()
numeric = features_train.select_dtypes(exclude='object').columns.to_list()


# In[32]:


cat_columns


# In[33]:


numeric


# In[171]:


from sklearn.preprocessing import OneHotEncoder

encoder_ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
encoder_ohe.fit(features_train[cat_columns])
new_columns = encoder_ohe.get_feature_names(cat_columns)
features_train[new_columns] = encoder_ohe.transform(features_train[cat_columns])
features_train = features_train.drop(cat_columns, axis=1)
features_test[new_columns] = encoder_ohe.transform(features_test[cat_columns])
features_test = features_test.drop(cat_columns, axis=1)
features_train.head()
features_test.head()


# In[172]:


features_train.head()


# In[173]:


features_train.columns == features_test.columns


# Масштабируем количественные признаки

# In[174]:


columns = ['MonthlyCharges', 'Days', 'TotalCharges']

scaler = StandardScaler()

scaler.fit(features_train[columns])

features_train = features_train.copy()
features_train[columns] = scaler.transform(features_train[columns])

features_test = features_test.copy()
features_test[columns] = scaler.transform(features_test[columns])


# ### Вывод  
# - Разделили выборку на обучающую и тестовую с параметрами test_size=0.25 и random_state=50623
# - Закондировали категориальные признаки с помощью One-Hot Encoding
# - Масштабировали количественные признаки с помощью StandardScaler

# # Обучение моделей

# Модель LogisticRegression

# In[175]:


get_ipython().run_cell_magic('time', '', "model = LogisticRegression(random_state = RANDOM_STATE, solver='lbfgs', class_weight='balanced')\nparam = { 'C': list(range(1,15,3))}\ngrid = GridSearchCV(model, param_grid=param, scoring='roc_auc', cv=2, verbose=True, n_jobs=-1)\nmodel = grid.fit(features_train, target_train)\nmodel_LR = model.best_estimator_\nscore_LR = model.best_score_\nprint('Best parameters:', model.best_params_)\nprint('Best score:', score_LR)")


# Модель RandomForestClassifier

# In[176]:


get_ipython().run_cell_magic('time', '', "\nmodel = RandomForestClassifier(random_state = RANDOM_STATE, class_weight='balanced')\nparam = {'n_estimators': list(range(50,300,50)), 'max_depth':[5,15], 'max_features':['sqrt', 'log2']}\ngrid = GridSearchCV(model, param_grid=param, scoring='roc_auc', cv=3, verbose=True, n_jobs=-1)\nmodel = grid.fit(features_train, target_train)\nmodel_RFC = model.best_estimator_\nscore_RFC = model.best_score_\nprint('Best parameters:', model.best_params_)\nprint('Best score:', score_RFC)")


# Модель LinearSVC

# In[177]:


get_ipython().run_cell_magic('time', '', "\nmodel = LinearSVC(random_state=RANDOM_STATE, max_iter=10000, class_weight='balanced', dual = True)\nparam = {'C': list(range(1,15,3)), 'penalty' : ['l1', 'l2']}\ngrid = GridSearchCV(model, param_grid=param, scoring='roc_auc', cv=3, verbose=True, n_jobs=-1)\nmodel = grid.fit(features_train, target_train)\nmodel_LSVC = model.best_estimator_\nscore_LSVC = model.best_score_\nprint('Best parameters:', model.best_params_)\nprint('Best score:', score_LSVC)")


# Модель LGBMClassifier

# In[178]:


get_ipython().run_cell_magic('time', '', "model = LGBMClassifier(n_jobs=-1, random_state=RANDOM_STATE, class_weight='balanced')\nparam = {'n_estimators': [300, 500, 1000],\n                  'learning_rate': [0.01, 0.1, 0.2, 0.5],\n                  'max_depth': range(1, 9)}\ngrid = GridSearchCV(model, param_grid=param, scoring='roc_auc', cv=3, verbose=0,  n_jobs=-1)\nmodel = grid.fit(features_train, target_train)\nmodel_LGBM = model.best_estimator_\nscore_LGBM = model.best_score_\nprint('Best parameters:', model.best_params_)\nprint('Best score:', score_LGBM)")


# Модель XGBClassifier

# In[179]:


get_ipython().run_cell_magic('time', '', "\nratio = target_train[target_train == 1].count() / target_train[target_train == 0].count()\n\nmodel = XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=ratio)\nparam = {'n_estimators': [300, 500, 1000],\n                  'learning_rate': [0.01, 0.1, 0.2, 0.5],\n                  'max_depth': range(1, 9)}\ngrid = GridSearchCV(model, param_grid=param, scoring='roc_auc', cv=3, verbose=0,  n_jobs=-1)\nmodel = grid.fit(features_train, target_train)\nmodel_XGB = model.best_estimator_\nscore_XGB = model.best_score_\nprint('Best parameters:', model.best_params_)\nprint('Best score:', score_XGB)")


# Модель CatBoostClassifier

# In[180]:


get_ipython().run_cell_magic('time', '', "\ncat_features = ['Type', 'PaperlessBilling', 'PaymentMethod',\n                'SeniorCitizen', 'Partner', 'InternetService', 'Dependents',\n       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',\n       'StreamingMovies', 'MultipleLines', 'StreamingTV']\n\nmodel = CatBoostClassifier(random_state=RANDOM_STATE, auto_class_weights='Balanced', cat_features=cat_features)\nparam = {'max_depth':range(1, 9), 'learning_rate':[0.01, 0.1, 0.2, 0.5], 'iterations': [300, 500, 1000]}\ngrid = GridSearchCV(model, param_grid=param, scoring='roc_auc', cv=3, verbose=0,  n_jobs=-1)\nmodel = grid.fit(features_train_not_ohe, target_train)\nmodel_CBC = model.best_estimator_\nscore_CBC = model.best_score_\nprint('Best parameters:', model.best_params_)\nprint('Best score:', score_CBC)")


# Сделаем сводную таблицу с результатами

# In[181]:


index = ['LogisticRegression',
         'RandomForestClassifier',
         'LinearSVC', 'LGBMClassifier', 'XGBClassifier', 'CatBoostClassifier']
data = {'ROC-AUC':[score_LR, score_RFC, score_LSVC, score_LGBM, score_XGB, score_CBC]}

scores_data = pd.DataFrame(data=data, index=index)
scores_data


# ### Вывод 
#  Лидерами обучения являются модели со следующими показателями:
#  - LGBMClassifier, ROC-AUC = 0.898903
#  - XGBClassifier, ROC-AUC = 0.913498
#  - CatBoostClassifier, ROC-AUC = 0.920756  
#  
# Для тестирования возмем лучшую модель из представленных с гиперпараметрами:
# - CatBoostClassifier, ROC-AUC = 0.920756,  iterations=1000,  learning_rate=0.5,  max_depth=2.

# # Тестирование лучшей модели

# In[182]:


pred_proba_test = model_CBC.predict_proba(features_test_not_ohe)[:, 1]
pred_test = model_CBC.predict(features_test_not_ohe)

print('Accuracy: ', model_CBC.score(features_test_not_ohe, target_test))
print('AUC-ROC: ', roc_auc_score(target_test, pred_proba_test))
fpr, tpr, thresholds = roc_curve(target_test, pred_proba_test)

plt.figure(figsize=(11, 5))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')

plt.show()


# Посмотрим на распределение важности признаков

# In[183]:


im_df = pd.DataFrame(model_CBC.feature_importances_, index = features_test_not_ohe.columns, columns=['import'])
im_df = im_df.sort_values(by='import', ascending=False)
im_df


# In[184]:


im_df.plot(kind='bar', figsize=(10, 5), title='Важность признаков', color='pink', legend=False)
plt.show()


# ### Вывод 
# 
# Показатели на тестовой выборке:
# - Accuracy:  0.9114139693356048
# - AUC-ROC:  0.9083790293650387  
# 
# Самыми важными признаками являются:
# - Время пользования услугами оператора (61.7%), 
# - Сумма потрачененных средств на услуги (16.7%), 
# - Ежемесячные траты на услуги (9.6%)

# # Итоговый вывод
# 
# 1. В проекте проведена работа по предсказанию отказа клиента от услуг оператора сотовой связи для предоставления первому дополнительных бонусов и сохранения клиента.
# 2. Проведено знакомство с данными. Выгружены и проверены типы данных и наличие пропусков.
# 3. Далее проведена предобработка данных. Обьединили датафреймы по ключу customerID, избавились от пропусков в последствии и изменили типы данных.
# 4. Исследовательский анализ данных. Визуализировали категориальные и количественные признаки. Выявили корреляцию в различных признаках. Убрали столбцы с датами и сгененрировали новый признак с количеством дней использования услуг. На основе предварительного анализа можно сказать, что со временем вероятность отказа от услуг снижается, также чаще от услуг отказываются те, кто больше за них платит. 
# 5. Подготовили данные для обучения модели. Разделили данные на выборки, закодировать категориальные признаки с помощью OneHotEncoder, для линейных моделей масштабировали числовые признаки.
# 6. Обучение моделей. Был произведен подбор гиперпараметров и обучение следующих моделей: 
# | Model | ROC_AUC |
# | ------ | ------ |
# | LogisticRegression | 0.768698 |
# | RandomForestClassifier | 0.818619 |
# | LinearSVC | 0.771330 |
# | LGBMClassifier | 0.898903 |
# | XGBClassifier | 0.913498 |
# | CatBoostClassifier | 0.920756 |
# 7. Тестирование наилучшей модели. Для теста была выбрана модель: 
# - CatBoostClassifier, ROC-AUC = 0.920756,  iterations=1000,  learning_rate=0.5,  max_depth=2.   
# 
# Показатели на тестовой выборке:  
# - Accuracy:  0.9114139693356048
# - AUC-ROC:  0.9083790293650387   
# 
# Самыми важными признаками являются:  
# - Время пользования услугами оператора (61.7%), 
# - Сумма потрачененных средств на услуги (16.7%), 
# - Ежемесячные траты на услуги (9.6%)
# 8. Рекомендации для оператор связи «Ниединогоразрыва.ком»: 
# - Стоит обращать особое внимание на новых пользователей. Возможно стоит предлагать новичкам более выгодные тарифы и услуги. 
# - Также стоит уделить внимание группам пользователей с высокими показателями оплат услуг в месяц. Хорошим вариантом будет предоставлять скидку на оплату за несколько месяцев и год. Хорошим спосбом удержать клиентов - ввести бонусную программу по типу "чем больше платишь - тем больше скидка/бонусы".

# # Отчет

# В проекте были выполнены следующие пункты: 
# - Знакомство с данными. Выгружены и проверены типы данных и наличие пропусков.
# - Предобработка данных. Добавлен целевой признак ухода/неухода клиента. Заменили в EndDate значения "No" датой выгрузки датасета. Заменили тип данных в BeginDate, EndDate и TotalCharges и избавились от попусков. Обьеденили датасеты и устранили пропуски. 
# - Исследовательский анализ данных. Визуализировали категориальные и количественные признаки. Выявили корреляцию в MonthlyCharges и TotalCharges, StreamingTV и StreamingMovies, Partner и Dependents. Убрали столбцы с датами и сгененрировали новый признак с количеством дней использования услуг. Избавились от столбца gender, так как он не влияет на уход клиента, но может помешать обученнию модели. На основе предварительного анализа предположили, что со временем вероятность отказа от услуг снижается, также чаще от услуг отказываются те, кто больше за них платит.
# - Подготовили данные для обучения модели. Разделили данные на выборки с параметрами test_size=0.25, random_state=50623; закодировать категориальные признаки с помощью OneHotEncoder, для линейных моделей масштабировали числовые признаки.
# - Обучение моделей. Для обучения были выбраны следующие признаки: 
# 1. *Категориальные*  
#     'Type' - тип оплаты,
#     'PaperlessBilling' - безналичный расчет,
#     'PaymentMethod' – способ оплаты,,
#     'Partner' - наличие супруга(и),
#     'Dependents' - наличие иждивенцев,
#     'SeniorCitizen' - наличие пенсионного статуса по возрасту,
#     'InternetService' - интернет сервис,
#     'OnlineSecurity' - онлайн защита,
#     'OnlineBackup' - резервное копирование,
#     'DeviceProtection' - защита устройства,
#     'TechSupport' - тех поддержка,
#     'StreamingTV' - ТВ,
#     'StreamingMovies' - фильмы,
#     'MultipleLines' - наличие возможности ведения параллельных линий во время
#     звонка. 
# 2. *Количественные*  
#     'MonthlyCharges' – ежемесячные траты на услуги,
#     'TotalCharges' - общие траты,
#     'Days' - общее количество дней пользования услугами оператора (сгенерированный признак).
# 3. *Целевой*  
#     'tag' - факт ухода клиента (сгенерированный признак).  
# 
# 
# Был произведен подбор гиперпараметров и обучение следующих моделей:
# 
# | Model | ROC_AUC |
# | ------ | ------ |
# | LogisticRegression | 0.768698 |
# | RandomForestClassifier | 0.818619 |
# | LinearSVC | 0.771330 |
# | LGBMClassifier | 0.898903 |
# | XGBClassifier | 0.913498 |
# | CatBoostClassifier | 0.920756 |
# 
# - Тестирование наилучшей модели. Для теста была выбрана модель:  
#     CatBoostClassifier, ROC-AUC = 0.920756,  iterations=1000,  learning_rate=0.5,  max_depth=2.    
#  
#  
# - Показатели на тестовой выборке:   
# 1. Accuracy:  0.9114139693356048
# 2. AUC-ROC:  0.9083790293650387   
#   
#   
# - Самыми важными признаками являются:  
# 1. Время пользования услугами оператора (61.7%), 
# 2. Сумма потрачененных средств на услуги (16.7%), 
# 3. Ежемесячные траты на услуги (9.6%)
#   
#   
# - Рекомендации для оператор связи «Ниединогоразрыва.ком»: 
# 1. Стоит обращать особое внимание на новых пользователей. Возможно стоит предлагать новичкам более выгодные тарифы и услуги. 
# 2. Также стоит уделить внимание группам пользователей с высокими показателями оплат услуг в месяц. Хорошим вариантом будет предоставлять скидку на оплату за несколько месяцев и год. Хорошим спосбом удержать клиентов - ввести бонусную программу по типу "чем больше платишь - тем больше скидка/бонусы".
#   
#   
# - Отличия от исходного плана работ:
# 1. приводить названия столбцов к нижнему регистру не понадобилось;
# 2. выбросы не были обнаружены;
# 3. были добавлены столцы с целевым признаком и временем пользования услугами;
# 4. обучалось 6 моделей, вместо заявленных 2-3, так как на начальном этапе не удавалось добиться нужных показателей метрики и пришлось импровизировать с различными моделями;
# 5. после тестирования была добавлена визуализация важности признаков.  
# 
# Трудностей в работе не встречалось, все этапы выполнения проекта являются важными и нельзя опускать ни один из них. 
# 
