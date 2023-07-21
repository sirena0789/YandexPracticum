## Описание проекта

Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.

Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Нам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 

Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

## Инструменты
- pandas
- numpy
- matplotlib.pyplot
- sklearn.model_selection, train_test_split
- sklearn.metrics, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
- sklearn.preprocessing, StandardScaler
- sklearn.utils, shuffle
- sklearn.linear_model, LogisticRegression
- sklearn.ensemble, RandomForestClassifier
- sklearn.tree, DecisionTreeClassifier
- itertools, product
- tqdm


## Итоги исследования

Результат был достигнут. Модель прошла проверку на качество и достигла нужного показателя F1.
  
Итоговые показатели на тестовой выборке:  

Полнота 0.7174447174447175  
Точность 0.5186500888099467  
F1 мера 0.6020618556701031  
AUC ROC 0.855885160969907    
 
У данной модели хорошая полнота и средняя точность. Я бы не советовала использовать данную модель для работы с реальными клиентами.   
