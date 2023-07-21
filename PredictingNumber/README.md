##  Описание проекта

Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Построим модель для такого предсказания.

Значение метрики *RMSE* на тестовой выборке должно быть не больше 48.

## Инструменты 

- pandas
- numpy
- matplotlib.pyplot
- sklearn.model_selection, train_test_split
- sklearn.metrics, mean_squared_error
- sklearn.model_selection, TimeSeriesSplit
- sklearn.model_selection, GridSearchCV
- sklearn.tree, DecisionTreeRegressor
- statsmodels.tsa.seasonal, seasonal_decompose
- statsmodels.tsa.statespace, sarimax
- statsmodels.tsa.stattools, adfuller
- catboost, Pool, CatBoostRegressor, cv
- lightgbm


## Итоги исследования

По итоговому рейтингу времени обучения/предсказания и метрики RMSE побеждает DecisionTreeRegressor, но я считаю нужным назначить лидером CatBoostRegressor, так как в данном случае качество предсказания куда важнее времени обучения. DecisionTreeRegressor в этом плане сильно проигрывает другим моделям.

Поэтому победителем нашего вечера назначается:  
- Модель: CatBoostRegressor;  
- Гиперпараметры: depth=6, learning_rate=0.1, random_state=12345, verbose=False;   
- RMSE на тестовой выборке: 47.03577;  
- Время обучения, сек: 1.26;  
- Время предсказания модели, мсек: 2.57.
