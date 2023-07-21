## Описание проекта

Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 

Заказчику важны:

- качество предсказания;
- скорость предсказания;
- время обучения.

## Итоги исследования

Итак, нашей итоговой моделью стал CatBoostRegressor с такими результатами:
- Гиперпараметры: learning_rate=0.2,random_state=1234, verbose=False;   
- RMSE на тестовой выборке: 1642.3484721466536;  
- Время обучения, сек: 2.94;  
- Время предсказания модели, мсек: 131.  