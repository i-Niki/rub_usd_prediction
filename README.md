src\final.py

Контейнер постороен на базе tensorflow, результатом выполнения является предсказание 
значений валюттной пары rub/usd.
За основу была взята нейронная сеть LSTM(Keras)
Логирование производилось с помощью библиотеки loguru
Метриками качества были выбраны mse,rmse и mae
исторические данные берутся с """https://www.calc.ru/kotirovka-dollar-ssha""" за период в 9 месяцев 

Для запуска 

cmd:
docker build -t dollar_rub_predictor .

docker run —rm dollar_rub_predictor
