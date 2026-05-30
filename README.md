# ml_pipeline
### Описание задачи
Разработать ML-модель для раннего выявления риска сердечных заболеваний на основе неинвазивных клинических показателей, позволяющую снизить количество пропущенных случаев
### Схема пайплайна
AutoML/
│
├─data/
├─src/
│  ├─data_loader.py
│  ├─pipeline.py
│  ├─train.py
│  └─predict.py
├─tests/
│  └─test_train.py
├─requirments.txt
└─Dockerfile
### ETL
Extract: Загрузка данных из data/heart.csv - data_loader.py
Transform: Заполнение пропусков + преобразование категориальных признаков с помощью OneHotEncoder, а числовых — с помощью StandardScaler - pipeline.py
Load: Данные идут сразу в модель через пайплайн - pipeline.py
### Модель
Выбирается натлучшая среди RandomForestClassifier, LogisticRegression, SVC, GradientBoostingClassifier, с подбором гиперпараметров через GridSearchCV
### Метрики на тестовой выборке
              precision    recall  f1-score   support

           0       0.94      0.79      0.86        82
           1       0.85      0.96      0.90       102

    accuracy                           0.89       184
   macro avg       0.90      0.88      0.88       184
weighted avg       0.89      0.89      0.88       184
### Docker
FROM python:3.11-slim                               ─ Базовый образ
WORKDIR /app                                        ─ Рабочая дериктория
COPY requirements.txt /app/                         ─ Копируем файл с зависимостями в контейнер
RUN pip install -r requirements.txt --no-cache-dir  ─ Установка необходимых зависимостей
COPY src/ /app/src/                                 ┐
COPY tests/ /app/tests/                             ┤─ Копирование файлов проекта в контейнер
COPY data/ /app/data/                               ┘
EXPOSE 8000
CMD ["pytest", "tests/"]                            ─ Запуск тестов при запуске контейнера

docker build -t ml_pipeline .                       ─ Создание образа
docker run ml_pipeline                              ─ Запуск контейнера

Функции контейнеризации: изоляция, оптимизация ресурсов, воспроизводимость, безопасность
### CI/CD
git add .
git commit -m ""
git push