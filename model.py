import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb


class WeatherPredictor:
    '''
    Прогнозирование температуры и влажности.

    Поддерживает два режима:

    1. predict_next_hour()
        temperature(T+1)
        humidity(T+1)

    2. predict_24_hours()
        temperature(T+1 ... T+24)
        humidity(T+1 ... T+24)
    '''

    def __init__(self,
                 lags            = (1, 2, 3, 6, 12, 24), # значения 1 час назад, 2 часа назад и т.д.
                 rolling_windows = (3, 6, 12, 24),       # признаки за последние 3 часа, 6 часов и т.д.
                 random_state    = 11
                ):

        self.lags = list(lags)
        self.rolling_windows = list(rolling_windows)
        self.random_state = random_state
        self.feature_columns = None
        self.model = None

        self.target_columns = ["temperature", "humidity"]


    def prepare_features(self, df):
        '''
        Создание признаков.
        '''

        df = df.copy().reset_index(drop=True)

        # Создаем лаги
        for col in self.target_columns:
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Скользящее окно
        for col in self.target_columns:
            for window in self.rolling_windows:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()

        return df.dropna().reset_index(drop = True)

    def train(self, df):
        '''
        Обучение модели.
        '''

        df = self.prepare_features(df)

        # Выделение таргетов
        y = df[self.target_columns]

        df = df.iloc[:-1].copy()
        y = y.iloc[:-1].copy()

        # Выделение признаков
        exclude = ["timestamp", "temperature", "humidity"]
        self.feature_columns = [c for c in df.columns if c not in exclude]
        X = df[self.feature_columns]

        # Разделение на тестовую и тренировочную выборки
        split = int(len(X) * 0.8)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        # XGBoost
        base_model = xgb.XGBRegressor(n_estimators  = 300,
                                      learning_rate = 0.1,
                                      max_depth     = 6,
                                      subsample     = 0.9,
                                      colsample_bytree = 0.9,
                                      random_state     = self.random_state,
                                      n_jobs           = -1
                                     )
        # Оболочка для мультитаргета
        self.model = MultiOutputRegressor(base_model)

        print("Модель обучается.")
        self.model.fit(X_train, y_train)

        # Оценка метриками
        prediction = self.model.predict(X_test)

        print('\nМетрики: ')
        for i, target in enumerate(self.target_columns):

            mae = mean_absolute_error(y_test.iloc[:, i], prediction[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], prediction[:, i]))
            r2 = r2_score(y_test.iloc[:, i], prediction[:, i])

            print(f"{target}: "
                  f"MAE  = {mae:.4f} "
                  f"RMSE = {rmse:.4f} "
                  f"R2   = {r2:.4f} "
                 )

        self.last_history = df.copy()

        return self

    def predict_next_hour(self, history):
        '''
        Прогноз на следующий час.
        '''

        if self.model is None:
            raise ValueError("Модель не обучена.")

        history = self.prepare_features(history)

        if len(history) == 0:
            raise ValueError("Недостаточно данных для построения лаговых признаков.")

        X = history[self.feature_columns].iloc[[-1]]

        prediction = self.model.predict(X)[0]

        return {"temperature": float(prediction[0]), "humidity": float(prediction[1])}

    def predict_24_hours(self, history):
        '''
        Прогноз следующих 24 часов (через рекурсию).
        '''

        if self.model is None:
            raise ValueError("Модель не обучена.")

        history = history.copy()

        if "timestamp" in history.columns:
            history["timestamp"] = pd.to_datetime(history["timestamp"])
            history = history.sort_values("timestamp").reset_index(drop=True)

        temp_prediction = []
        hum_prediction = []
        timestamps = []

        for i in range(24):
            prepared = self.prepare_features(history)
            X = prepared[self.feature_columns].iloc[[-1]]

            pred = self.model.predict(X)[0]

            temperature = float(pred[0])
            humidity = float(pred[1])

            temp_prediction.append(temperature)
            hum_prediction.append(humidity)

            # Создаем новую строку
            last = history.iloc[-1].copy()

            if "timestamp" in history.columns:
                new_timestamp = last["timestamp"] + pd.Timedelta(hours = 1)
                last["timestamp"] = new_timestamp
                timestamps.append(new_timestamp)

                last["hour"] = new_timestamp.hour
                last["dayofweek"] = new_timestamp.dayofweek
                last["month"] = new_timestamp.month
                last["dayofyear"] = new_timestamp.dayofyear

                if "is_weekend" in history.columns:
                    last["is_weekend"] = 1 if new_timestamp.dayofweek >= 5 else 0

            last["temperature"] = temperature
            last["humidity"] = humidity

            # Из минусов реализации: pressure,wind_speed,production_level,maintenance,power,tariff_enc остаются замороженными на уровне
            # последней строки в датафрейме.
            history = pd.concat([history, pd.DataFrame([last])], ignore_index = True)

        result = {"temperature": temp_prediction, "humidity": hum_prediction}

        if timestamps:
            result["timestamp"] = timestamps

        return result

    def evaluate(self, df):
        '''
        Оценка качества модели.
        '''

        if self.model is None:
            raise ValueError("Модель не обучена.")

        df = self.prepare_features(df)
        y = df[self.target_columns]

        df = df.iloc[:-1].copy()
        y = y.iloc[:-1].copy()

        X = df[self.feature_columns]

        # Разделение на тестовую и тренировочную выборки
        split = int(len(X) * 0.8)
        
        X_test = X.iloc[split:]
        y_test = y.iloc[split:]

        prediction = self.model.predict(X_test)
        
        metrics = {}
        for i, target in enumerate(self.target_columns):

            mae  = mean_absolute_error(y_test.iloc[:, i], prediction[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], prediction[:, i]))
            r2   = r2_score(y_test.iloc[:, i], prediction[:, i])

            metrics[target] = {"MAE": mae,
                               "RMSE": rmse,
                               "R2": r2
                              }

        return metrics

    def save_model(self, path = "weather_predictor.pkl"):
        '''
        Сохранение модели
        '''

        if self.model is None:
            raise ValueError("Модель не обучена.")

        joblib.dump({"model": self.model,
                     "feature_columns": self.feature_columns,
                     "lags": self.lags,
                     "rolling_windows": self.rolling_windows,
                     "target_columns": self.target_columns,
                    },
                    path
                   )

        print(f"Модель сохранена в {path}")

    def load_model(self, path="weather_predictor.pkl"):
        '''
        Загрузка модели
        '''

        data = joblib.load(path)

        self.model           =  data["model"]
        self.feature_columns = data["feature_columns"]
        self.lags            = data["lags"]
        self.rolling_windows = data["rolling_windows"]
        self.target_columns  = data["target_columns"]

        print(f"Модель загружена.")

if __name__ == "__main__":
    df = pd.read_csv('clear_dataset.csv', parse_dates = ['timestamp'])
    print("Данные загружены.")

    predictor = WeatherPredictor()
    predictor.load_model('weather_predictor.pkl')

    print('\nПредсказание на следующий час (2025-01-01 00:00:00)')
    history = df
    prediction = predictor.predict_next_hour(history)
    print('Температура   Влажность')
    print(f"{prediction["temperature"]:8.2f}", f"{prediction["humidity"]:12.2f}")

    print("\nПредсказания на следующие 24 часа:")
    print('        Время        Температура   Влажность')
    prediction = predictor.predict_24_hours(history)
    for t, temp, hum in zip(prediction["timestamp"],
                            prediction["temperature"],
                            prediction["humidity"]
                           ):
        print(t,f"{temp:9.2f}", f"{hum:12.2f}")
