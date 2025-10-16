# analysis_script.py
"""
Скрипт для анализа файлов fivesec_predictions.csv.
Вычисляет метрики (MSE, MAE, точность тренда) и строит графики для моделей RF, XGB, LGB.
"""

from logger import pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_csv_files(directory):
    """
    Загружает все CSV файлы из указанной директории и объединяет их в один DataFrame.
    """
    csv_files = list(Path(directory).glob("fivesec_predictions*.csv"))
    if not csv_files:
        print(f"Не найдено CSV файлов в {directory}")
        return None
    
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('Europe/Moscow')
            df['fivesec_pred_time'] = pd.to_datetime(df['fivesec_pred_time']).dt.tz_convert('Europe/Moscow')
            dfs.append(df)
            print(f"Загружен файл: {file}, строк: {len(df)}")
        except Exception as e:
            print(f"Ошибка при загрузке {file}: {e}")
    
    if not dfs:
        print("Нет данных для анализа")
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
    print(f"Общий DataFrame: строк={len(combined_df)}, столбцы={combined_df.columns.tolist()}")
    return combined_df

def calculate_metrics(df):
    """
    Вычисляет MSE, MAE и точность тренда для каждой модели.
    Возвращает словарь с метриками.
    """
    metrics = {}
    for model in ['rf', 'xgb', 'lgb']:
        error_col = f'fivesec_error_{model}'
        trend_acc_col = f'fivesec_trend_accuracy_{model}'
        
        if error_col not in df.columns or trend_acc_col not in df.columns:
            print(f"Пропущены столбцы для модели {model}")
            continue
        
        valid = df[df[error_col].notna()]
        if valid.empty:
            print(f"Нет валидных данных для модели {model}")
            continue
        
        mse = mean_squared_error(valid['actual_price'], valid[f'fivesec_pred_{model}'])
        mae = mean_absolute_error(valid['actual_price'], valid[f'fivesec_pred_{model}'])
        trend_accuracy = valid[trend_acc_col].mean()
        
        metrics[model] = {
            'mse': mse,
            'mae': mae,
            'trend_accuracy': trend_accuracy,
            'mean_error': valid[error_col].mean(),
            'std_error': valid[error_col].std(),
            'median_error': valid[error_col].median(),
            'count': len(valid)
        }
    
    return metrics

def plot_errors(df, output_dir):
    """
    Строит график ошибок предсказаний по времени для всех моделей.
    """
    fig = go.Figure()
    for model in ['rf', 'xgb', 'lgb']:
        error_col = f'fivesec_error_{model}'
        if error_col in df.columns:
            valid = df[df[error_col].notna()]
            fig.add_trace(go.Scatter(
                x=valid['timestamp'],
                y=valid[error_col],
                mode='lines+markers',
                name=f'Ошибка ({model.upper()})',
                line=dict(dash='dash' if model != 'rf' else 'solid')
            ))
    
    fig.update_layout(
        title="Ошибки предсказаний по времени",
        xaxis_title="Время (MSK)",
        yaxis_title="Ошибка (USDT)",
        template="plotly_dark",
        showlegend=True
    )
    fig.write_html(output_dir / "errors_plot.html")
    print(f"График ошибок сохранён в {output_dir / 'errors_plot.html'}")

def plot_trend_accuracy(df, output_dir):
    """
    Строит график точности тренда по времени для всех моделей.
    """
    fig = go.Figure()
    for model in ['rf', 'xgb', 'lgb']:
        trend_acc_col = f'fivesec_trend_accuracy_{model}'
        if trend_acc_col in df.columns:
            valid = df[df[trend_acc_col].notna()]
            fig.add_trace(go.Scatter(
                x=valid['timestamp'],
                y=valid[trend_acc_col],
                mode='lines+markers',
                name=f'Точность тренда ({model.upper()})',
                line=dict(dash='dash' if model != 'rf' else 'solid')
            ))
    
    fig.update_layout(
        title="Точность предсказания тренда по времени",
        xaxis_title="Время (MSK)",
        yaxis_title="Точность тренда",
        template="plotly_dark",
        showlegend=True
    )
    fig.write_html(output_dir / "trend_accuracy_plot.html")
    print(f"График точности тренда сохранён в {output_dir / 'trend_accuracy_plot.html'}")

def plot_predictions(df, output_dir):
    """
    Строит график фактической и предсказанных цен для всех моделей.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['actual_price'],
        mode='lines',
        name='Фактическая цена',
        line=dict(color='blue')
    ))
    
    for model in ['rf', 'xgb', 'lgb']:
        pred_col = f'fivesec_pred_{model}'
        if pred_col in df.columns:
            valid = df[df[pred_col].notna()]
            fig.add_trace(go.Scatter(
                x=valid['timestamp'],
                y=valid[pred_col],
                mode='lines',
                name=f'Предсказанная ({model.upper()})',
                line=dict(dash='dash' if model != 'rf' else 'solid')
            ))
    
    fig.update_layout(
        title="Фактическая и предсказанные цены",
        xaxis_title="Время (MSK)",
        yaxis_title="Цена (USDT)",
        template="plotly_dark",
        showlegend=True
    )
    fig.write_html(output_dir / "predictions_plot.html")
    print(f"График предсказаний сохранён в {output_dir / 'predictions_plot.html'}")

def save_metrics_table(metrics, output_dir):
    """
    Сохраняет таблицу метрик в CSV и HTML.
    """
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(output_dir / "metrics_summary.csv")
    metrics_df.to_html(output_dir / "metrics_summary.html")
    print(f"Таблица метрик сохранена в {output_dir / 'metrics_summary.csv'} и {output_dir / 'metrics_summary.html'}")

def main(directory="logs"):
    """
    Основная функция анализа.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"analysis_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    df = load_csv_files(directory)
    if df is None:
        return
    
    metrics = calculate_metrics(df)
    if not metrics:
        print("Не удалось вычислить метрики")
        return
    
    # Выводим метрики в консоль
    for model, m in metrics.items():
        print(f"\nМодель {model.upper()}:")
        print(f"  MSE: {m['mse']:.2f}")
        print(f"  MAE: {m['mae']:.2f}")
        print(f"  Точность тренда: {m['trend_accuracy']:.2%}")
        print(f"  Средняя ошибка: {m['mean_error']:.2f}")
        print(f"  Стд. отклонение ошибки: {m['std_error']:.2f}")
        print(f"  Медиана ошибки: {m['median_error']:.2f}")
        print(f"  Количество записей: {m['count']}")
    
    # Строим графики
    plot_errors(df, output_dir)
    plot_trend_accuracy(df, output_dir)
    plot_predictions(df, output_dir)
    
    # Сохраняем таблицу метрик
    save_metrics_table(metrics, output_dir)

if __name__ == "__main__":
    main()