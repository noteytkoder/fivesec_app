# realtime_analysis.py
"""
Реал-тайм приложение Dash для анализа файлов fivesec_predictions.csv.
Обновляет метрики и графики каждые 5 секунд, позволяет выбирать файлы или папку.
"""

from logger import pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
from pathlib import Path
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Укажите вашу папку с CSV файлами
CSV_DIR = "csv"  # Замените на нужный путь, например, "C:/path/to/csvs"

def load_csv_files(directory, selected_files=None):
    """
    Загружает выбранные CSV файлы или все из директории и объединяет в один DataFrame.
    """
    directory = Path(directory)
    if selected_files:
        csv_files = [directory / f for f in selected_files if (directory / f).is_file()]
    else:
        csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        return None, "Не найдено CSV файлов"
    
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('Europe/Moscow')
            df['fivesec_pred_time'] = pd.to_datetime(df['fivesec_pred_time']).dt.tz_convert('Europe/Moscow')
            dfs.append(df)
        except Exception as e:
            return None, f"Ошибка при загрузке {file}: {e}"
    
    if not dfs:
        return None, "Нет данных для анализа"
    
    combined_df = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
    return combined_df, f"Загружено {len(dfs)} файлов, строк: {len(combined_df)}"

def calculate_metrics(df):
    """
    Вычисляет MSE, MAE и точность тренда для каждой модели.
    Возвращает словарь с метриками и HTML таблицу.
    """
    metrics = {}
    table_rows = []
    for model in ['rf', 'xgb', 'lgb']:
        error_col = f'fivesec_error_{model}'
        trend_acc_col = f'fivesec_trend_accuracy_{model}'
        
        if error_col not in df.columns or trend_acc_col not in df.columns:
            continue
        
        valid = df[df[error_col].notna()]
        if valid.empty:
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
        
        table_rows.append(html.Tr([
            html.Td(model.upper()),
            html.Td(f"{mse:.2f}"),
            html.Td(f"{mae:.2f}"),
            html.Td(f"{trend_accuracy:.2%}"),
            html.Td(f"{valid[error_col].mean():.2f}"),
            html.Td(f"{valid[error_col].std():.2f}"),
            html.Td(f"{valid[error_col].median():.2f}"),
            html.Td(f"{len(valid)}")
        ]))
    
    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Модель"),
            html.Th("MSE"),
            html.Th("MAE"),
            html.Th("Точность тренда"),
            html.Th("Средняя ошибка"),
            html.Th("Стд. отклонение"),
            html.Th("Медиана ошибки"),
            html.Th("Записей")
        ])),
        html.Tbody(table_rows)
    ], style={'width': '100%', 'text-align': 'center', 'color': 'white'})
    
    return metrics, table

def create_plots(df):
    """
    Создаёт три графика: ошибки, точность тренда, предсказания.
    """
    # График ошибок
    error_fig = go.Figure()
    for model in ['rf', 'xgb', 'lgb']:
        error_col = f'fivesec_error_{model}'
        if error_col in df.columns:
            valid = df[df[error_col].notna()]
            error_fig.add_trace(go.Scatter(
                x=valid['timestamp'],
                y=valid[error_col],
                mode='lines+markers',
                name=f'Ошибка ({model.upper()})',
                line=dict(dash='dash' if model != 'rf' else 'solid')
            ))
    error_fig.update_layout(
        title="Ошибки предсказаний по времени",
        xaxis_title="Время (MSK)",
        yaxis_title="Ошибка (USDT)",
        template="plotly_dark",
        showlegend=True
    )

    # График точности тренда
    trend_fig = go.Figure()
    for model in ['rf', 'xgb', 'lgb']:
        trend_acc_col = f'fivesec_trend_accuracy_{model}'
        if trend_acc_col in df.columns:
            valid = df[df[trend_acc_col].notna()]
            trend_fig.add_trace(go.Scatter(
                x=valid['timestamp'],
                y=valid[trend_acc_col],
                mode='lines+markers',
                name=f'Точность тренда ({model.upper()})',
                line=dict(dash='dash' if model != 'rf' else 'solid')
            ))
    trend_fig.update_layout(
        title="Точность предсказания тренда",
        xaxis_title="Время (MSK)",
        yaxis_title="Точность тренда",
        template="plotly_dark",
        showlegend=True
    )

    # График предсказаний
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(
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
            pred_fig.add_trace(go.Scatter(
                x=valid['timestamp'],
                y=valid[pred_col],
                mode='lines',
                name=f'Предсказанная ({model.upper()})',
                line=dict(dash='dash' if model != 'rf' else 'solid')
            ))
    pred_fig.update_layout(
        title="Фактическая и предсказанные цены",
        xaxis_title="Время (MSK)",
        yaxis_title="Цена (USDT)",
        template="plotly_dark",
        showlegend=True
    )

    return error_fig, trend_fig, pred_fig

# Инициализация Dash приложения
app = dash.Dash(__name__)
app.title = "FiveSec Predictions Analysis"

# Получаем список доступных CSV файлов
csv_files = [f.name for f in Path(CSV_DIR).glob("*.csv")]

# Макет приложения
app.layout = html.Div([
    html.H1("Анализ предсказаний FiveSec", style={'color': 'white', 'text-align': 'center'}),
    html.Label("Выберите папку или файлы:", style={'color': 'white'}),
    dcc.Dropdown(
        id='file-selector',
        options=[{'label': 'Все файлы в папке', 'value': 'all'}] + [{'label': f, 'value': f} for f in csv_files],
        value='all',
        multi=True,
        style={'backgroundColor': '#1a1a1a', 'color': '#000000'}
    ),
    html.Div(id='status-message', style={'color': 'white', 'margin': '10px'}),
    dcc.Graph(id='errors-plot'),
    dcc.Graph(id='trend-accuracy-plot'),
    dcc.Graph(id='predictions-plot'),
    html.Div(id='metrics-table', style={'margin': '20px'}),
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0)  # Обновление каждые 5 секунд
], style={'backgroundColor': '#1a1a1a', 'padding': '20px'})

@app.callback(
    [Output('errors-plot', 'figure'),
     Output('trend-accuracy-plot', 'figure'),
     Output('predictions-plot', 'figure'),
     Output('metrics-table', 'children'),
     Output('status-message', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('file-selector', 'value')]
)
def update_analysis(n_intervals, selected_files):
    """
    Обновляет графики и метрики в реальном времени.
    """
    if selected_files == 'all' or not selected_files:
        df, status = load_csv_files(CSV_DIR)
    else:
        df, status = load_csv_files(CSV_DIR, selected_files)
    
    if df is None:
        return go.Figure(), go.Figure(), go.Figure(), "Нет данных", status
    
    metrics, table = calculate_metrics(df)
    error_fig, trend_fig, pred_fig = create_plots(df)
    
    return error_fig, trend_fig, pred_fig, table, status

if __name__ == "__main__":
    app.run(debug=True, port=8070)