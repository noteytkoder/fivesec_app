# dashboard/analysis.py
"""
Реал-тайм анализ предсказаний из fivesec_predictions.csv.
Интегрировано в dashboard.
"""
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from logger import pd, setup_logger
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config_manager import load_config, load_environment_config
from dash_auth import BasicAuth
import dash_bootstrap_components as dbc  # Для стилей и таблиц
import os
import logging
import traceback

# Настройка логгера
logger = setup_logger()

def load_csv_files(directory, config):
    """
    Загружает fivesec_predictions.csv и возвращает DataFrame.
    """
    logger.debug(f"Запуск load_csv_files: directory={directory}")
    try:
        directory = Path(directory)
        csv_file = directory / 'fivesec_predictions.csv'
        logger.debug(f"Проверка файла: {csv_file}")
        
        if not csv_file.is_file():
            logger.warning(f"Файл {csv_file} не найден")
            return None, f"Файл {csv_file} не найден"
        
        try:
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(config['timezone'])
            df['fivesec_pred_time'] = pd.to_datetime(df['fivesec_pred_time']).dt.tz_convert(config['timezone'])
            logger.info(f"Успешно загружен файл {csv_file}, размер DataFrame: {df.shape}")
            return df, f"Загружен файл, строк: {len(df)}"
        except Exception as e:
            logger.error(f"Ошибка при загрузке {csv_file}: {str(e)}", exc_info=True)
            return None, f"Ошибка при загрузке {csv_file}: {str(e)}"
    except Exception as e:
        logger.error(f"Ошибка в load_csv_files: {str(e)}", exc_info=True)
        return None, f"Ошибка в load_csv_files: {str(e)}"

def calculate_metrics(df):
    """
    Вычисляет MSE, MAE и точность тренда для каждой модели.
    Возвращает словарь с метриками и HTML таблицу.
    """
    logger.debug("Запуск calculate_metrics")
    try:
        metrics = {}
        table_rows = []
        for model in ['rf', 'xgb', 'lgb']:
            error_col = f'fivesec_error_{model}'
            trend_acc_col = f'fivesec_trend_accuracy_{model}'
            
            logger.debug(f"Проверка модели {model}: error_col={error_col}, trend_acc_col={trend_acc_col}")
            if error_col not in df.columns or trend_acc_col not in df.columns:
                logger.warning(f"Пропуск модели {model}: отсутствуют столбцы {error_col} или {trend_acc_col}")
                continue
            
            valid = df[df[error_col].notna()]
            if valid.empty:
                logger.warning(f"Пропуск модели {model}: нет валидных данных")
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
            
            logger.debug(f"Метрики для {model}: MSE={mse:.2f}, MAE={mae:.2f}, Trend Accuracy={trend_accuracy:.2%}")
            
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
        
        table = dbc.Table([
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
        ], bordered=True, hover=True, responsive=True, striped=True, style={'color': 'white'})
        
        logger.info(f"Рассчитаны метрики: {len(metrics)} моделей обработано")
        return metrics, table
    except Exception as e:
        logger.error(f"Ошибка в calculate_metrics: {str(e)}", exc_info=True)
        return {}, "Ошибка при расчете метрик"

def create_plots(df, config):
    """
    Создаёт два графика: ошибки, предсказания.
    Использует темную тему и цветовые настройки из config.
    """
    logger.debug("Запуск create_plots")
    try:
        # График ошибок
        error_fig = go.Figure()
        for model, color in [('rf', 'orange'), ('xgb', 'purple'), ('lgb', 'green')]:
            error_col = f'fivesec_error_{model}'
            if error_col in df.columns:
                valid = df[df[error_col].notna()]
                error_fig.add_trace(go.Scatter(
                    x=valid['timestamp'],
                    y=valid[error_col],
                    mode='lines+markers',
                    name=f'Ошибка ({model.upper()})',
                    line=dict(dash='dash' if model != 'rf' else 'solid', color=color)
                ))
                logger.debug(f"Добавлен график ошибок для модели {model}")
        error_fig.update_layout(
            title="Ошибки предсказаний по времени",
            xaxis_title="Время (MSK)",
            yaxis_title="Ошибка (USDT)",
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
            line=dict(color=config['visual']['real_price_color'])
        ))
        for model, color in [('rf', 'orange'), ('xgb', 'purple'), ('lgb', 'green')]:
            pred_col = f'fivesec_pred_{model}'
            if pred_col in df.columns:
                valid = df[df[pred_col].notna()]
                pred_fig.add_trace(go.Scatter(
                    x=valid['timestamp'],
                    y=valid[pred_col],
                    mode='lines',
                    name=f'Предсказанная ({model.upper()})',
                    line=dict(dash='dash' if model != 'rf' else 'solid', color=color)
                ))
                logger.debug(f"Добавлен график предсказаний для модели {model}")
        pred_fig.update_layout(
            title="Фактическая и предсказанные цены",
            xaxis_title="Время (MSK)",
            yaxis_title="Цена (USDT)",
            template="plotly_dark",
            showlegend=True
        )

        logger.info("Графики успешно созданы")
        return error_fig, pred_fig
    except Exception as e:
        logger.error(f"Ошибка в create_plots: {str(e)}", exc_info=True)
        return go.Figure(), go.Figure()

def create_analysis_app():
    """
    Создает Dash приложение для анализа предсказаний.
    """
    logger.debug("Запуск create_analysis_app")
    try:
        config = load_config()
        env_config = load_environment_config()
        logger.info(f"Конфигурация загружена: app_env={config.get('app_env')}, port_analysis={env_config.get('port_analysis')}")
        
        app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])  # Темная тема
        logger.debug("Dash приложение инициализировано с темой DARKLY")
        
        creds = {config["auth"]["username"]: config["auth"]["password"]}
        BasicAuth(app, creds)  # Добавляем аутентификацию
        logger.debug("BasicAuth настроен")

        CSV_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'logs')
        logger.debug(f"Установлен CSV_DIR: {CSV_DIR}")
        
        app.layout = html.Div([
            html.H1("Анализ предсказаний FiveSec", style={'color': 'white', 'text-align': 'center'}),
            html.Div(id='status-message', style={'color': 'white', 'margin': '10px'}),
            dcc.Graph(id='errors-plot'),
            dcc.Graph(id='predictions-plot'),
            html.Div(id='metrics-table', style={'margin': '20px'}),
            dcc.Interval(id='interval-component', interval=config['visual']['update_interval'], n_intervals=0)
        ], style={'backgroundColor': '#1a1a1a', 'padding': '20px'})
        logger.debug("Layout приложения создан")
        
        @app.callback(
            [Output('errors-plot', 'figure'),
             Output('predictions-plot', 'figure'),
             Output('metrics-table', 'children'),
             Output('status-message', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_analysis(n_intervals):
            logger.debug(f"Запуск update_analysis: n_intervals={n_intervals}")
            try:
                df, status = load_csv_files(CSV_DIR, config)
                
                if df is None:
                    logger.warning("Данные не загружены, возвращаются пустые графики")
                    return go.Figure(), go.Figure(), "Нет данных", status
                
                metrics, table = calculate_metrics(df)
                error_fig, pred_fig = create_plots(df, config)
                
                logger.info("Обновление анализа завершено успешно")
                return error_fig, pred_fig, table, status
            except Exception as e:
                logger.error(f"Ошибка в update_analysis: {str(e)}", exc_info=True)
                return go.Figure(), go.Figure(), "Ошибка при обновлении", f"Ошибка: {str(e)}"
        
        logger.info("Dash приложение для анализа успешно создано")
        return app
    except Exception as e:
        logger.error(f"Ошибка при создании приложения анализа: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logger.debug("Запуск приложения в режиме отладки")
        app = create_analysis_app()
        port = load_environment_config().get('port_analysis', 8070)
        logger.info(f"Запуск сервера на порту {port}")
        app.run_server(debug=True, port=port)
    except Exception as e:
        logger.error(f"Ошибка при запуске сервера: {str(e)}", exc_info=True)
        raise