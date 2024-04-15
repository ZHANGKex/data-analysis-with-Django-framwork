# analysis/urls.py

from django.urls import path
from .views import (home, upload_csv, first_views, description_view, missing_value_view, open_prices_view, 
distribution_open_prices_view, close_prices_view, distribution_closing_prices_view, low_prices_view, 
high_prices_view, stock_prices_view, rolling_stats_view, pairplot_view ,correlation_matrix_view, candlestick_view,
plot_technical_indicators_view, monthly_trends_view,acf_pacf_view, event_analysis_view)

urlpatterns = [
    path('', home, name='home'),
    path('upload-data/', upload_csv, name='upload'),
    path('first_views/',first_views, name='first_views'),
    path('description/', description_view, name='description'),
    path('missing_value/', missing_value_view, name='missing_value'),
    path('open-prices/', open_prices_view, name='open-prices'),
    path('distribution-opening-prices/', distribution_open_prices_view, name='distribution-opening-prices'),
    path('close-prices/', close_prices_view, name='close-prices'),
    path('distribution-closing-prices/', distribution_closing_prices_view, name='distribution-closing-prices'),
    path('low-prices/', low_prices_view, name='low-prices'),
    path('high-prices/', high_prices_view, name='high-prices'),
    path('stock-prices/', stock_prices_view, name='stock-prices'),
    path('rolling-stats/', rolling_stats_view, name='rolling-stats'),
    path('pairplot/', pairplot_view, name='pairplot'),
    path('correlation-matrix/', correlation_matrix_view, name='correlation-matrix'),
    path('candlestick/', candlestick_view, name='candlestick'),
    path('plot_technical_indicators/', plot_technical_indicators_view, name='plot_technical_indicators'),
    path('monthly-trends/', monthly_trends_view, name='monthly-trends'),
    path('acf-pacf/', acf_pacf_view, name='acf-pacf'),
    path('event-analysis/', event_analysis_view, name='event-analysis'),
]
