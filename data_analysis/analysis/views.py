from django.shortcuts import render
from django.core.paginator import Paginator
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import os
import logging
from .models import StockData
from plotly.offline import plot
from statsmodels.tsa.stattools import acf, pacf

# 构建 CSV 文件的相对路径
csv_file_path = os.path.join('analysis', 'static', 'tesla_stock_data.csv')
# 使用相对路径打开 CSV 文件

def upload_csv(request):
    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']
        csv_data = pd.read_csv(csv_file)

        # 将日期列设置为索引
        csv_data['Date'] = pd.to_datetime(csv_data['Date'])
        csv_data.set_index('Date', inplace=True)

        # 分析数据并保存到数据库中
        for i in range(len(csv_data)):
            stock_data = StockData(
                date = csv_data.index[i],
                year=csv_data['Year'][i],
                open_price=csv_data['Open'][i],
                high_price=csv_data['High'][i],
                low_price=csv_data['Low'][i],
                close_price=csv_data['Close'][i],
                volume=csv_data['Volume'][i],
                adj_close=csv_data['Adj Close'][i])
            stock_data.save()
        
    return render(request, 'home.html')

def home(request):
    return render(request, 'home.html')

def first_views(request):
    stock_data = StockData.objects.all()
    paginator = Paginator(stock_data, 10)  # Show 10 rows per page

    page_number = request.GET.get('page')  # Get the page number from the URL
    page_obj = paginator.get_page(page_number)  # Get the desired page

    context = {'page_obj': page_obj}
    return render(request, 'first_views.html', context)

def description_view(request):
    stock_data = StockData.objects.all() 
    data = pd.DataFrame(list(stock_data.values()))

    if not data.empty:
        description = data.describe().T
    else:
        description = pd.DataFrame()
    context = {'description': description.to_html(classes='table table-striped')}
    # 渲染到模板，传递描述性统计数据
    return render(request, 'description.html', context)

def missing_value_view(request):
    stock_data = StockData.objects.all()
    data = pd.DataFrame(list(stock_data.values()))
    # 检查缺失值
    if not data.empty:
        # 计算每个字段的缺失值数量
        missing_values = data.isnull().sum()
        missing_values_dict = missing_values.to_dict()
    else:
        missing_values_dict = {}

    # 将缺失值信息传递到模板
    context = {'missing_values': missing_values_dict}
    return render(request, 'missing_values.html', context)

def open_prices_view(request):
    stock_data = StockData.objects.all() 
    data = pd.DataFrame(list(stock_data.values()))

    # 确保数据不为空，并且数据中包含日期和开盘价
    if not data.empty and 'date' in data and 'open_price' in data:
        data.sort_values('date', inplace=True)  # 按日期排序

        # 创建图表
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['open_price'], mode='lines', name='Open Prices'))
        fig.update_layout(title='Tesla Stock Open Prices Over Time',
                          xaxis_title='Date',
                          yaxis_title='Open Price')

        # 将图表转换为HTML
        graph_html = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        graph_html = "<div>No data available.</div>"

    context = {'graph_html': graph_html}
    return render(request, 'open_prices.html', context)

def distribution_open_prices_view(request):
    stock_data = StockData.objects.all()
    data = pd.DataFrame(list(stock_data.values()))

    # 确保数据不为空，并且包含开盘价
    if not data.empty and 'open_price' in data:
        # 创建直方图
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Histogram(x=data['open_price'], nbinsx=30, marker_color='skyblue'))
        fig.update_layout(title='Distribution of Tesla Stock Opening Prices',
                          xaxis_title='Opening Price',
                          yaxis_title='Frequency')

        # 将图表转换为 HTML
        graph_html = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        graph_html = "<div>No data available.</div>"

    context = {'graph_html': graph_html}
    return render(request, 'distribution_opening_prices.html', context)

def close_prices_view(request):
    # 从数据库获取 Tesla 的股票数据
    stock_data = StockData.objects.all()
    data = pd.DataFrame(list(stock_data.values()))

    # 确保数据不为空，并且数据中包含日期和收盘价
    if not data.empty and 'date' in data and 'close_price' in data:
        data.sort_values('date', inplace=True)  # 按日期排序

        # 创建图表
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['close_price'], mode='lines', name='Close Prices'))
        fig.update_layout(title='Tesla Stock Close Prices Over Time',
                          xaxis_title='Date',
                          yaxis_title='Close Price')

        # 将图表转换为 HTML
        graph_html = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        graph_html = "<div>No data available.</div>"

    context = {'graph_html': graph_html}
    return render(request, 'close_prices.html', context)

def distribution_closing_prices_view(request):
    # 从数据库获取 Tesla 的股票数据
    stock_data = StockData.objects.all()  # 确保你的模型中有一个可以区分股票的字段
    data = pd.DataFrame(list(stock_data.values()))

    # 确保数据不为空，并且包含收盘价
    if not data.empty and 'close_price' in data:
        # 创建直方图
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Histogram(x=data['close_price'], nbinsx=30, marker_color='skyblue'))
        fig.update_layout(title='Distribution of Stock Close Prices',
                          xaxis_title='Close Price',
                          yaxis_title='Frequency')

        # 将图表转换为 HTML
        graph_html = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        graph_html = "<div>No data available.</div>"

    context = {'graph_html': graph_html}
    return render(request, 'distribution_closing_prices.html', context)

def low_prices_view(request):
    # Fetching Tesla's stock data
    stock_data = StockData.objects.all()  # Ensure your model has a field to identify the stock
    data = pd.DataFrame(list(stock_data.values()))

    # Make sure data is not empty and contains the necessary 'date' and 'low' price data
    if not data.empty and 'date' in data and 'low_price' in data:
        data.sort_values('date', inplace=True)  # Sorting data by date

        # Create the plot
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['low_price'], mode='lines', name='Low', line=dict(color='skyblue')))
        fig.update_layout(title='Tesla Stock Low Prices Over Time',
                          xaxis_title='Date',
                          yaxis_title='Price')

        # Convert the plot to HTML
        graph_html = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        graph_html = "<div>No data available.</div>"

    context = {'graph_html': graph_html}
    return render(request, 'low_prices.html', context)

def high_prices_view(request):
    # Fetching Tesla's stock data from the database
    stock_data = StockData.objects.all()  # Adjust based on your model's structure
    data = pd.DataFrame(list(stock_data.values()))

    # Ensure there is data and it includes 'date' and 'high_price'
    if not data.empty and 'date' in data and 'high_price' in data:
        data.sort_values('date', inplace=True)  # Sort data by date

        # Create the plot
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['high_price'], mode='lines', name='High', line=dict(color='skyblue')))
        fig.update_layout(title='Tesla Stock High Prices Over Time',
                          xaxis_title='Date',
                          yaxis_title='Price')

        # Convert the plot to HTML
        graph_html = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        graph_html = "<div>No data available.</div>"

    context = {'graph_html': graph_html}
    return render(request, 'high_prices.html', context)

def stock_prices_view(request):
    # Fetching Tesla's stock data from the database
    stock_data = StockData.objects.all() # Adjust based on your model's structure
    data = pd.DataFrame(list(stock_data.values()))

    # Ensure there is data and it includes necessary columns like 'date', 'open_price', 'high_price', and 'low_price'
    if not data.empty and 'date' in data and 'open_price' in data and 'high_price' in data and 'low_price' in data:
        data.sort_values('date', inplace=True)  # Sorting data by date for proper chronological plotting

        # Create the Plotly figure
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=data['date'], y=data['open_price'], mode='lines', name='Open'))
        fig.add_trace(go.Scatter(x=data['date'], y=data['high_price'], mode='lines', name='High'))
        fig.add_trace(go.Scatter(x=data['date'], y=data['low_price'], mode='lines', name='Low'))
        fig.update_layout(
            title='Stock Prices Over Time',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white'
        )

        # Convert the plot to HTML
        graph_html = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        graph_html = "<div>No data available.</div>"

    context = {'graph_html': graph_html}
    return render(request, 'stock_prices.html', context)

def rolling_stats_view(request):
    # Fetching Tesla's stock data from the database
    stock_data = StockData.objects.all()  # Make sure your model has a way to distinguish the stock
    data = pd.DataFrame(list(stock_data.values()))

    # Check if data contains the necessary columns
    if not data.empty and 'date' in data and 'close_price' in data:
        data.sort_values('date', inplace=True)  # Sort by date for proper plotting

        # Calculate rolling window statistics
        data.set_index('date', inplace=True)  # Set date as the index for rolling calculations
        rolling_mean = data['close_price'].rolling(window=30).mean()
        rolling_std = data['close_price'].rolling(window=30).std()

        # Create the Plotly figure
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['close_price'], mode='lines', name='Closing Price'))
        fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Mean'))
        fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, mode='lines', name='Rolling Std'))
        fig.update_layout(
            title='Rolling Window Statistics',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white'
        )

        # Convert the plot to HTML
        graph_html = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        graph_html = "<div>No data available.</div>"

    context = {'graph_html': graph_html}
    return render(request, 'rolling_stats.html', context)

def pairplot_view(request):
    # Fetch Tesla's stock data from the database
    stock_data = StockData.objects.all()  # Adjust if your model distinguishes stocks differently
    data = pd.DataFrame(list(stock_data.values()))

    # Ensure data is not empty and contains the required fields
    if not data.empty:
        # Create the pairplot using Plotly Express
        pairplot = px.scatter_matrix(data,
                                     dimensions=data.columns,
                                     title='Pairplot for Feature Relationships')
        pairplot.update_traces(diagonal_visible=False,
                               marker=dict(opacity=0.6),
                               selector=dict(type='scatter'),
                               showupperhalf=False)
        pairplot.update_layout(title_font_size=20,
                               title_y=0.9,
                               title_x=0.5,
                               title_font_color='blue',
                               plot_bgcolor='lightgrey')

        # Convert the plot to HTML
        graph_html = plot(pairplot, output_type='div', include_plotlyjs=False)
    else:
        graph_html = "<div>No data available.</div>"

    context = {'graph_html': graph_html}
    return render(request, 'pairplot.html', context)

def correlation_matrix_view(request):
    # Fetch Tesla's stock data
    stock_data = StockData.objects.all()
    data = pd.DataFrame(list(stock_data.values('open_price', 'high_price', 'low_price', 'close_price', 'volume', 'adj_close')))

    # Handle missing values by dropping them (for simplicity, though filling might be better depending on context)
    data.dropna(inplace=True)

    # Generate correlation matrix plot
    if not data.empty:
        correlation_matrix = data.corr()

        # Prepare annotation text, ensuring it matches the dimensions of `correlation_matrix`
        annotation_text = correlation_matrix.round(2).astype(str).values.tolist()  # Convert to list of lists of strings

        fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.columns.tolist(),
            annotation_text=annotation_text,  # correctly formatted text
            colorscale='Viridis',
            showscale=True
        )
        fig.update_layout(
            title='Correlation Matrix of Tesla Stock Features',
            xaxis_title='Features',
            yaxis_title='Features',
            xaxis={'side': 'bottom'},
            margin=dict(l=10, r=10, t=30, b=10)
        )

        # Convert plot to HTML div element
        div = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        div = "<div>No data available to display the correlation matrix.</div>"

    context = {'graph_html': div}
    return render(request, 'correlation_matrix.html', context)

def candlestick_view(request):
    # 从数据库获取数据并按日期排序
    stock_data = StockData.objects.all().order_by('date')
    df = pd.DataFrame(list(stock_data.values()))

    # 转换数据类型，确保日期为索引
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 准备 K 线图所需数据
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open_price'],
        high=df['high_price'],
        low=df['low_price'],
        close=df['close_price'],
        name='Candlestick'
    )])

    # 添加移动平均线
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close_price'].rolling(window=20).mean(),
        mode='lines',
        line=dict(width=1.5),
        name='20-Day MAVG'
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close_price'].rolling(window=50).mean(),
        mode='lines',
        line=dict(width=1.5),
        name='50-Day MAVG'
    ))

    # 设置图表布局
    fig.update_layout(
        title='Tesla Stock Candlestick Chart with Moving Averages',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False  # 隐藏范围滑块
    )

    # 将图表转换为 HTML div
    div = plot(fig, output_type='div', include_plotlyjs=True)

    return render(request, 'candlestick_chart.html', {'graph_html': div})

def calculate_technical_indicators(data):
    # Calculate RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    # Calculate MACD (Moving Average Convergence Divergence)
    shortEMA = data['Close'].ewm(span=12, adjust=False).mean()
    longEMA = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = shortEMA - longEMA
    signalEMA = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Signal_Line'] = signalEMA

    # Calculate Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['MA20'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower_Band'] = data['MA20'] - 2 * data['Close'].rolling(window=20).std()

    return data

def plot_technical_indicators_view(request):
    # 获取数据并计算技术指标
    stock_data = StockData.objects.all().order_by('date')
    df = pd.DataFrame(list(stock_data.values()))
    df.rename(columns={
        'open_price': 'Open',
        'high_price': 'High',
        'low_price': 'Low',
        'close_price': 'Close',
        'volume': 'Volume',
        'date': 'Date'  # 确保有一个日期字段
    }, inplace=True)
    tesla_data = calculate_technical_indicators(df)  # 假设df已经正确格式化

    # 创建绘图
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=('RSI', 'MACD and Signal Line', 'Bollinger Bands'))

    # RSI
    fig.add_trace(go.Scatter(x=tesla_data.index, y=tesla_data['RSI'], name='RSI', line=dict(color='blue')), row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=tesla_data.index, y=tesla_data['MACD'], name='MACD', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=tesla_data.index, y=tesla_data['Signal_Line'], name='Signal Line', line=dict(color='green')), row=2, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=tesla_data.index, y=tesla_data['Close'], name='Closing Price', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=tesla_data.index, y=tesla_data['Upper_Band'], name='Upper Band', line=dict(color='red', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=tesla_data.index, y=tesla_data['Lower_Band'], name='Lower Band', line=dict(color='green', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=tesla_data.index, y=tesla_data['Upper_Band'], fill='tonexty', mode='none', name='Bollinger Band Fill', fillcolor='rgba(128, 128, 128, 0.3)'), row=3, col=1)

    # 更新布局
    fig.update_layout(height=900, title_text='Tesla Stock Technical Analysis')

    # 将图表转换为HTML div
    graph_html = plot(fig, output_type='div', include_plotlyjs=False)

    return render(request, 'technical_analysis.html', {'graph_html': graph_html})

def monthly_trends_view(request):
    # 获取数据
    stock_data = StockData.objects.all().order_by('date')
    df = pd.DataFrame(list(stock_data.values()))

    # 确保日期列正确，并设置为索引
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 提取月份和年份
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    # 按月份和年份分组，计算平均收盘价
    monthly_mean = df.groupby(['Year', 'Month'])['close_price'].mean()

    # 将结果转换为新的DataFrame，以便绘图
    monthly_mean_df = monthly_mean.reset_index()
    monthly_mean_df['Date'] = pd.to_datetime(monthly_mean_df['Year'].astype(str) + '-' + monthly_mean_df['Month'].astype(str))

    # 使用Plotly绘制月度趋势
    fig = px.line(monthly_mean_df, x='Date', y='close_price',
                  labels={'close_price': 'Mean Closing Price'},
                  title='Monthly Trends of Tesla Stock Closing Prices')
    fig.update_layout(xaxis_title='Date', yaxis_title='Mean Closing Price')

    # 将图表转换为HTML div
    graph_html = plot(fig, output_type='div', include_plotlyjs=False)

    return render(request, 'monthly_trends.html', {'graph_html': graph_html})

def acf_pacf_view(request):
    # Fetch data from the database
    stock_data = StockData.objects.all().order_by('date')
    df = pd.DataFrame(list(stock_data.values()))

    # Ensure the date column is in the correct format and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Calculate ACF and PACF
    lag_acf = acf(df['close_price'], nlags=40)
    lag_pacf = pacf(df['close_price'], nlags=40)

    # Create ACF Plot
    acf_fig = go.Figure()
    acf_fig.add_trace(go.Bar(x=list(range(41)), y=lag_acf, name='ACF'))
    acf_fig.update_layout(
        title='Autocorrelation Function of Tesla Stock Closing Prices',
        xaxis_title='Lag',
        yaxis_title='Autocorrelation',
        template='plotly_white'
    )

    # Create PACF Plot
    pacf_fig = go.Figure()
    pacf_fig.add_trace(go.Bar(x=list(range(41)), y=lag_pacf, name='PACF'))
    pacf_fig.update_layout(
        title='Partial Autocorrelation Function of Tesla Stock Closing Prices',
        xaxis_title='Lag',
        yaxis_title='Partial Autocorrelation',
        template='plotly_white'
    )

    # Convert plots to HTML divs
    acf_graph_html = plot(acf_fig, output_type='div', include_plotlyjs=False)
    pacf_graph_html = plot(pacf_fig, output_type='div', include_plotlyjs=False)

    return render(request, 'acf_pacf.html', {'acf_graph_html': acf_graph_html, 'pacf_graph_html': pacf_graph_html})

def event_analysis_view(request):
    # Fetch data from the database
    stock_data = StockData.objects.all().order_by('date')
    df = pd.DataFrame(list(stock_data.values()))

    # Ensure the date column is in the correct format and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Define event date and price
    event_date = pd.Timestamp('2023-06-01')
    event_price = df.loc[event_date, 'close_price'] if event_date in df.index else None

    # Create Plotly figure
    fig = go.Figure()

    # Add stock price line
    fig.add_trace(go.Scatter(x=df.index, y=df['close_price'], mode='lines', name='Tesla Stock Price', line=dict(color='blue')))

    # Add event line and marker if event date exists in data
    if event_price:
        # Convert Timestamp to the correct format for Plotly
        formatted_event_date = event_date.strftime('%Y-%m-%d')
        timestamp = (event_date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') * 1000
        fig.add_vline(x=timestamp, line=dict(color='red', dash='dash'), annotation_text="Event Date")
        fig.add_trace(go.Scatter(x=[formatted_event_date], y=[event_price], mode='markers', marker=dict(color='red', size=10), name='Event: XYZ'))

    # Set layout
    fig.update_layout(
        title='Event Analysis: Impact on Tesla Stock Price',
        xaxis_title='Date',
        yaxis_title='Closing Price',
        showlegend=True,
        template='plotly_white'
    )

    # Convert plot to HTML div
    graph_html = plot(fig, output_type='div', include_plotlyjs=True)

    return render(request, 'event_analysis.html', {'graph_html': graph_html})