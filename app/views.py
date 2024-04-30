from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter

import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt
import qrcode

import json

from .models import Project

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import os
import json
from django.conf import settings

# Construct the file path relative to the 'app' directory
file_path = os.path.join(settings.BASE_DIR, 'app', 'valid.json')

try:
    with open(file_path, 'r') as json_file:
        Valid_Ticker = json.load(json_file)
except FileNotFoundError:
    # Handle the case when the file is not found
    Valid_Ticker = []

def index(request):
    return render(request,'home.html')


def allticket(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'allticket.html', {
        'ticker_list': ticker_list
    })


def predictticket(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'predict.html', {
        'ticker_list': ticker_list
    })

# The Predict Function to implement Machine Learning as well as Plotting
def finalresult(request, ticker_value, number_of_days):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        return render(request, 'API_Down.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})

    
    if ticker_value not in Valid_Ticker:
        return render(request, 'Invalid_Ticker.html', {})
    
    if number_of_days < 0:
        return render(request, 'Negative_Days.html', {})
    
    if number_of_days > 365:
        return render(request, 'Overflow_days.html', {})
    

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')



    # ========================================== Machine Learning ==========================================

    pred_dict2 = {"Date": [], "Prediction": []}
    pred_dict3 = {"Date": [], "Prediction": []}


    try:
        df_ml = yf.download(tickers = ticker_value, period='3mo', interval='1h')
    except:
        ticker_value = 'AAPL'
        df_ml = yf.download(tickers = ticker_value, period='3mo', interval='1m')

    # Fetching ticker values from Yahoo Finance API 
    df_ml = df_ml[['Adj Close']]
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'],axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
    # Applying Linear Regression
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    # Prediction Score
    confidence = clf.score(X_test, y_test)
    # Predicting for 'n' days stock data
    forecast_prediction = clf.predict(X_forecast)
    forecast = forecast_prediction.tolist()

    # Making predictions for Linear Regression
    y_pred_lr = clf.predict(X_test)

    # Calculate Linear Regression metrics
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_mse = mean_squared_error(y_test, y_pred_lr)
    lr_r2 = r2_score(y_test, y_pred_lr)

    # Overall accuracy in percentage for Linear Regression
    overall_accuracy_lr = lr_r2 * 100
    overall_accuracy_lr = round(overall_accuracy_lr, 2)

    print("Linear Regression Metrics:")
    print("MAE:", lr_mae)
    print("MSE:", lr_mse)
    print("R-squared:", lr_r2)
    print("Overall Accuracy:", overall_accuracy_lr, "%")
    
        
    # Support Vector Regression
    svr = SVR(kernel='linear')
    svr.fit(X_train, y_train)
    forecast_prediction_svr = svr.predict(X_forecast)
    forecast_svr = forecast_prediction_svr.tolist()

    # Calculate SVR metrics
    svr_mae = mean_absolute_error(y_test, svr.predict(X_test))
    svr_mse = mean_squared_error(y_test, svr.predict(X_test))
    svr_r2 = r2_score(y_test, svr.predict(X_test)) * 100
    svr_r2 = round(svr_r2, 2)

    # Store predictions in pred_dict2
    for i in range(0, len(forecast_svr)):
        formatted_prediction = "{:.2f}".format(forecast_svr[i])
        pred_dict2["Date"].append((dt.datetime.today() + dt.timedelta(days=i)).strftime('%Y-%m-%d'))
        pred_dict2["Prediction"].append(formatted_prediction)

    # Random Forest Regression
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    forecast_prediction_rf = rf.predict(X_forecast)
    forecast_rf = forecast_prediction_rf.tolist()

    # Calculate RF metrics
    rf_mae = mean_absolute_error(y_test, rf.predict(X_test))
    rf_mse = mean_squared_error(y_test, rf.predict(X_test))
    rf_r2 = r2_score(y_test, rf.predict(X_test)) * 100
    rf_r2 = round(rf_r2, 2)

    # Store predictions in pred_dict3
    for i in range(0, len(forecast_rf)):
        formatted_prediction = "{:.2f}".format(forecast_rf[i])
        pred_dict3["Date"].append((dt.datetime.today() + dt.timedelta(days=i)).strftime('%Y-%m-%d'))
        pred_dict3["Prediction"].append(formatted_prediction)


    pred_df2 = pd.DataFrame(pred_dict2)
    pred_df3 = pd.DataFrame(pred_dict3)

    print("SVR Metrics:")
    print("MAE:", svr_mae)
    print("MSE:", svr_mse)
    print("R-squared:", svr_r2)
    print("\nRandom Forest Metrics:")
    print("MAE:", rf_mae)
    print("MSE:", rf_mse)
    print("R-squared:", rf_r2)

    

    # ========================================== Plotting predicted data ======================================


    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        formatted_prediction = "{:.2f}".format(forecast[i])
        pred_dict["Date"].append((dt.datetime.today() + dt.timedelta(days=i)).strftime('%Y-%m-%d'))
        pred_dict["Prediction"].append(formatted_prediction)
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#cc33ff", plot_bgcolor="#cc33ff", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')
    sample_graph= px.bar(pred_df,x='Date',y='Date')
    sample_graph.update_layout(
        paper_bgcolor="#1a8cff",
        plot_bgcolor="#ccccff",  
        font_color="white",
        height=1000
    )
    sample_graph = plot(sample_graph , auto_open=False , output_type='div')
    
    # Convert pred_df2 to the format of pred_dict
    pred_dict2 = {"Date": pred_df2['Date'].tolist(), "Prediction": pred_df2['Prediction'].tolist()}

    # Convert pred_df3 to the format of pred_dict
    pred_dict3 = {"Date": pred_df3['Date'].tolist(), "Prediction": pred_df3['Prediction'].tolist()}


    # ========================================== Display Ticker Info ==========================================

    ticker = pd.read_csv('app/Data/Tickers.csv')
    to_search = ticker_value
    ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                    'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Last_Sale = ticker.Last_Sale[i]
            Net_Change = ticker.Net_Change[i]
            Percent_Change = ticker.Percent_Change[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Volume = ticker.Volume[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break

    # ========================================== Page Render section ==========================================
    

    return render(request, "final.html", 
                  context={ 
                      'plot_div': plot_div, 
                      'confidence' : confidence, 
                      'forecast': forecast, 
                      'ticker_value':ticker_value,
                      'number_of_days':number_of_days,
                      'plot_div_pred':plot_div_pred,
                      'sample_graph':sample_graph,
                      'Symbol':Symbol,
                      'Name':Name,
                      'Last_Sale':Last_Sale,
                      'Net_Change':Net_Change,
                      'Percent_Change':Percent_Change,
                      'Market_Cap':Market_Cap,
                      'Country':Country,
                      'IPO_Year':IPO_Year,
                      'Volume':Volume,
                      'Sector':Sector,
                      'Industry':Industry,
                      'pred_dict':pred_dict,
                      'pred_dict2':pred_dict2,
                      'pred_dict3':pred_dict3,
                      'acc1':overall_accuracy_lr,
                      'acc2':svr_r2,
                      'acc3':rf_r2
                      })
