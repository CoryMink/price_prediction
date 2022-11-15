import pandas    as pd
import numpy     as np
import yfinance  as yf
import pandas_ta as ta
import datetime  as dt
import streamlit as st
import time
from sklearn.model_selection  import train_test_split
from sklearn.linear_model     import LinearRegression

#==================================================================================

def get_pred(sym):
    """
    sym = symbol based on Yahoo Finance
    get prediction of 
    """
    # Data request from Yahoo Finance
    tickers=sym
    enddate = dt.datetime.now()
    startdate = enddate - dt.timedelta(minutes=15000)
    lists1 = []
    for intervals in['15m','30m','60m','90m']:
        data = yf.download(tickers=tickers, start=startdate, end=enddate, interval=intervals,progress=False)
        data['NextClose'] = data['Close'].shift(-1)
        data.reset_index(inplace=True)
        data.dropna(inplace=True)
        X = data[['Open','High','Low']]
        y = data['NextClose']
        X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False)
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        y_pred = lr.predict(X_test)
        lists1.append(y_pred[-1])
        # ==============================================================
    tickers=sym
    enddate = dt.datetime.now()
    startdate = enddate - dt.timedelta(days=15000)
    lists2 = []
    for intervals in['1d','5d']:
        data = yf.download(tickers=tickers, start=startdate, end=enddate, interval=intervals,progress=False)
        data['NextClose'] = data['Close'].shift(-1)
        data.reset_index(inplace=True)
        data.dropna(inplace=True)
        X = data[['Open','High','Low']]
        y = data['NextClose']
        X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False)
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        y_pred = lr.predict(X_test)
        lists2.append(y_pred[-1])
        # ===============================================================
    tickers=sym
    enddate = dt.datetime.now()
    startdate = enddate - dt.timedelta(weeks=1000)
    lists3 = []
    for intervals in['1wk']:
        data = yf.download(tickers=tickers, start=startdate, end=enddate, interval=intervals,progress=False)
        data['NextClose'] = data['Close'].shift(-1)
        data.reset_index(inplace=True)
        data.dropna(inplace=True)
        X = data[['Open','High','Low']]
        y = data['NextClose']
        X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False)
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        y_pred = lr.predict(X_test)
        lists3.append(y_pred[-1])
        # ===================================================================
    tickers=sym
    enddate = dt.datetime.now()
    startdate = enddate - dt.timedelta(weeks=1000)
    lists4 = []
    for intervals in['1mo']:
        data = yf.download(tickers=tickers, start=startdate, end=enddate, interval=intervals,progress=False)
        data['NextClose'] = data['Close'].shift(-1)
        data.reset_index(inplace=True)
        data.dropna(inplace=True)
        X = data[['Open','High','Low']]
        y = data['NextClose']
        X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False)
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        y_pred = lr.predict(X_test)
        lists4.append(y_pred[-1])
    index = ['15 min','30 min','60 min','90 min','01 day','05 day','01 week','01 month']
    lists = lists1+lists2+lists3+lists4
    lists = pd.Series(lists).astype(float).round(5)
    lists.index = index
    lists = pd.DataFrame(lists)
    lists.columns = ['Price']
    return lists
    
#=================================================================================================

def sr_lv(sym):
    tickers=sym
    enddate = dt.datetime.now()
    startdate = enddate - dt.timedelta(days=180)
    df = yf.download(tickers=tickers, start=startdate, end=enddate, interval='1d',progress=False)
    df.drop(columns=['Adj Close','Volume'], inplace=True)
    df.columns = ['open','high','low','close']
    
    def support(df,l,n1,n2):
    
        for i in range(l-n1+1,l+1):
            if(df['low'][i]>df['low'][i-1]):
                return 0
        for i in range(l+1,l+n2+1):
            if(df['low'][i]<df['low'][i-1]):
                return 0
        return 1

    def resistance(df,l,n1,n2):
    
        for i in range(l-n1+1,l+1):
             if(df['high'][i]>df['high'][i-1]):
                return 0
        for i in range(l+1,l+n2+1):
            if(df['high'][i]<df['high'][i-1]):
                return 0
        return 1
    
    ss = []
    rr = []
    n1 = 2 
    n2 = 2 
    for row in range(5,len(df)-n2):
        if support(df,row,n1,n2):
            ss.append((row,df['low'][row],1))
        if resistance(df,row,n1,n2):
            rr.append((row,df['high'][row],2))
    
    sslist = [x[1] for x in ss if x[2]==1]
    rrlist = [x[1] for x in rr if x[2]==2]
    sslist.sort()
    rrlist.sort()

    for i in range(1,len(sslist)):
        if(i>=len(sslist)):
            break
        if abs(sslist[i]-sslist[i-1])<=0.005:
            sslist.pop(i)

    for i in range(1,len(rrlist)):
        if(i>=len(rrlist)):
            break
        if abs(rrlist[i]-rrlist[i-1])<=0.001:
            rrlist.pop(i)

    sr = rrlist+sslist
    for i in range(1,len(sr)):
        if(i>=len(sr)):
            break
        if abs(sr[i]-sr[i-1])<=0.01:
            sr.pop(i)
    
    sr = pd.Series(sr).round(5)
    sr = pd.DataFrame(sr)
    sr.columns = ['SR level']
    return sr.sort_values(by='SR level',ascending=False)
#==================================================================================================

# Streamlit app

st.title("Ready to make some profit?💵")

line = '<font color=#FFFFFF>==================================\
======================================================</font>'
st.markdown(line, unsafe_allow_html=True)

col1, col2 = st.columns(2, gap='small')

with col1:
    st.subheader('Currency Pairs💱')
   
    st.write('Major pairs🎖️')
    if st.button('EUR/USD'):
        sym = 'EURUSD=x'
    if st.button('GBP/USD'):
        sym = 'GBPUSD=x'
    if st.button('USD/JPY'):
        sym = 'USDJPY=x'
    if st.button('USD/CHF'):
        sym = 'USDCHF=x'
    if st.button('USD/CAD'):
        sym = 'USDCAD=x'
    if st.button('AUD/USD'):
        sym = 'AUDUSD=x'
    if st.button('NZD/USD'):
        sym = 'NZDUSD=x'
    st.write('')
    st.write('Minor pairs🎀')
    if st.button('EUR/GBP'):
        sym = 'EURGBP=x'
    if st.button('GBP/JPY'):
        sym = 'GBPJPY=x'
    if st.button('GBP/CHF'):
        sym = 'GBPCHF=x'
    if st.button('GBP/AUD'):
        sym = 'GBPAUD=x'
    if st.button('EUR/JPY'):
        sym = 'EURJPY=x'
    if st.button('AUD/CAD'):
        sym = 'AUDCAD=x'

with col2:
    
    try:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('Selected Currency Pair:',sym.strip('=x'))

        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        st.subheader('Price Prediction 📈')
        
        st.table(get_pred(sym).style\
            .highlight_max(color='#19CEB3',axis=0)\
            .highlight_min(color='#D62B70', axis=0))
        st.write('Max value: 🟢\t Min value: 🔴 ')
        st.write('='*43)

        st.subheader('Support and Resistance in 180 days 📊📅')
        st.table(sr_lv(sym))

        
    except NameError:
        st.write('')
        st.error('Please click currency pair button', icon='🚩')

st.write('')
st.write('')
st.write('')


remind_me = '<font color=#D62B70>Reminder: The graph might get to the predicted price level, but the close prediction price is not the acutal next close price</font>'
st.markdown(remind_me, unsafe_allow_html=True)