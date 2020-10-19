import pandas
import requests
import pandas_datareader as pdr
from datetime import datetime
from bs4 import BeautifulSoup


def get_sp500():
    sd = datetime(2020, 1, 1)
    ed = datetime(2020, 9, 4)

    vix = pdr.get_data_yahoo(symbols='^vix', start=sd, end=ed)["Adj Close"]
    spx = pdr.get_data_yahoo(symbols='^SP500TR', start=sd, end=ed)["Adj Close"]
    vfx = pdr.get_data_yahoo(symbols='VFINX', start=sd, end=ed)["Adj Close"]
    vbx = pdr.get_data_yahoo(symbols='VBMFX', start=sd, end=ed)["Adj Close"]
    rom = pdr.get_data_yahoo(symbols='ROMO', start=sd, end=ed)["Adj Close"]
    vmt = pdr.get_data_yahoo(symbols='VMOT', start=sd, end=ed)["Adj Close"]

    bnp = pandas.concat([vix ,vfx ,vbx ,rom, vmt, spx], axis=1)
    bnp = bnp.sort_values(by="Date",ascending=False)
    bnp.head(80)
    bnp.to_csv('Result.csv')

def get_rate():
    dfs = pandas.read_html('https://rate.bot.com.tw/xrt/quote/l6m/JPY')
    df = dfs[0].iloc[:,0:6]
    df.columns = ['掛牌日期', '幣別', '現金匯率買入','現金匯率賣出', '即期匯率買入','即期匯率賣出']
    print(df)


def get_news():
    res = requests.get('https://news.ltn.com.tw/list/breakingnews')
    soup = BeautifulSoup(res.text,'lxml')
    newsary = []
    for li in soup.select('ul.list li'):
        #print(li)
        title = li.select_one('.title').text
        dt = li.select_one('.time').text
        link = li.select_one('a').get('href')
        newsary.append({'title':title, 'time': dt, 'laink':link})
    newsdf = pandas.DataFrame(newsary)
    newsdf.head()
    print(newsdf)
