import pandas
import requests
import pandas_datareader as pdr
from datetime import datetime
from bs4 import BeautifulSoup
from utils import handle_date


def get_sp500(start_date, end_date):
    s_y, s_m, s_d = handle_date(start_date)
    e_y, e_m, e_d = handle_date(end_date)
    sd = datetime(s_y, s_m, s_d)
    ed = datetime(e_y, e_m, e_d)
    stock_list = ['^vix', '^SP500TR', 'VFINX', 'VBMFX', 'ROMO', 'VMOT']
    concat_list = []
    for stock in stock_list:
        print(stock)
        res = pdr.get_data_yahoo(symbols=stock, start=sd, end=ed)["Adj Close"]
        concat_list.append(res)
    print(concat_list)
    bnp = pandas.concat(concat_list, axis=1)
    bnp = bnp.sort_values(by="Date",ascending=False)
    bnp.head(80)
    bnp.to_csv('Result_{0}_{1}.csv'.format(start_date, end_date))

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

if __name__ == "__main__":
    get_sp500('1990.1.1','2020.10.20')
