import pandas
import requests
import pandas_datareader as pdr
from datetime import datetime
from bs4 import BeautifulSoup
from utils import handle_date


def get_raw_data(start_date, end_date, stock_list=[], columns=()):
    s_y, s_m, s_d = handle_date(start_date)
    e_y, e_m, e_d = handle_date(end_date)
    sd = datetime(s_y, s_m, s_d)
    ed = datetime(e_y, e_m, e_d)
    concat_list = []
    print(stock_list)
    for stock in stock_list:
        try:
            res = pdr.get_data_yahoo(symbols=stock, start=sd, end=ed)["Adj Close"]
            concat_list.append(res)
        except Exception as e:
            print('[ERROR] {0} {1}'.format(stock, str(e)))
            pass
    if not concat_list:
        raise Exception('[ERROR] No data found from {0} to {1}'.format(start_date, end_date))
    bnp = pandas.concat(concat_list, axis=1)
    bnp = bnp.sort_values(by="Date",ascending=True)
    bnp.head(80)
    if columns:
        bnp.columns = columns
    return bnp

def get_sp500(start_date, end_date, filename, folder,stock_list=[], columns=()):
    folder=folder+"/"+filename
    bnp = get_raw_data(start_date, end_date,stock_list=stock_list, columns=columns)
    bnp.to_csv(folder)

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

#if __name__ == "__main__":
    #stock_list = ['VFINX','VBMFX','VMOT','RWM','DOG','SH','^SP500TR']
    #get_sp500('2020.07.03','2020.07.03','PRED.csv',stock_list=stock_list)
    #bmp = get_data('2020-09-11','2020-09-11',stock_list=stock_list)
    #print(bmp)
