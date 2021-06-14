import os
from datetime import datetime
import pandas as pd
from constants import FUND_LIST, FOLDER, FILES, FILES_PATH, FUND_NAME_COL, \
    FUND_DIV, FUND_MONTH, FUND_FEATURES, FMFUNDCLASSINFOC_ID, DATE, PROFIT, PERFORMANCEID, \
    BEGIN_DATE_FOR_TEST, FIRST_DATE, TRAIN_DATA, VALIDATE_DATA, FUND_FEATURES, END_DATE, \
    FEATURE_TRAIN_DATA, FEATURE_VALIDATE_DATA, FUND_NUMBER, RAW_TRAIN_DATA, \
    RAW_VALIDATE_DATA, RAW_DATA, NEWS_DATA


class FundDataHandler():

    def __init__(self):
        #self.df_fund_div = pd.read_csv(FUND_DIV)
        self.df_raw = pd.read_csv(RAW_DATA)
        self.df_fund = pd.read_csv(FUND_MONTH)
        self.df_features = pd.read_csv(FUND_FEATURES)
        self.df_news = pd.read_csv(NEWS_DATA)
        self.df_features = self.df_features.drop(columns=['index'])
        self.fund_list = FUND_LIST

    def get_news_data(self, df):
        dates = list(set(df['Date']))
        vals = []
        for date in dates:
            val = df[df['Date'] == date]['label'].mean()
            vals.append(val)
        max_val = abs(max(vals, key=abs)) + 1
        vals = [val / max_val for val in vals]
        d = {'NEWS': vals}
        df_res = pd.DataFrame(data=d)
        df_res.index = dates
        df_res.index.name = 'Date'
        df_res = df_res.sort_index(ascending=True)
        #df_res = df_res.drop(columns=['Date'],axis=1)
        df_res = df_res.ffill().fillna(df_res.mean())
        return df_res

    def get_raw_data(self):
        frames = []
        for f in self.fund_list:
            df_tmp = self.df_raw[self.df_raw['ISIN_CDE'] == f ]
            df_tmp_price = df_tmp['SELL_PRICE']
            df_tmp_price.index = df_tmp['PRICE_DATE']
            df_tmp_price.index.name = 'Date'
            frames.append(df_tmp_price)
        df_res =  pd.concat(frames, axis=1).sort_index(ascending=True)
        df_res.columns = self.fund_list
        df_res.index.name = 'Date'
        return df_res

    def handle_date(self, df, index):
        # Change Date to 'xxxx-xx-01'
        dates = df[index]
        new_dates = []
        for date in dates:
            #print(date.split('-')[-1])
            y, m, d = date.split('-')
            n_y, n_m, n_d = y, m, d
            if int(d) <= 15:
                n_d = str(1).zfill(2)
            else:
                n_d = str(1).zfill(2)
                if m == '12':
                    n_y = str(int(y)+1)
                    n_m = str(1).zfill(2)
                else:
                    n_m = str((int(m) + 1)).zfill(2)
            new_date = '-'.join([n_y,n_m.zfill(2),n_d.zfill(2)])
            new_dates.append(new_date)
        df['Date'] = new_dates
        return df

    def change_to_pct(self, df):
        df.index = df['Date']
        res = df.drop(columns=['Date'],axis=1)
        return res.pct_change()

    def filter_cols(self, df, cols):
        return df[cols]

    def read_fund(self):
        df_fund = None
        pre_df_fund = pd.read_csv(FILES_PATH[0], names=FOUND_NAME_COL)
        for f in FILES_PATH[1:]:
            cur_df_fund = pd.read_csv(f, names=FOUND_NAME_COL)
            df_fund = pre_df_fund.append(cur_df_fund)
            pre_df_fund = df_fund
        if df_fund:
            return df_fund
        return pre_df_fund

    def _base_check_fund_id(self, df, key, val):
        res = df.loc[df[key] == val]
        if res.empty:
            raise Exception('{0} == {1} not exist'.format(key, val))
        return res

    def select_fund_by_pfid(self, _id):
        return self._base_check_fund_id(self.df_fund, PERFORMANCEID, _id)

    def check_fund_div_by_id(self, _id):
        res = self._base_check_fund_id(self.df_fund_div, FMFUNDCLASSINFOC_ID, _id)
        return res.iloc[0][PERFORMANCEID]

    def select_fund_by_id(self, _id):
        return self._base_check_fund_id(self.df_fund, FMFUNDCLASSINFOC_ID, _id)

    def get_fund_profit_by_id(self, _id):
        res = self.select_fund_by_id(_id)
        res.index = res[DATE]
        return res[PROFIT]

    def get_fund_profit_by_pfid(self, _id):
        res = self.select_fund_by_pfid(_id)
        res.index = res['DATAYM']
        return res['RET1M']

    def combine_fund_profit_by_id(self):
        frames = []
        performance_ids = []
        for _id in FMFUNDCLASSINFOC_IDS:
            performance_ids.append(self.check_fund_div_by_id(_id))
            df_profit = self.get_fund_profit_by_id(_id)
            frames.append(df_profit)
        df_res =  pd.concat(frames, axis=1).sort_index(ascending=True)
        df_res.columns = performance_ids
        return df_res

    def filter_date(self, df, first_date, end_date):
        return df[first_date:end_date]
        #try:
        #    first_row_number = df.index.get_loc(first_date)
        #    end_row_number = df.index.get_loc(end_date)
        #except:
        #    raise Exception('[ERROR] Cannot find {} or {} in index'.format(first_date, end_date))
        #return df[first_row_number:end_row_number+1]

    def get_date_between(self, df, first_date, end_date):
        return df[first_row_number:end_row_number]

    def check_first_date_valid(self, df, date):
        res_valid = []
        res_invalid = []
        columns = df.columns
        for i in columns:
            df_profit = df[i]
            first_valid_date = df_profit.first_valid_index()
            if datetime.strptime(first_valid_date, '%Y-%m-%d') > datetime.strptime(FIRST_DATE, '%Y-%m-%d'):
                print('[WARNING] {} first valid date {} is earlier first date {}'.format(i, first_valid_date, FIRST_DATE))
                #pass
                res_invalid.append(i)
            else:
                print('[INFO] {} is valid date'.format(i))
                res_valid.append(i)
        return res_valid, res_invalid

    def combine_fund_profit_by_pfid(self):
        frames = []
        performance_ids = []
        ids = []
        for _id in self.fund_list:
            try:
                df_profit = self.get_fund_profit_by_pfid(_id)
            except:
                print('[WARNING] {0} not found'.format(_id))
                continue
            ids.append(_id)
            frames.append(df_profit)
        df_res =  pd.concat(frames, axis=1).sort_index(ascending=True)
        df_res.columns = ids
        df_res.index.name = 'Date'
        return df_res

    def combine_df(self, df_list):
        frames = df_list
        df_res = pd.concat(frames, axis=1).sort_index(ascending=True)
        df_res.index.name = 'Date'
        return df_res

    def fund_profit_to_csv(self, df, path, col):
        df = df.reset_index(level=[col])
        df.to_csv(path, index=False)

    def seperate_train_test(self, df, date):
        # data format str 2021-04-28
        #try:
        #    row_number = df.index.get_loc(date)
        #except:
        #    raise Exception('[ERROR] Cannot find {} in index'.format(date))
        df_train = df[:date]
        df_validation = df[date:]
        return df_train, df_validation

    def handle_news(self):
        df_news = self.handle_date(self.df_news, 'date')
        df_news = self.get_news_data(df_news)
        f_valid, f_invalid = self.check_first_date_valid(df_news, FIRST_DATE)
        df_news = self.filter_cols(df_news, f_valid)
        df_news = self.filter_date(df_news, FIRST_DATE, END_DATE)
        return df_news

    def handle_features(self):
        df_features = self.handle_date(self.df_features, 'Date')
        df_features = self.change_to_pct(self.df_features)
        f_valid, f_invalid = self.check_first_date_valid(df_features, FIRST_DATE)
        df_features = self.filter_cols(df_features, f_valid)
        df_features = self.filter_date(df_features, FIRST_DATE, END_DATE)
        return df_features

    def handle_funds(self):
        df_fund = self.combine_fund_profit_by_pfid()
        valid, invalid = self.check_first_date_valid(df_fund, FIRST_DATE)
        df_fund = self.filter_cols(df_fund, valid[:FUND_NUMBER])
        df_fund = self.filter_date(df_fund, FIRST_DATE, END_DATE)
        return df_fund

    def handle_raw(self):
        df_raw = self.get_raw_data()
        valid, invalid = self.check_first_date_valid(df_raw, FIRST_DATE)
        df_raw = self.filter_cols(df_raw, valid[:FUND_NUMBER])
        df_raw = self.filter_date(df_raw, FIRST_DATE, END_DATE)
        return df_raw


    def run(self):
        df_features = self.handle_features()
        df_funds = self.handle_funds()
        df_raw = self.handle_raw()
        df_news = self.handle_news()

        # Check index size the same
        if len(df_features.index) != len(df_funds.index):
            raise Exception('[ERROR] Index not the same')

        # Combine fund and feature
        df_features = self.combine_df([df_funds, df_features, df_news])
        print(df_features.head())

        # Seperate and save raw
        df_raw_train, df_raw_validation = self.seperate_train_test(df_raw, BEGIN_DATE_FOR_TEST)
        self.fund_profit_to_csv(df_raw_train, RAW_TRAIN_DATA, 'Date')
        self.fund_profit_to_csv(df_raw_validation, RAW_VALIDATE_DATA, 'Date')

        # Seperate and save features
        df_feature_train, df_feature_validation = self.seperate_train_test(df_features, BEGIN_DATE_FOR_TEST)
        self.fund_profit_to_csv(df_feature_train, FEATURE_TRAIN_DATA, 'Date')
        self.fund_profit_to_csv(df_feature_validation, FEATURE_VALIDATE_DATA, 'Date')

        # Seperate and save funds
        df_train, df_validation = self.seperate_train_test(df_funds, BEGIN_DATE_FOR_TEST)
        self.fund_profit_to_csv(df_train, TRAIN_DATA, 'Date')
        self.fund_profit_to_csv(df_validation, VALIDATE_DATA, 'Date')

    def test(self):
        df_news = self.handle_news()
        print(df_news['NEWS'].isnull().values.any())

if __name__ == '__main__':
    fdh = FundDataHandler()
    fdh.run()
    #fdh.test()
