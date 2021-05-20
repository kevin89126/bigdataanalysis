import os
from datetime import datetime
import pandas as pd
from constants import PERFORMANCEIDS, FOLDER, FILES, FILES_PATH, FUND_NAME_COL, \
    FUND_DIV, FUND_MONTH, FUND_FEATURES, FMFUNDCLASSINFOC_ID, DATE, PROFIT, PERFORMANCEID, \
    BEGIN_DATE_FOR_TEST, FIRST_DATE, TRAIN_DATA, VALIDATE_DATA, FUND_FEATURES, END_DATE


class FundDataHandler():

    def __init__(self):
        #self.df_fund_div = pd.read_csv(FUND_DIV)
        self.df_fund = pd.read_csv(FUND_MONTH)
        self.df_features = pd.read_csv(FUND_FEATURES)

    def handle_features_date(self, df):
        # Change Date to 'xxxx-xx-01'
        dates = df['Date']
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
            new_date = '-'.join([n_y,n_m,n_d])
            new_dates.append(new_date)
        df['Date'] = new_dates
        return df

    def change_to_pct(self, df):
        df.index = df['Date']
        res = df.drop(columns=['Date'],axis=1)
        return res.pct_change()

    def handle_features(self):
        df_features = self.handle_features_date(self.df_features)
        df_features = self.change_to_pct(self.df_features)
        f_valid, f_invalid = self.check_first_date_valid(self.df_features, FIRST_DATE)
        df_features = self.filter_cols(self.df_features, f_valid)
        df_features = self.filter_date(df_features, FIRST_DATE, END_DATE)
        print(df_features.head())
        print(df_features.tail())
        pass

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
        try:
            first_row_number = df.index.get_loc(first_date)
            end_row_number = df.index.get_loc(end_date)
        except:
            raise Exception('[ERROR] Cannot find {} or {} in index'.format(first_date, end_date))
        return df[first_row_number:end_row_number+1]

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
        for _id in PERFORMANCEIDS:
            try:
                df_profit = self.get_fund_profit_by_pfid(_id)
            except:
                print('[WARNING] {0} not found'.format(_id))
                continue
            ids.append(_id)
            frames.append(df_profit)
        df_res =  pd.concat(frames, axis=1).sort_index(ascending=True)
        df_res.columns = ids
        return df_res

    def combine_fund_feature(self, df_fund, df_features):
        frames = [df_fund, df_features]
        return pd.concat(frames, axis=1).sort_index(ascending=True)

    def fund_profit_to_csv(self, df, path):
        df = df.reset_index(level=['DATAYM'])
        df.to_csv(path, index=False)

    def seperate_train_test(self, df, date):
        # data format str 2021-04-28
        try:
            row_number = df.index.get_loc(date)
        except:
            raise Exception('[ERROR] Cannot find {} in index'.format(date))
        df_train = df[:row_number]
        df_validation = df[row_number:]
        return df_train, df_validation

    def run(self):
        df_fund = self.combine_fund_profit_by_pfid()
        self.check_first_date_valid(df_fund, FIRST_DATE)
        df_fund = self.filter_date(df_fund, FIRST_DATE, END_DATE)
        df_train, df_validation = self.seperate_train_test(df_fund, BEGIN_DATE_FOR_TEST)
        #res_fund = self.combine_fund_feature(df_fund, self.df_features)
        self.fund_profit_to_csv(df_train, TRAIN_DATA)
        self.fund_profit_to_csv(df_validation, VALIDATE_DATA)
        print(df_train.head())
        print(df_train.tail())
        print(df_validation.head())

if __name__ == '__main__':
    fdh = FundDataHandler()
    #print(fdh.df_features.head())
    #print(fdh.df_features.tail())
    #res = fdh.combine_fund_profit_by_pfid()
    #print(res.head())
    #fdh.fund_profit_to_csv(res, './out/test.csv')
    #fdh.run()
    fdh.handle_features()
    #fdh.handle_features_date()
