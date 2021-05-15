import os
import pandas as pd

FOLDER = './Data/'
FILES = ['Fund_RT_Monthly.csv']
FILES_PATH = [FOLDER + FILE for FILE in FILES]
FUND_NAME_COL = ['Date','FMFUNDCLASSINFOC_ID','Current','Profit','Currency']
FUND_DIV = FOLDER + 'Fund_DIV.csv'
FUND_MONTH = FOLDER + 'Fund_RT_Monthly.csv'
FUND_FEATURES = FOLDER + 'features.csv'
FMFUNDCLASSINFOC_ID = 'FMFUNDCLASSINFOC_ID'
DATE = 'Date'
PROFIT = 'Profit'
PERFORMANCEIDS = ['0P00000AP4','0P00000AP8','0P00000AP9']
PERFORMANCEID = 'PERFORMANCEID'


class FundDataHandler():

    def __init__(self):
        #self.df_fund_div = pd.read_csv(FUND_DIV)
        self.df_fund = pd.read_csv(FUND_MONTH)
        self.df_features = self.change_to_pct(pd.read_csv(FUND_FEATURES))

    def change_to_pct(self, df):
        df.index = df['Date']
        res = df.drop(columns=['Date'],axis=1)
        return res.pct_change()

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

    def combine_fund_profit_by_pfid(self):
        frames = []
        performance_ids = []
        for _id in PERFORMANCEIDS:
            df_profit = self.get_fund_profit_by_pfid(_id)
            frames.append(df_profit)
        df_res =  pd.concat(frames, axis=1).sort_index(ascending=True)
        df_res.columns = PERFORMANCEIDS
        return df_res

    def combine_fund_feature(self, df_fund, df_features):
        frames = [df_fund, df_features]
        return pd.concat(frames, axis=1).sort_index(ascending=True)

    def fund_profit_to_csv(self, df, path):
        df = df.reset_index(level=['DATAYM'])
        df.to_csv('./out/test.csv', index=False)

    def run(self):
        df_fund = fdh.combine_fund_profit_by_pfid()
        res_fund = self.combine_fund_feature(df_fund, self.df_features)
        print(res_fund.head())
        print(res_fund.tail())

if __name__ == '__main__':
    fdh = FundDataHandler()
    #print(fdh.df_features.head())
    #print(fdh.df_features.tail())
    #res = fdh.combine_fund_profit_by_pfid()
    #print(res.head())
    #fdh.fund_profit_to_csv(res, './out/test.csv')
    fdh.run()
