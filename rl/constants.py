# PATH
FOLDER = './data/'
TRAIN_DATA = FOLDER + 'funds_profit_train.csv'
VALIDATE_DATA = FOLDER + 'funds_profit_validation.csv'
FEATURE_TRAIN_DATA = FOLDER + 'features_train.csv'
FEATURE_VALIDATE_DATA = FOLDER + 'features_validation.csv'
FUND_FEATURES = FOLDER + 'features_2.csv'
FILES = ['Fund_RT_Monthly.csv']
FILES_PATH = [FOLDER + FILE for FILE in FILES]
FUND_DIV = FOLDER + 'Fund_DIV.csv'
FUND_MONTH = FOLDER + 'Fund_RT_Monthly.csv'

# FUND DATA
BEGIN_DATE_FOR_TEST = '2015-01-01'
FIRST_DATE = '2003-01-01'
END_DATE = '2020-05-01'
FUND_NAME_COL = ['Date','FMFUNDCLASSINFOC_ID','Current','Profit','Currency']
FMFUNDCLASSINFOC_ID = 'FMFUNDCLASSINFOC_ID'
DATE = 'Date'
PROFIT = 'Profit'
PERFORMANCEID = 'ISINCODE'
PERFORMANCEIDS = [
    #'TW000T4719Y8',
    #'TW000T4807Y1',
    #'TW000T2122C3',
    #'XS0076593267',
    #'TW000T3635B5',
    #'TW000T5003D0',
    #'XS1807183543',
    #'TW000T3783C1',
    #'XS2044958234',
    #'XS2114069094',
    #'XS2216779608',
    #'US00206RKE17',
    #'XS2321624491',
    #'LU1802467206',
    #'TW000T3635E9',
    #'XS1925406586',
    #'TW000T3524D7',
    #'XS1238805102',
    #'XS2086746927',
    #'XS0085517661',
    'TW000T4719Y8',
    'LU0156897901',
    #'TW000T1607Y8',
    'TW000T1615Y1',
    #'TW000T1811B4',
    #'TW000T0737B2',
    #'TW000T0728Y3',
    #'TW000T4512A7',
    #'TW000T2111A0',
    #'TW000T3620Y9',
    #'TW000T2275B1',
    #'TW000T2275A3',
    'TW000T0716Y8',
    #'TW000T2117A7',
    #'TW000T3618Y3',
    #'TW000T3615Y9',
    #'TW000T2103Y7',
    #'LU0820561818',
    #'TW000T1810B6',
    #'LU0441901922',
    #'TW000T1602Y9',
    #'TW000T1809B8',
    #'TW000T1622Y7',
    #'TW000T0928A9',
    #'TW000T3607Y6',
    #'TW000T3731Y4',
    #'TW000T2252Y2',
    #'TW000T1626A8',
    #'TW000T4744A6',
    #'LU1035779427',
    #'TW000T2113B4',
    #'TW000T1811A6',
    #'TW000T0921Y4',
    #'TW000T1621Y9',
    #'LU1008669860',
    #'TW000T3732B0',
    #'TW000T1651A6',
    #'IE00B9276V44',
    #'TW000T1809A0',
    #'TW000T2274Y6',
    #'IE00BBGB0V45',
    #'TW000T2269Y6',
    #'TW000T0736Y6',
    #'TW000T3209A1',
    #'TW000T2111B8',
    #'TW000T3622Y5',
    #'TW000T2271Y2',
    #'TW000T0920Y6',
    'TW000T0708Y5',
    #'TW000T0737A4',
    'TW000T0717Y6',
    #'TW000T4744B4',
    'TW000T0718Y4',
    'TW000T0911Y5',
    #'TW000T1139A2',
    #'TW000T3621Y7',
    #'TW000T1142Y6',
    #'TW000T3610Y0',
    #'TW000T4508B3',
    'LU0065014192',
    #'TW000T2104Y5',
    #'TW000T0922Y2',
    #'TW000T0826Y5',
    #'TW000T0574Y1',
    #'TW000T0928B7',
    'TW000T1110Y3',
    #'TW000T2219Y1',
    #'LU0911417367',
    #'TW000T3259B4',
    'TW000T3201Y8',
    #'TW000T3619Y1',
    #'TW000T0425Y6',
    #'TW000T1610Y2',
    #'LU0889565320',
    #'LU0300736062',
    'LU0122376428',
    #'TW000T0737C0',
    #'TW000T0739Y0',
    #'TW000T2108A6',
    #'TW000T0723Y4',
    #'TW000T3623Y3',
    #'TW000T1125Y1',
    #'LU1008670108',
    #'TW000T0431Y4',
    #'IE00B9276R08',
    #'LU0738912566',
    'LU0157308031',
    #'TW000T1619Y3',
    #'TW000T3626B4',
    #'TW000T2284Y5',
    #'LU0756536545',
    #'TW000T1626B6',
    #'LU0266512127',
    #'TW000T2107Y8',
    #'LU0995084695',
    #'SG9999001135',
    #'TW000T0728B1',
    'TW000T3208Y3',
    #'TW000T3251Y3',
    'TW000T1109Y5',
    'TW000T0906Y5',
    #'LU0215049551',
    #'TW000T3740Y5',
    'LU0029871042',
    #'TW000T0575Y8',
    'TW000T3605Y0',
    #'TW000T0428Y0',
    'LU0098860793',
    #'TW000T4512B5',
    #'TW000T1623Y5',
    #'TW000T1611Y0'
]

