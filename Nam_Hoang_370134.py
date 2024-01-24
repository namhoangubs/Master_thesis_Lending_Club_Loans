import os
os. chdir('C:/Users/Lenovo/Desktop/Master Thesis/Inputs/')
import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

years = range(2007, 2024)

df_data_dict = pd.read_excel('C:/Users/Lenovo/Desktop/Master Thesis/Inputs/file_layout.xlsx', 
                             sheet_name='Monthly Performance Data File', 
                             header=1)

columns_rename_mapper = {'Estimated Loan-to-Value (ELTV)': 'est_ltv',
                                'Monthly Reporting Period': 'rep_period',
                                # 'Current Deferred UPB': 'deferred_upb', 
                                'Current Interest Rate': 'int_rate', 
                                'Remaining Months to Legal Maturity': 'remaining_months', 
                                'Loan Age': 'loan_age', 
                                'Current Loan Delinquency Status': 'loan_status', 
                                'Current Actual UPB': 'actual_upb', 
                                'Loan Sequence Number': 'loan_id', 
                                #  'Repurchase Flag': 'repur_flag', 
                                #  'Modification Flag': 'mod_flag', 
                                 'Zero Balance Code': 'zero_bal_code', 
                                #  'Zero Balance Effective Date': 'zero_bal_d', 
                                #  'Due Date of Last Paid Installment (DDLPI)': 'ddlpi',
                                 'Delinquency Due to Disaster': 'delq_by_disaster',
                                }


cols_to_del = ['Non MI Recoveries', 'Net Sales Proceeds', 
                'Delinquent Accrued Interest', 'Actual Loss Calculation',
                'Miscellaneous Expenses', 'Taxes and Insurance', 
                'Maintenance and Preservation Costs', 'Legal Costs', 
                'Expenses', 'MI Recoveries', 'Step Modification Flag', 
                'Modification Flag', 'Deferred Payment Plan', 'Modification Cost', 
                'Current Month Modification Cost', 'Borrower Assistance Status Code',
                'Due Date of Last Paid Installment (DDLPI)', 
                'Defect Settlement Date',
                'Zero Balance Removal UPB', 
                'Zero Balance Effective Date', 
                'Interest Bearing UPB',
                'Current Deferred UPB',
                ]

np.random.seed(0)
selected_loan_idxes_paid = set()
selected_loan_idxes_charged = set()
df_list = []
path_data = 'sample_svcg_{0}.txt'
for year in years:
    path = path_data.format(year)
    
    df_m = pd.read_csv(path, 
                       sep='|', 
                       header=None, 
                       names=df_data_dict['ATTRIBUTE NAME'].values)
    print(path)    
    selected_charged = set(df_m.loc[df_m['Zero Balance Code'].isin([2.0, 3.0, 9.0]), 'Loan Sequence Number'].values.tolist())
    q_loan_idxes_paid = df_m.loc[df_m['Zero Balance Code']==1.0, 'Loan Sequence Number'].values.tolist()
    selected_paid = np.random.choice(q_loan_idxes_paid, len(selected_charged), replace=False)
    selected_paid = set(selected_paid)

    selected_loan_idxes_paid.update(selected_paid)
    selected_loan_idxes_charged.update(selected_charged)

    df_list.append(df_m.loc[df_m['Loan Sequence Number'].isin(selected_paid), :])
    df_list.append(df_m.loc[df_m['Loan Sequence Number'].isin(selected_charged), :])

    del df_m

print("len(selected_loan_idxes_paid)", len(selected_loan_idxes_paid))
print("len(selected_loan_idxes_charged)", len(selected_loan_idxes_charged))
print('total selected loan indexes', len(selected_loan_idxes_paid)+len(selected_loan_idxes_charged))

df = pd.concat(df_list, ignore_index=True)
df.rename(columns=columns_rename_mapper, inplace=True)
df.drop(columns=cols_to_del, inplace=True)

######## DATA AT ORIGINATION #########

df_data_dict_orig = pd.read_excel('file_layout.xlsx', 
                             sheet_name='Origination Data File', 
                             header=1)

np.random.seed(0)
df_list = []
path_orig_data = 'sample_orig_{0}.txt'
for year in years:
    path = path_orig_data.format(year)
    print(path)
    df_o = pd.read_csv(path, 
                       sep='|', 
                       header=None, 
                       names=df_data_dict_orig['ATTRIBUTE NAME'].values)
    df_paid = df_o.loc[df_o['Loan Sequence Number'].isin(selected_loan_idxes_paid), :].reset_index(drop=True)
    df_charged = df_o.loc[df_o['Loan Sequence Number'].isin(selected_loan_idxes_charged), :].reset_index(drop=True)
    df_list.append(df_paid)
    df_list.append(df_charged)

df_orig = pd.concat(df_list, ignore_index=True)

columns_rename_mapper_orig = {
                                 'Loan Sequence Number': 'loan_id', 
                                 'Credit Score': 'o_credit_score', 
                                 'Property Valuation Method': 'o_pt_val_md', 
                                 'Number of Borrowers': 'o_num_brwrs',
                                 'Original Loan Term': 'o_term', 
                                 'Loan Purpose': 'o_purp',
                                 'Postal Code': 'o_zip',
                                 'Maturity Date': 'o_mat_d',
                                 'Property Type': 'o_prop_type', 
                                 'Property State': 'o_prop_st',
                                 'Original Interest Rate': 'o_int_rate', 
                                 'Original Loan-to-Value (LTV)': 'o_ltv',
                                 'Original UPB': 'o_upb', 
                                 'Original Debt-to-Income (DTI) Ratio': 'o_dti', 
                                 'Original Combined Loan-to-Value (CLTV)': 'o_cltv', 
                                 'Occupancy Status': 'o_occ_stat',
                                 'Number of Units': 'o_units', 
                                 'Mortgage Insurance Percentage (MI %)': 'o_mi',
                                 'First Time Homebuyer Flag': 'o_first_flag',
                                 'Interest Only (I/O) Indicator': 'o_int_only', 
                                 'Prepayment Penalty Mortgage (PPM) Flag': 'o_ppm', 
                                 'Super Conforming Flag': 'o_sp_cnfm', 
                                 'Program Indicator': 'o_prgm_ind',
                                 'Amortization Type (Formerly Product Type)': 'o_amtz_type', 
                                 'Metropolitan Statistical Area (MSA) Or Metropolitan Division': 'o_msa', 
                                 'First Payment Date': 'o_init_pay_d', 
                                 'Channel': 'o_chan', 
                                 'HARP Indicator': 'o_harp_ind',
                                }

df_orig.drop(columns=['Pre-HARP Loan Sequence Number'], inplace=True, errors='ignore')

df_orig.drop(columns=['Servicer Name', 'Seller Name'], inplace=True, errors='ignore')

df_orig.rename(columns=columns_rename_mapper_orig, inplace=True)

percent_missing = pd.DataFrame(df_orig.isnull().sum()*100/ len(df_orig), columns=['missing percentage'])
percent_missing.sort_values(by='missing percentage', ascending=False, inplace=True)
percent_missing

cols_to_del = []
loan_idxes_to_del = set()

loan_idxes_to_del.update(df_orig.loc[df_orig['o_credit_score']==9999, 'loan_id'].values.tolist())

df_orig['o_ind_sup_cfm'] = 0
df_orig.loc[df_orig['o_sp_cnfm']=='Y', 'o_ind_sup_cfm'] = 1
df_orig.drop(columns=['o_sp_cnfm'], inplace=True, errors='ignore')
df_orig['o_ind_sup_cfm'].value_counts(dropna=False)

df_orig['o_msa'].nunique()
# MSA variable represents the metropolitan division which is a more granular level of Property State. Since this is a categorical
# variable and it has 417 values, one-hot-encoding this varaibles will create additional 417 columns which will dampen the model calibration
# process. We opt for property state value instead of msa
cols_to_del.append('o_msa')

df_orig['o_prgm_ind'].value_counts(dropna=False)

df_orig.loc[(df_orig['o_prgm_ind']=='9') | (df_orig['o_prgm_ind']==9), 'o_prgm_ind'] = 0
df_orig.loc[df_orig['o_prgm_ind'].isin(['H', 'F']), 'o_prgm_ind'] = 1
df_orig['o_prgm_ind'].value_counts(dropna=False)

df_orig['o_ppm'].value_counts(dropna=False)
# N    16869
# Y        3
# Since there are only 3 incidents where the prepayment mortgage flag is Y, this column does not add any value at all
cols_to_del.append('o_ppm')

df_orig['o_num_brwrs'].value_counts(dropna=False)
# 1     9065
# 2     7805
# 99       1
# 3        1
# Group 3, 99 number of borrowers to group of 2 borrowers

df_orig.loc[(df_orig['o_num_brwrs']>2), 'o_num_brwrs'] = 2

df_orig['o_purp'].value_counts(dropna=False)

df_orig['o_prop_type'].value_counts(dropna=False)

df_orig['o_prop_st'].value_counts(dropna=False)

df_orig['o_int_rate'].plot(kind='hist', bins=30)

df_orig['o_ltv'].hist(bins=30)

## remove loan idxes that have unknown ltv
df_orig.loc[df_orig['o_ltv']==999, :] 
# No loan with unknown ltv = 0

df_orig[['o_ltv', 'o_cltv']].corr()
# High correlation between origination ltv and cltv => exclude cltv
cols_to_del.append('o_cltv')

df_orig['o_upb'].plot(kind='hist', bins=100)

df_orig['o_dti'].plot(kind='hist', bins=100)

df_orig.loc[df_orig['o_dti']==999, :].count()
# too many missing values - account for at least 12% of the data => considering missing value imputation technique (MICE might be a solution to this)

df_orig['o_occ_stat'].value_counts(dropna=False)

df_orig['o_amtz_type'].value_counts(dropna=False)

df_orig.loc[df_orig['o_mi']==999, :].count()

df_orig['o_chan'].value_counts(dropna=False)

df_orig['o_first_flag'].value_counts(dropna=False)
# N    15000
# Y     1867
# 9        5
# We include 5 cases where the status is not applicable/unknown to N flag
df_orig.loc[(df_orig['o_first_flag']=='9'), 'o_first_flag'] = 'N'

df_orig.loc[df_orig['o_first_flag']=='N', 'o_first_flag'] = 0
df_orig.loc[df_orig['o_first_flag']=='Y', 'o_first_flag'] = 1

df_orig['o_int_only'].value_counts(dropna=False)
# Remove this column since it has only 1 value
cols_to_del.append('o_int_only')

df_orig['o_harp_ind'].value_counts(dropna=False)
# Too many NaN across obs, better not to include this column
cols_to_del.append('o_harp_ind')

df_orig.drop(columns=cols_to_del, inplace=True, errors='ignore')

percent_missing = pd.DataFrame(df_orig.isnull().sum()*100/ len(df_orig), columns=['missing percentage'])
percent_missing.sort_values(by='missing percentage', ascending=False, inplace=True)
percent_missing

df_orig.columns

keep_features = [
                 ## loan
                 'loan_id', 'o_init_pay_d', 'o_term', 
                 'o_upb', 'o_mi', 'o_ltv', 'o_int_rate', 
                 'o_chan', 'o_purp', 'o_ind_sup_cfm',
                 ## borrower
                 'o_credit_score', 'o_first_flag', 
                 'o_dti', 'o_num_brwrs',
                 ## property
                 'o_units', 'o_occ_stat', 'o_prop_st',
                 'o_prop_type', 'o_pt_val_md'
                 ]

df_orig = df_orig.loc[:, keep_features]


percent_missing = pd.DataFrame(df.isnull().sum()*100/ len(df), columns=['missing percentage'])
percent_missing.sort_values(by='missing percentage', ascending=False, inplace=True)
percent_missing

df.loc[df['delq_by_disaster']=='Y', 'delq_by_disaster'] = 1
df.loc[df['delq_by_disaster'].isnull(), 'delq_by_disaster'] = 0

df['delq_by_disaster'].value_counts(dropna=False)

df['est_ltv'].plot(kind='hist', bins=100)
df.loc[(df['est_ltv']=='999') | (df['est_ltv']== 999), 'est_ltv'] = np.NaN

#df.loc[df['loan_status']=='0', 'loan_status'] = 0
df.loc[df['loan_status']=='RA', 'loan_status'] = -1

df['loan_status'] = df['loan_status'].apply(np.int)
df.loc[df['loan_status'] > 3, 'loan_status'] = 3

selected_loan_idxes = df_orig['loan_id'].values.tolist()

def check_valid_seq(df, col):
    '''
    a valid loan monthly performance sequence satisfies:
    1) starts with age 0
    2) loan ages are consecutive
    '''
    seq = sorted(list(df.loc[:, col].values))
    # print(seq)
    # print(list(range(len(seq))))
    res = seq==list(range(len(seq)))
    # print('res', res)
    return res

invalid_loan_idxes = set()
for i, loan_id in enumerate(selected_loan_idxes):
    df_m = df.loc[df['loan_id']==loan_id, :]
    if not check_valid_seq(df_m, 'loan_age'):
        invalid_loan_idxes.add(loan_id)
    if i%2000==0:
        print('{} loans have been checked'.format(i))
        
df_orig = df_orig.loc[~df_orig['loan_id'].isin(invalid_loan_idxes), :]
df = df.loc[~df['loan_id'].isin(invalid_loan_idxes), :]

df_orig.to_csv('df_orig.csv')
df.to_csv('df.csv')

#################################################

def extract_windows(array, clearing_time_index, max_time, sub_window_size):
    examples = []
    # start = clearing_time_index + 1 - sub_window_size + 1
    start = clearing_time_index - sub_window_size + 1

    for i in range(max_time+1):
        if start+sub_window_size+i<sub_window_size:
            example = np.zeros((sub_window_size, array.shape[1])) ## zero padding
            example[-(start+sub_window_size+i):] = array[:start+sub_window_size+i]
        else:
            example = array[start+i:start+sub_window_size+i]

        examples.append(np.expand_dims(example, 0))
    
    return np.vstack(examples)

def shift_yr_qtr(yr_qtr):
    year = int(yr_qtr[:2])
    q = int(yr_qtr[-1])
    q_sum = (year*4+q-1)-1
    year = q_sum//4
    q = q_sum%4+1
    if year < 10:
        return '0' + str(year)+'Q'+str(q)
    else:
        return str(year)+'Q'+str(q)

def shift_yr_mth(yr_mth):
    year = int(yr_mth[:4])
    month = int(yr_mth[4:])
    mth_sum = (year*12+month-1)-1
    year = mth_sum//12
    month = mth_sum%12+1
    if year< 10:
        if month<10:
            return '0' + str(year)+'0'+str(month)
        else:
            return '0' + str(year)+str(month)
    else:
        if month<10:
            return str(year)+'0'+str(month)
        else:
            return str(year)+str(month)
    

df_orig['o_yr_qtr'] = df_orig['loan_id'].apply(lambda x: x[1:5])
## shift year and quarter since economic factors can have a quarter lag
df_orig['o_yr_qtr'] = df_orig['o_yr_qtr'].apply(shift_yr_qtr) 

df_orig['o_st_yr_qtr'] = df_orig['o_prop_st']+'_'+df_orig['o_yr_qtr']

df_orig.head()

df_orig['o_yr_qtr_shifted'] = df_orig['o_yr_qtr'].apply(shift_yr_qtr)

df_orig.loc[df_orig['o_prop_st']=='GU'].count()
df_orig = df_orig[df_orig['o_prop_st'] != 'GU']
us_states = set(sorted(df_orig['o_prop_st'].unique().tolist()))

df_combined = pd.merge(df, df_orig, on='loan_id')
df_combined.sort_values(by=['loan_id', 'loan_age'], inplace=True)

df_combined['rep_period'] = df_combined['rep_period'].astype(str)
## shift rep_period 
df_combined['shifted_rep_period'] = df_combined['rep_period'].apply(shift_yr_mth)

## all date-related information are shifted by one month
## so that the lags of economic factors are considered, i.e., one-month or one-quarter lag
df_combined['o_rep_period'] = df_combined.groupby('loan_id').transform('first')['shifted_rep_period']
df_combined['o_st_rep_period'] = df_combined['o_prop_st']+'_'+df_combined['o_rep_period'].astype(str)

df_combined['st_rep_period'] = df_combined['o_prop_st']+'_'+df_combined['shifted_rep_period'].astype(str)
df_combined['rep_yr_qtr'] = df_combined['rep_period'].astype(str).apply(lambda x: x[2:4]+'Q'+str(((int(x[-2:])-1)//3)+1))
df_combined['rep_yr_qtr'] = df_combined['rep_yr_qtr'].apply(shift_yr_qtr)
df_combined['st_rep_yr_qtr'] = df_combined['o_prop_st'] + '_' + df_combined['rep_yr_qtr']

df_combined.head()

df_hpi = pd.read_excel('HPI_PO_state.xls')
df_hpi['st_rep_yr_qtr'] = df_hpi['state']+'_'+df_hpi['yr'].apply(lambda x: str(x)[-2:])+'Q'+df_hpi['qtr'].apply(lambda x: str(x))
df_hpi.drop(columns=['Warning', 'index_nsa'], inplace=True)
df_hpi.rename(columns={'index_sa': 'hpi'}, inplace=True)
df_hpi.reset_index(inplace=True, drop=True)
df_hpi.sort_values(by=['state', 'yr', 'qtr'], inplace=True)
df_hpi.drop(columns=['state', 'yr', 'qtr'], inplace=True)

df_hpi

type(us_states)

states = list(us_states)

unemployment_rate = pd.DataFrame(columns=['ue_rate','st_rep_period'])
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
for state in states:
    dataframe = pd.read_excel(f'C:\\Users\\Lenovo\\Desktop\\Master Thesis\\Inputs\\{state}_ue_2007_2023.xlsx')
    dataframe = dataframe.iloc[10:]
    header = dataframe.iloc[0]
    dataframe.columns = header
    dataframe = dataframe.iloc[1:]
    dataframe.set_index(list(dataframe)[0], inplace = True)
    state = state.upper()
    for index in dataframe.index:
        order = 0
        for month in months:
            new_row = {'ue_rate': dataframe.loc[index][order], 'st_rep_period': f'{state}_{index}{month}'}
            unemployment_rate = unemployment_rate.append(new_row, ignore_index=True)
            order = order + 1


dir_chgoff = 'charge_off_rate_2007_2023.csv'
df_chgoff = pd.read_csv(dir_chgoff)
df_chgoff['observation_date'] = pd.to_datetime(df_chgoff['observation_date'])
df_chgoff['rep_yr_qtr'] = df_chgoff['observation_date'].apply(lambda x: str(x.year)[-2:])+'Q'+df_chgoff['observation_date'].apply(lambda x: str(x.quarter))
df_chgoff.rename(columns={'CORSFRMT100S': 'charge_off_rate'}, inplace=True)


df_chgoff.drop(columns=['observation_date'], inplace=True)
df_chgoff.head()

## to be updated ##