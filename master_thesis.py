import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt
import hvplot.pandas
from numpy import array

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc,
    #plot_confusion_matrix, plot_roc_curve
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import statsmodels.api as sm

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

import category_encoders as ce
import miceforest as mf
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from sklearn.feature_selection import RFECV, RFE
from category_encoders.cat_boost import CatBoostEncoder

pd.set_option('display.float', '{:.2f}'.format)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)


## FUNCTIONS ##

def woe(data_in, target, variable, bins, binning):
    
    df = data_in
    df2 = data_in[[target, variable]].rename(columns={target: 'Target', variable: 'Variable'}).dropna()
    
    if binning == 'True':
       df2['key'] = pd.qcut(df2.Variable, bins, labels=False, duplicates='drop')
    if binning == 'False':
       df2['key'] = df2.Variable
    table = pd.crosstab(df2.key, df2.Target, margins= True)
    table = table.drop(['All'], axis=0)
    table = table.rename(columns={1: 'deft', 0: 'nondeft'}).reset_index(drop=False)

    table.loc[:, 'fracdeft'] = table.deft/np.sum(table.deft)
    table.loc[:, 'fracnondeft'] = table.nondeft/np.sum(table.nondeft)

    table.loc[:, 'WOE'] = np.log(table.fracdeft/table.fracnondeft)
    table.loc[:, 'IV'] = (table.fracdeft-table.fracnondeft)*table.WOE
    
    table.rename(columns={'WOE': variable}, inplace=True)
    table=table.add_suffix('_WOE')
    table.rename(columns={table.columns[0]: 'key' }, inplace = True)
    WOE = table.iloc[:, [0,-2]]
    
    df = pd.merge(df, df2.key, right_index=True, left_index=True)
      
    outputWOE = pd.merge(df, WOE, on='key').drop(['key'], axis=1)
    outputIV = pd.DataFrame(data={'name': [variable], 'IV': table.IV_WOE.sum()})
    
    return outputWOE, outputIV

def percentage_above_bar_relative_to_xgroup(ax):
    all_heights = [[p.get_height() for p in bars] for bars in ax.containers]
    for bars in ax.containers:
        for i, p in enumerate(bars):
            total = sum(xgroup[i] for xgroup in all_heights)
            percentage = f'{(100 * p.get_height() / total) :.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), size=11, ha='center', va='bottom', rotation=90)

def evaluate_nn(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
    elif train==False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
def plot_learning_evolution(r):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(r.history['loss'], label='Loss')
    plt.plot(r.history['val_loss'], label='val_Loss')
    plt.title('Loss evolution during trainig')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(r.history['AUC'], label='AUC')
    plt.plot(r.history['val_AUC'], label='val_AUC')
    plt.title('AUC score evolution during trainig')
    plt.legend();

def nn_model(num_columns, num_labels, hidden_units, dropout_rates, learning_rate):
    inp = tf.keras.layers.Input(shape=(num_columns, ))
    x = BatchNormalization()(inp)
    x = Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = Dense(hidden_units[i], activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rates[i + 1])(x)
    x = Dense(num_labels, activation='sigmoid')(x)
  
    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=[AUC(name='PR')])
    return model

## DATA IMPORT ##
data = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Master Thesis\\lending_club_loan_two.csv")

data['loan_status'].value_counts()
data.head()
data.info()

# Check for missing data
for column in data.columns:
    if data[column].isna().sum() != 0:
        missing = data[column].isna().sum()
        portion = (missing / data.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")

# Drop unecessary columns:
data.drop(['emp_title', 'title', 'earliest_cr_line'], axis = 1, inplace = True)

# Assign 1 for charged off and 0 for fully paid
data['loan_status'] = data['loan_status'].map({'Charged Off': 1, 'Fully Paid': 0})

# checking distribution of several categorical variables:
    # term:
sns.countplot(x='term', data=data, hue='loan_status')
    # home_ownership
sns.countplot(x='home_ownership', data=data, hue='loan_status') # this suggests that there should be a category for other none and any
data.loc[(data.home_ownership == 'ANY') | (data.home_ownership == 'NONE'), 'home_ownership'] = 'OTHER'
    # verification status
sns.countplot(x='verification_status', data=data, hue='loan_status') # source verified and verified as similar fully paid and charged off distribution, we will merge them into 1 category
data.loc[(data.verification_status == 'Source Verified'), 'verification_status'] = 'Verified'
    # grade and sub_grade:
plt.subplot(2, 2, 1)
sns.countplot(x='grade', data=data, hue='loan_status')
plt.subplot(2, 2, 2)
sns.countplot(x='sub_grade', data=data, hue='loan_status')
# We should consider keeping grade instead of sub-grade in case of one-hot-encoding. Having too granular categorization will impair computing performance

# Perform target encoding for grade variables
target_encoder = ce.TargetEncoder(cols=['grade'])
data_encoded = target_encoder.fit_transform(data['grade'], data['loan_status'])
data['grade_encoded'] = data_encoded
    # emp_length

plt.figure(figsize=(12, 8))
ax3 = sns.countplot(x="emp_length", hue="loan_status", data=data)
ax3.set(xlabel='emp_length', ylabel='Count')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
percentage_above_bar_relative_to_xgroup(ax3)
plt.show()
# most of the loans fall under 10 year+ bucket. Majority of the categories have fully_paid/charged-off ratio hoovers around 8/2, thus making emp_length as a feature less useful

    # purpose
plt.figure(figsize=(12, 8))
ax3 = sns.countplot(x="purpose", hue="loan_status", data=data)
ax3.set(xlabel='purpose', ylabel='Count')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
percentage_above_bar_relative_to_xgroup(ax3)
plt.show()
# most of the loans fall under debt_consolidation, followed by credit_card. Majority of the categories have fully_paid/charged-off ratio ranging between 8/2 and 7/3, thus making purpose as a feature less 

    # application_type
sns.countplot(x='application_type', data=data, hue='loan_status')
data.loc[(data.home_ownership == 'JOINT') | (data.home_ownership == 'DIRECT_PAY'), 'application_type'] = 'INDIVIDUAL'
    # initial_list_status
sns.countplot(x='initial_list_status', data=data, hue='loan_status')

    # zip_code
data['zip_code'] = data.address.apply(lambda x: x[-5:])
data = pd.get_dummies(data, columns=['zip_code'], drop_first=True)
data.drop('address', axis=1, inplace=True)

    # One hot encoding categorical variables:
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(data[['term', 'home_ownership', 'verification_status', 'application_type', 'initial_list_status']])
encoded_cols = encoder.transform(data[['term', 'home_ownership', 'verification_status', 'application_type', 'initial_list_status']])

encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['term', 'home_ownership', 'verification_status', 'application_type', 'initial_list_status']))
result_df = pd.concat([data, encoded_df], axis=1)


drop_vars = ['sub_grade', 'emp_length', 'issue_d']
data.drop(drop_vars, axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(data.drop('loan_status', axis=1), data['loan_status'], test_size=0.2, random_state=42)
X_train.reset_index(inplace=True)
X_test.reset_index(inplace=True)


y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

X_train.drop('index', axis=1, inplace=True)
X_test.drop('index', axis=1, inplace=True)

encoder = CatBoostEncoder(verbose=1, handle_missing='value', handle_unknown='value')

encoder.fit(X_train, y_train)
# Encode categorical variables
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Check for missing data
for column in X_test_encoded.columns:
    if X_test_encoded[column].isna().sum() != 0:
        missing = X_train_encoded[column].isna().sum()
        portion = (missing / data.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")

drop_vars = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'issue_d',
             'purpose', 'initial_list_status', 'application_type']

result_df.drop(drop_vars, axis = 1, inplace = True)


kds = mf.ImputationKernel(
  X_train_encoded,
  save_all_iterations=True,
  random_state=100
)

# Run the MICE algorithm for 5 iterations - the more iteration the more accurate imputation. Recommended number of iteration is 5.
kds.mice(5)

# Return the completed dataset.
X_train_encoded = kds.complete_data()

nan_columns = X_train_encoded.columns[X_train_encoded.isna().any()]
nan_columns

kds = mf.ImputationKernel(
  X_test_encoded,
  save_all_iterations=True,
  random_state=100
)

# Run the MICE algorithm for 5 iterations - the more iteration the more accurate imputation. Recommended number of iteration is 5.
kds.mice(5)

# Return the completed dataset.
X_test_encoded = kds.complete_data()

nan_columns = X_test_encoded.columns[X_test_encoded.isna().any()]
nan_columns


cont_vars = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc',
             'pub_rec', 'revol_bal', 'revol_util', 'mort_acc', 'total_acc']

plt.figure(figsize=(12, 8))
sns.heatmap(data_imputed[cont_vars].corr(), annot=True, cmap='viridis')

for col in cont_vars:
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    data_imputed[col].hist(grid=False, bins=100)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[col])
    plt.title(col)
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(data_imputed.drop('loan_status', axis=1), data_imputed['loan_status'], test_size=0.2, random_state=42)

X_train.reset_index(inplace=True)
X_test.reset_index(inplace=True)


y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

X_train.drop('index', axis=1, inplace=True)
X_test.drop('index', axis=1, inplace=True)

X_train_cont = X_train[cont_vars]
X_test_cont = X_test[cont_vars]


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_encoded), 
                            columns = X_train_encoded.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), 
                           columns = X_test_encoded.columns)

#X_train.drop(cont_vars, axis = 1, inplace = True)
#X_test.drop(cont_vars, axis = 1, inplace = True)

#X_train = pd.concat([X_train, X_train_cont], axis=1)
#X_test = pd.concat([X_test, X_test_cont], axis=1)



y_train.value_counts(normalize = True)*100 # charge-off/paid-off ratio is 19.6/80.4

df_train = pd.concat([y_train, X_train], axis=1)


#np.random.seed(11)
#sampler = TomekLinks(sampling_strategy='majority', n_jobs=-1)
#sampler = SMOTEENN(sampling_strategy= 0.3, n_jobs = 8, random_state= 10)
#X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
#df_resampled = pd.concat([y_resampled, X_resampled], axis=1)

#woe_df = pd.DataFrame(columns=['Variable', 'Information_Value'])

#for column in df_train.columns:
#    if column != 'loan_status':
#        # Calculate point-biserial correlation
#        _, IV = woe(data_in = df_train, target = 'loan_status', variable = column, bins = 10, binning = 'True')
        
#        # Store results in the DataFrame
#        woe_df.loc[len(woe_df)] = [IV['name'][0], IV['IV'][0]]

#woe_df = woe_df.sort_values(by='Information_Value', ascending=False)

#woe_df

chi_squared_df = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
cat_vars = ['zip_code_05113', 'zip_code_11650', 'zip_code_22690', 'zip_code_29597', 'zip_code_22690', 'zip_code_30723',
            'zip_code_48052', 'zip_code_70466', 'zip_code_86630', 'zip_code_93700', ]
df_train_cont = df_train[cont_vars]

for column in df_train_cont.columns:
    if column != 'loan_status':
            # Create a contingency table
        contingency_table = pd.crosstab(df_train_cont['loan_status'], df_train_cont[column])
            
            # Perform the chi-square test
        chi2, p, _, _ = chi2_contingency(contingency_table)
            
            # Store results in the DataFrame
        chi_squared_df.loc[len(chi_squared_df)] = [column, chi2, format(p, '.3f')]    
        #chi_squared_df = chi_squared_df.append({'Variable': column, 'Chi2': chi2, 'P-value': format(p, '.3f')}, ignore_index=True)

print(chi_squared_df)





## Random Forest feature selection ##
rfc = RandomForestClassifier(n_jobs=-1, max_depth=5, class_weight='balanced')
rfecv_rfc = RFECV(rfc, step=1, cv=4, verbose=1, scoring='f1')
rfecv_rfc.fit(X_train, y_train)

print("Feature ranking: ", rfecv_rfc.ranking_)

mask_rf = rfecv_rfc.get_support()
features_rf = array(X_train.columns)
best_features_rf = features_rf[mask_rf]

print("All features: ", X_train.loc[:, X_train.columns].shape[1])
print(features_rf)

print("Selected best: ", best_features_rf.shape[0])
print(features_rf[mask_rf])

## XGBoost feature selection ##
xgbc = XGBClassifier()
rfecv_xgbc = RFECV(xgbc, step=1, cv=4, verbose=1, scoring='f1')
rfecv_xgbc.fit(X_train, y_train)

print("Feature ranking: ", rfecv_xgbc.ranking_)

mask_xgbc = rfecv_xgbc.get_support()
features_xgbc = array(X_train.columns)
best_features_xgbc = features_xgbc[mask_xgbc]

print("All features: ", X_train.loc[:, X_train.columns].shape[1])
print(features_xgbc)

print("Selected best: ", best_features_xgbc.shape[0])
print(features_xgbc[mask_xgbc])

## LGBM feature selection ##
lgbc = XGBClassifier()
rfecv_lgbc = RFECV(lgbc, step=1, cv=4, verbose=1, scoring='f1')
rfecv_lgbc.fit(X_train, y_train)

print("Feature ranking: ", rfecv_lgbc.ranking_)

mask_lgbc = rfecv_lgbc.get_support()
features_lgbc = array(X_train.columns)
best_features_lgbc = features_lgbc[mask_lgbc]

print("All features: ", X_train.loc[:, X_train.columns].shape[1])
print(features_lgbc)

print("Selected best: ", best_features_lgbc.shape[0])
print(features_lgbc[mask_lgbc])

## logistic regression feature selection ##
lr = XGBClassifier()
rfecv_lr = RFECV(lr, step=1, cv=4, verbose=1, scoring='f1')
rfecv_lr.fit(X_train, y_train)

print("Feature ranking: ", rfecv_lr.ranking_)

mask_lr = rfecv_lr.get_support()
features_lr = array(X_train.columns)
best_features_lr = features_lr[mask_lr]

print("All features: ", X_train.loc[:, X_train.columns].shape[1])
print(features_lr)

print("Selected best: ", best_features_lr.shape[0])
print(features_lr[mask_lr])

selected_vars = ['int_rate', 'grade_encoded', 'dti', 'revol_util', 'annual_inc']
X_selected = X_resampled[selected_vars]

num_columns = X_train_scaled.shape[1]
num_labels = 1
hidden_units = [150, 150, 150]
dropout_rates = [0.1, 0, 0.1, 0]
learning_rate = 1e-3

model = nn_model(
    num_columns=num_columns, 
    num_labels=num_labels,
    hidden_units=hidden_units,
    dropout_rates=dropout_rates,
    learning_rate=learning_rate
)
r = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=10,
    batch_size=32
)

y_test_pred = model.predict(X_test)
evaluate_nn(y_test, y_test_pred.round(), train=False)


