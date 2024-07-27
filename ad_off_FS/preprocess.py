import pandas as pd

# Loading data from csv file
# id 49, 36, 30, 1 skipped
# RT_df is the 
RT_df = pd.read_excel("/Users/lambert/NHanCE/Data/RT.xlsx") #response time
HR_df = pd.read_csv("/Users/lambert/NHanCE/Data/hr.csv") #heart rate
HR_dr_df = HR_df[HR_df['task'].astype(str).str.endswith('1')] #Driving data

# Different dimension than AD_off
# subjective_df = pd.read_csv("/Users/lambert/lab/Data/subjective.csv") 

# Indices of when AD_off != NA
ad_off_valid_indices = RT_df.dropna(subset=['AD_off']).index

### data of 227 driving trials excluding NA in AD_off
# Ad_off:
AD_off = RT_df[['AD_off']].iloc[ad_off_valid_indices]

### Heart Rate Metrics
meanRR = HR_dr_df[['Mean RR  (ms):']].iloc[ad_off_valid_indices]
maxHR = HR_dr_df[['Max HR (beats/min):']].iloc[ad_off_valid_indices]
meanHR = HR_dr_df[['Mean HR (beats/min):']].iloc[ad_off_valid_indices]

### Heart Rate Variability Metrics (Time domain)
RMSSD = HR_dr_df[['RMSSD (ms):']].iloc[ad_off_valid_indices]
SDHR = HR_dr_df[['SD HR (beats/min):']].iloc[ad_off_valid_indices]
SDNN = HR_dr_df[['SDNN (ms):']].iloc[ad_off_valid_indices]

### Heart Rate Variability Metrics (Frequency domain)
HF = HR_dr_df[['HF (ms^2):']].iloc[ad_off_valid_indices]
LF = HR_dr_df[['LF (ms^2):']].iloc[ad_off_valid_indices]
total_power = HR_dr_df[['Total power (ms^2):']].iloc[ad_off_valid_indices]



# Subjective
#subjective_overall = subjective_df[['overall']].iloc[ad_off_valid_indices]
#subjective_overall_array = subjective_overall.to_numpy().flatten()

#RMSSD suspeced to have an inverse relationship with AD_Off
