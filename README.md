# Healthcare-analytics
# Generate the complete set of dashboard visuals (static images) from visits_df and KPI table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

# Ensure sorted dates for time series
visits_sorted_df = visits_df.sort_values('Registration_Date')

# Helper to save and show plots
plot_files = []

def save_show(fig_name):
    out_path = './' + fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.show()
    plot_files.append(out_path)

# 1) KPI card-like summary as a figure
plt.figure(figsize=(10, 2.2))
plt.axis('off')
card_text = (
    'Total Patients: ' + str(int(kpi_total_patients)) + '    '
    + 'Total Registrations: ' + str(int(kpi_total_admissions)) + '    '
    + 'Avg Camp Duration (days): ' + str(round(kpi_avg_los, 2)) + '    '
    + 'Bed Occupancy (proxy): ' + str(round(kpi_bed_occ, 2))
)
plt.text(0.01, 0.5, card_text, fontsize=13, va='center')
save_show('dashboard_kpi_cards.png')

# 2) Total registrations trend over time (monthly)
monthly_regs_df = visits_sorted_df.dropna(subset=['Registration_Date']).copy()
monthly_regs_df['Reg_Month'] = monthly_regs_df['Registration_Date'].dt.to_period('M').dt.to_timestamp()
monthly_counts_df = monthly_regs_df.groupby('Reg_Month').size().reset_index(name='Registrations')
monthly_counts_df = monthly_counts_df.sort_values('Reg_Month')

plt.figure(figsize=(10, 4))
sns.lineplot(data=monthly_counts_df, x='Reg_Month', y='Registrations', marker='o', linewidth=2)
plt.title('Total Registrations Over Time (Monthly)')
plt.xlabel('Month')
plt.ylabel('Registrations')
save_show('dashboard_registrations_trend_monthly.png')

# 3) Category1 distribution (camp type)
cat1_counts_df = visits_df['Category1'].fillna('Unknown').value_counts().reset_index()
cat1_counts_df.columns = ['Category1','Count']

plt.figure(figsize=(8, 4))
sns.barplot(data=cat1_counts_df, x='Category1', y='Count', palette='viridis')
plt.title('Registrations by Camp Type (Category1)')
plt.xlabel('Camp Type')
plt.ylabel('Registrations')
save_show('dashboard_registrations_by_camp_type.png')

# 4) Department proxy: Employer_Category (top 12)
emp_counts_df = visits_df['Employer_Category'].fillna('Unknown').value_counts().head(12).reset_index()
emp_counts_df.columns = ['Employer_Category','Count']

plt.figure(figsize=(10, 5))
sns.barplot(data=emp_counts_df, y='Employer_Category', x='Count', palette='magma')
plt.title('Top Employer Categories by Registrations')
plt.xlabel('Registrations')
plt.ylabel('Employer Category')
save_show('dashboard_registrations_by_employer_category_top12.png')

# 5) Age distribution
age_series = visits_df['Age'].dropna()
plt.figure(figsize=(9, 4.5))
sns.histplot(age_series, bins=30, kde=True, color='#2a9d8f')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
save_show('dashboard_age_distribution.png')

# 6) Income distribution
income_series = visits_df['Income'].dropna()
plt.figure(figsize=(9, 4.5))
sns.histplot(income_series, bins=30, kde=True, color='#457b9d')
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Count')
save_show('dashboard_income_distribution.png')

# 7) Donations by camp type (Category1)
donation_df = visits_df[['Category1','Donation']].copy()
donation_df['Category1'] = donation_df['Category1'].fillna('Unknown')
donation_df['Donation'] = pd.to_numeric(donation_df['Donation'], errors='coerce')
donation_by_cat_df = donation_df.dropna().groupby('Category1')['Donation'].sum().reset_index().sort_values('Donation', ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=donation_by_cat_df, x='Category1', y='Donation', palette='cubehelix')
plt.title('Total Donation by Camp Type')
plt.xlabel('Camp Type')
plt.ylabel('Total Donation')
save_show('dashboard_donation_by_camp_type.png')

# 8) Health score by camp type
health_df = visits_df[['Category1','Health_Score']].copy()
health_df['Category1'] = health_df['Category1'].fillna('Unknown')
health_df['Health_Score'] = pd.to_numeric(health_df['Health_Score'], errors='coerce')
health_df = health_df.dropna(subset=['Health_Score'])

plt.figure(figsize=(8, 4.5))
sns.boxplot(data=health_df, x='Category1', y='Health_Score', palette='Set2')
plt.title('Health Score by Camp Type')
plt.xlabel('Camp Type')
plt.ylabel('Health Score')
save_show('dashboard_health_score_by_camp_type_box.png')

# 9) Engagement vs Health Score (Online_Follower as proxy)
eng_df = visits_df[['Online_Follower','LinkedIn_Shared','Twitter_Shared','Facebook_Shared','Health_Score']].copy()
for col in ['Online_Follower','LinkedIn_Shared','Twitter_Shared','Facebook_Shared']:
    eng_df[col] = pd.to_numeric(eng_df[col], errors='coerce')
eng_df['Health_Score'] = pd.to_numeric(eng_df['Health_Score'], errors='coerce')
eng_df = eng_df.dropna(subset=['Health_Score'])

plt.figure(figsize=(8, 5))
sns.scatterplot(data=eng_df.sample(min(len(eng_df), 5000), random_state=42), x='Online_Follower', y='Health_Score', alpha=0.25)
plt.title('Health Score vs Online Follower')
plt.xlabel('Online Follower (0/1)')
plt.ylabel('Health Score')
save_show('dashboard_health_score_vs_online_follower.png')

# 10) Stall visits distribution (only third camp has it)
stall_series = visits_df['Number_of_stall_visited'].dropna()
plt.figure(figsize=(9, 4.5))
sns.histplot(stall_series, bins=20, kde=False, color='#e76f51')
plt.title('Distribution of Number of Stalls Visited (Third Camp)')
plt.xlabel('Number of Stalls Visited')
plt.ylabel('Count')
save_show('dashboard_stall_visits_distribution.png')

# 11) Heatmap of correlations among numeric variables
num_cols = ['Var1','Var2','Var3','Var4','Var5','Donation','Health_Score','Age','Income','Education_Score','Number_of_stall_visited']
num_existing_cols = [c for c in num_cols if c in visits_df.columns]
num_df = visits_df[num_existing_cols].copy()
for c in num_existing_cols:
    num_df[c] = pd.to_numeric(num_df[c], errors='coerce')

corr_df = num_df.corr(numeric_only=True)
plt.figure(figsize=(10, 7))
sns.heatmap(corr_df, cmap='RdBu_r', center=0, linewidths=0.5)
plt.title('Correlation Heatmap (Numeric Features)')
save_show('dashboard_correlation_heatmap.png')

# 12) Bed occupancy proxy time series
if len(bed_util_df) > 0:
    bed_util_sorted_df = bed_util_df.sort_values('Date')
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=bed_util_sorted_df, x='Date', y='Bed_Occupancy_Rate', linewidth=2, color='#264653')
    plt.title('Bed Occupancy Rate Over Time (Proxy)')
    plt.xlabel('Date')
    plt.ylabel('Occupancy Rate (Active Camps / ' + str(int(cap_val)) + ')')
    save_show('dashboard_bed_occupancy_proxy.png')

print('\
'.join(plot_files))
