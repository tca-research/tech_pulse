import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, conversion
import random
import os
import time

levels_csv_path = "Data/input/tech_sector_salaries/levels_fyi/au_levelsfyi_detailed_data.csv"

# Check if CSV exists and its age
run_script = True
if os.path.exists(levels_csv_path):
    mtime = os.path.getmtime(levels_csv_path)
    age_seconds = time.time() - mtime
    one_week_seconds = 7 * 24 * 60 * 60
    if age_seconds < one_week_seconds:
        run_script = False
# Run the script only if CSV is older than 1 week or missing
if run_script:
    with open("Scripts/scrape_levels_fyi.py") as f:
        levels_fyi_scrape_code = f.read()
    exec(levels_fyi_scrape_code)

# Load datasets for analysis
try:
    wgea_salary_df = pd.read_csv('Data/input/tech_sector_salaries/wgea/WGEA_salary_data.csv')
    wgea_workforce_composition = pd.read_csv("Data/input/wgea_public_dataset_2024/wgea_workforce_composition_2024.csv")
    wgea_mgmt = pd.read_csv("Data/input/wgea_public_dataset_2024/wgea_workforce_management_statistics_2024.csv")
    abn_df = pd.read_csv('Data/input/tech_sector_salaries/tca/Tech_Council_ABNs.csv')
    jobs_per_industry = pd.read_csv("Data/input/tech_sector_jobs/lfs-tablebuilder/workers in the tech sector.csv")
    levels_fyi_salaries_df = pd.read_csv("Data/input/tech_sector_salaries/levels_fyi/au_levelsfyi_detailed_data.csv")
    ict_job_ad_salary = pd.read_csv("Data/input/tech_sector_salaries/seek/SEEK average advertised salary - ICT v other - July 2025.csv", skiprows=3)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all data files are in the correct directories.")
    exit()

# TECH ROLES & TALENT
#--------------------#
#--------------------#
## TECH OCCUPATION GROWTH
labour_force_survey = ro.r('readabs::read_lfs_datacube("EQ08")')

with conversion.localconverter(ro.default_converter + pandas2ri.converter):
    lfs = conversion.rpy2py(labour_force_survey)

lfs['date'] = pd.to_datetime(lfs['date'], origin='1970-01-01', unit='D')

tech_occupations_4_digit = ['1350 ICT Managers nfd',	 
  '1351 ICT Managers',  
  '2232 ICT Trainers',  
  '2247 Management and Organisation Analysts', 
  '2249 Other Information and Organisation Professionals',	 
  '2252 ICT Sales Professionals',  
  '2324 Graphic and Web Designers, and Illustrators',  
  '2334 Electronics Engineers',  
  '2600 ICT Professionals nfd',  
  '2610 Business and Systems Analysts, and Programmers nfd',  
  '2611 ICT Business and Systems Analysts',  
  '2612 Multimedia Specialists and Web Developers',  
  '2613 Software and Applications Programmers',  
  '2620 Database and Systems Administrators, and ICT Security Specialists nfd',	 
  '2621 Database and Systems Administrators, and ICT Security Specialists',	 
  '2630 ICT Network and Support Professionals nfd',  
  '2631 Computer Network Professionals', 	 
  '2632 ICT Support and Test Engineers',  
  '2633 Telecommunications Engineering Professionals',  
  '3100 Engineering, ICT and Science Technicians nfd',  
  '3124 Electronic Engineering Draftspersons and Technicians',  
  '3130 ICT and Telecommunications Technicians nfd',  
  '3131 ICT Support Technicians',  
  '3132 Telecommunications Technical Specialists',  
  '3424 Telecommunications Trades Workers'
]

lfs['tech_occupation'] =  np.where(lfs['occupation_of_main_job__anzsco_2013_v1.2'].isin(tech_occupations_4_digit), 'Tech', 'Not tech')
lfs_tech = lfs[lfs.tech_occupation=='Tech']
lfs_tech = lfs_tech.sort_values('date').groupby(['date', 'occupation_of_main_job__anzsco_2013_v1.2', 'tech_occupation'])['employed_total_000'].sum().reset_index()
lfs_tech['Occupation'] = lfs_tech['occupation_of_main_job__anzsco_2013_v1.2'].str.replace(r'^\d{4}\s*', '', regex=True)
lfs_tech['Number'] = round(lfs_tech['employed_total_000']*1000)

category_map = {
    'Business and Systems Analysts, and Programmers nfd': 'Business & Data Analysts',
    'ICT Business and Systems Analysts': 'Business & Data Analysts',
    'Software and Applications Programmers': 'Software Engineers',
    'Computer Network Professionals': 'Network & Security',
    'ICT Network and Support Professionals nfd': 'Network & Security',
    'Database and Systems Administrators, and ICT Security Specialists': 'Network & Security',
    'ICT Support Technicians': 'Network & Security',
    'ICT Support and Test Engineers': 'Network & Security',
    'ICT Trainers': 'Technical Management',
    'ICT Managers': 'Technical Management',
    'ICT Managers nfd': 'Technical Management',
    'Management and Organisation Analysts': 'Business & Data Analysts',
    'Graphic and Web Designers, and Illustrators': 'Product, Graphic, UX Design',
    'Multimedia Specialists and Web Developers': 'Software Engineers',
    'Electronic Engineering Draftspersons and Technicians': 'Hardware Engineers',
    'Electronics Engineers': 'Hardware Engineers',
    'Engineering, ICT and Science Technicians nfd': 'Hardware Engineers',
    'ICT Professionals nfd': 'Information Technology',
    'ICT Sales Professionals': 'Information Technology',
    'ICT and Telecommunications Technicians nfd': 'Telecom',
    'Telecommunications Engineering Professionals': 'Telecom',
    'Telecommunications Technical Specialists': 'Telecom',
    'Telecommunications Trades Workers': 'Telecom',
    'Other Information and Organisation Professionals': 'Information Technology'
}

lfs_tech_simplified = lfs_tech[['date', 'Occupation','Number']]
lfs_tech_simplified['Category'] = lfs_tech_simplified['Occupation'].map(category_map)
lfs_tech_simplified['Month Year'] = lfs_tech_simplified['date'].dt.strftime('%B %Y')
lfs_tech_simplified_long =  lfs_tech_simplified[['Month Year', 'Number', 'Category']].groupby(['Month Year', 'Category'], as_index=False)['Number'].sum()
lfs_tech_simplified_wide = lfs_tech_simplified_long.pivot(index='Month Year', columns='Category', values='Number').reset_index()

cols = list(lfs_tech_simplified_wide.columns)
cols.insert(1, cols.pop(cols.index('Software Engineers')))
lfs_tech_simplified_wide = lfs_tech_simplified_wide[cols]

lfs_tech_simplified_wide.to_csv('Data/output/dashboard/tech_jobs_occupations_over_time_WIDE-SIMPLE.csv', index=False)

## TECH ROLES AS A SHARE OF LABOUR FORCE

lfs_perc_occ = lfs
lfs_perc_occ = lfs.groupby(['date', 'tech_occupation']).size().reset_index(name='n')
lfs_perc_occ['percent'] = lfs_perc_occ['n'] / lfs_perc_occ.groupby('date')['n'].transform('sum') * 100
lfs_perc_occ = lfs_perc_occ[lfs_perc_occ.tech_occupation != 'Not tech']
lfs_perc_occ['% of labour force in tech occupations (smoothed)'] = round(lfs_perc_occ['percent'].rolling(window=3, center=True).mean(),1)
lfs_perc_occ = lfs_perc_occ.dropna(subset=['% of labour force in tech occupations (smoothed)'])
lfs_perc_occ["Month Year"] = lfs_perc_occ["date"].dt.strftime("%B %Y")
lfs_perc_occ.to_csv('Data/output/dashboard/tech_jobs_tech_occupations_as_percent_of_labour_force.csv', index = False)

## TOTAL EMPLOYMENT IN DIRECT TECH SECTOR

## -- Data downloaded from Table Builder

jobs_per_industry['Date'] = jobs_per_industry['Date'].fillna(method='ffill')
jobs_per_industry['Number'] = jobs_per_industry['Number'] * 1000

jobs_tech_sector = jobs_per_industry.groupby('Date', as_index=False)['Number'].sum()
jobs_tech_sector['Industry'] = 'Total number of people in the tech sector'
jobs_tech_sector = jobs_tech_sector[['Date', 'Industry', 'Number']]

jobs_tech = pd.concat([jobs_per_industry, jobs_tech_sector], ignore_index=True)
jobs_tech['Date'] = pd.to_datetime(jobs_tech['Date'], format='%b-%y')
jobs_tech['Month Year'] = jobs_tech['Date'].dt.strftime('%B %Y')
jobs_tech.pivot(index = 'Date', columns = 'Industry', values = 'Number').to_csv("Data/output/dashboard/jobs_in_tech_companies_WIDE.csv", index = False)

## SALARY DISTRICTUION BY TECH ROLES & CAREER STAGE

levels_fyi_salaries_df['Salary'] = levels_fyi_salaries_df['Salary'].apply(lambda x: x * 1000 if 10 <= x < 100 else x)
salary_levels_summary = levels_fyi_salaries_df[(levels_fyi_salaries_df['Metric'] == "Summary")]
salary_levels_summary.to_csv('Data/output/dashboard/tech_jobs_pay_within_percentile_level_occupation.csv', index = False)

## TOP RANKING TECH CO'S

top_companies_pay_role = levels_fyi_salaries_df[(levels_fyi_salaries_df['Measurement'] == 'Ranking') & (levels_fyi_salaries_df['Metric'] == 'Top Company') & levels_fyi_salaries_df['Level'] == 'All']
top_location_pay_role = levels_fyi_salaries_df[(levels_fyi_salaries_df['Measurement'] == 'Ranking') & (levels_fyi_salaries_df['Metric'] == 'Top Location') & levels_fyi_salaries_df['Level'] == 'All']

cities = ['Sydney', 'Melbourne', 'Perth', 'Brisbane', 'Canberra', 'Darwin', 'Adelaide']
top_location_pay = top_location_pay_role[['Label', 'Salary', 'Job Title']].drop_duplicates()
top_location_pay['type'] = 'Location'
top_companies_pay = top_companies_pay_role[['Label', 'Salary', 'Job Title']].drop_duplicates()
top_companies_pay['type'] = 'Company'
top_rank = pd.concat([top_companies_pay, top_location_pay], ignore_index=True)
top_rank.to_csv('Data/output/dashboard/tech_jobs_top_rank_long.csv', index = False)

## ICT PAY JOB ADS

## -- Data directly received from SEEK
ict_job_ad_salary  = ict_job_ad_salary.pivot(index = 'date', columns = 'industry', values= 'ave_salary').reset_index()
ict_job_ad_salary['date'] = pd.to_datetime(ict_job_ad_salary['date'], format = '%d/%m/%Y')
ict_job_ad_salary['Month Year'] = ict_job_ad_salary['date'].dt.strftime('%B %Y')
ict_job_ad_salary.to_csv('Data/output/dashboard/ict_job_ads.csv', index = False)

## CROSS SECTOR COMPARISIONS

# Convert ABN columns to string for consistent merging and filtering
wgea_salary_df['Employer ABN'] = wgea_salary_df['Employer ABN'].astype(str)
abn_df['ABN'] = abn_df['ABN'].astype(str)

# Filter the WGEA data to include only companies from the Tech Council ABN list
tech_council_wgea_salary_df = wgea_salary_df[wgea_salary_df['Employer ABN'].isin(abn_df['ABN'])].copy()

# Define columns for cleaning
salary_columns = [
    'Total workforce - average total remuneration ($)*',
    'Upper quartile - average total remuneration ($)',
    'Upper-middle quartile - average total remuneration ($)',
    'Lower-middle quartile  - average total remuneration ($)',
    'Lower quartile - average total remuneration ($)'
]

percentage_columns = [
    'Average total remuneration GPG (%)',
    'Average base salary GPG (%)',
    'Median total remuneration GPG (%)',
    'Median base salary GPG (%)',
    '2022-23 Median total remuneration GPG (%)',
    '2022-23 Median base salary GPG (%)',
    'Total workforce % women',
    'Upper quartile % women',
    'Upper-middle quartile % women',
    'Lower-middle quartile % women',
    'Lower quartile % women'
]

# Clean and convert salary and percentage columns to a numeric format
# This removes '$', ',', '%' and handles 'NC' values
for col in salary_columns:
    wgea_salary_df[col] = wgea_salary_df[col].replace({'\$': '', ',': ''}, regex=True).astype(float)
    tech_council_wgea_salary_df[col] = tech_council_wgea_salary_df[col].replace({'\$': '', ',': ''}, regex=True).astype(float)

for col in percentage_columns:
    wgea_salary_df[col] = wgea_salary_df[col].replace({'%': ''}, regex=True)
    wgea_salary_df[col] = pd.to_numeric(wgea_salary_df[col], errors='coerce') / 100
    tech_council_wgea_salary_df[col] = tech_council_wgea_salary_df[col].replace({'%': ''}, regex=True)
    tech_council_wgea_salary_df[col] = pd.to_numeric(tech_council_wgea_salary_df[col], errors='coerce') / 100

# Define the other sectors of interest
other_sectors_df = wgea_salary_df[wgea_salary_df['Industry (ANZSIC Division)'].isin([
    'Financial and Insurance Services',
    'Professional, Scientific and Technical Services'
])].copy()

# Group by industry division and calculate the average salary and women percentage
sector_comparison_df = other_sectors_df.groupby('Industry (ANZSIC Division)').agg(
    Avg_Total_Remuneration=('Total workforce - average total remuneration ($)*', 'mean'),
    Avg_Women_Percentage=('Total workforce % women', 'mean')
).reset_index()

anzsic_class = ['Software Publishing', 'Internet Publishing and Broadcasting', 'Telecommunications Services', 'Internet Service Providers and Web Search Portals', 'Data Processing and Web Hosting Services', 'Computer System Design and Related Services']
direct_tech_sector_df =  wgea_salary_df[wgea_salary_df['Industry (ANZSIC Class)'].isin(anzsic_class)]
direct_tech_sector_df['Sector'] = 'Direct Tech Sector'

direct_tech_sector_df['Industry (ANZSIC Division)'] = 'Direct Tech Sector (All Companies)'

direct_tech_sector_df = direct_tech_sector_df.groupby('Industry (ANZSIC Division)').agg(
    Avg_Total_Remuneration=('Total workforce - average total remuneration ($)*', 'mean'),
    Avg_Women_Percentage=('Total workforce % women', 'mean')
).reset_index()

# Add the tech sector to the comparison
tech_avg_remuneration = tech_council_wgea_salary_df['Total workforce - average total remuneration ($)*'].mean()
tech_avg_women = tech_council_wgea_salary_df['Total workforce % women'].mean()
tech_row = pd.DataFrame([['Tech Sector (TCA Members)', tech_avg_remuneration, tech_avg_women]], 
                        columns=['Industry (ANZSIC Division)', 'Avg_Total_Remuneration', 'Avg_Women_Percentage'])

### Cross-Sector Comparison (Average Salary & Gender Representation) 

final_comparison_df = pd.concat([sector_comparison_df, direct_tech_sector_df, tech_row], ignore_index=True)
final_comparison_df['Avg_Total_Remuneration'] = final_comparison_df['Avg_Total_Remuneration'].apply(
    lambda x: f"${x:,.0f}" 
)
final_comparison_df['Avg_Women_Percentage'] = final_comparison_df['Avg_Women_Percentage'].apply(
    lambda x: f"{x*100:.1f}%" 
)
final_comparison_df = final_comparison_df.sort_values(
    by='Avg_Total_Remuneration',
    ascending=False,
    key=lambda col: col.str.replace(r'[$,]', '', regex=True).astype(float) if col.dtype == "object" else col
)

final_comparison_df.rename(columns={'Industry (ANZSIC Division)': 'Industry', 'Avg_Total_Remuneration':'Average Total Remuneration', 'Avg_Women_Percentage': 'Average % of Women'}, inplace=True)
final_comparison_df.to_csv("Data/output/dashboard/salaries_comparision.csv", index = False)



## PAY QUARTILES

# Calculate average WGEA quartile salaries for Tech Council members
avg_wgea_quartile_salaries = {
    'Q4': tech_council_wgea_salary_df['Upper quartile - average total remuneration ($)'].mean(),
    'Q3': tech_council_wgea_salary_df['Upper-middle quartile - average total remuneration ($)'].mean(),
    'Q2': tech_council_wgea_salary_df['Lower-middle quartile  - average total remuneration ($)'].mean(),
    'Q1': tech_council_wgea_salary_df['Lower quartile - average total remuneration ($)'].mean()
}
avg_wgea_total_salaries = {
    'Total workforce': tech_council_wgea_salary_df['Total workforce - average total remuneration ($)*'].mean()
}

for quartile, salary in avg_wgea_quartile_salaries.items():
    f"  - {quartile}: ${salary:,.0f}"

# Filter Levels.FYI for all percentile and median salary data
levels_percentile_df = levels_fyi_salaries_df[(levels_fyi_salaries_df['Metric'] == 'Summary') & (levels_fyi_salaries_df['Measurement'] == 'Percentile')].copy()
levels_percentile_df['Level'] = levels_percentile_df['Level'].str.strip()

# Find specific roles for a more comprehensive narrative
def find_example_roles(df, salary_min, salary_max):
    examples = []
    # Prioritize well-known technical and non-technical roles
    prioritized_roles = [
        'Software Engineer', 'Data Scientist', 'Product Manager', 
        'Product Designer', 'Marketing', 'Recruiter', 'Legal', 'Ux Researcher', 
        'Business Development', 'Technical Program Manager', 'Sales Engineer',  
        'Data Science Manager', 'Software Engineer Manager', 'Security Analyst',
        'Graphic Designer', 'Marketing Operations', 'Product Design Manager', 'Hardware Engineer'
    ]
    for role in prioritized_roles:
        roles_df = df[(df['Job Title'] == role) & 
                      (df['Salary'] >= salary_min) & 
                      (df['Salary'] <= salary_max)]
        if not roles_df.empty:
            for _, row in roles_df.head(1).iterrows():
                examples.append(f"{row['Label']} {row['Level']} {row['Job Title']}")
    return examples

levels_percentile_df = levels_fyi_salaries_df[(levels_fyi_salaries_df['Metric'] == 'Summary') & 
                                (levels_fyi_salaries_df['Measurement'] == 'Percentile') &
                                (levels_fyi_salaries_df['Level'] != 'All')].copy()

# Find example roles for each quartile based on salary ranges
q1_examples = find_example_roles(levels_percentile_df, 0, avg_wgea_quartile_salaries['Q1'])
q2_examples = find_example_roles(levels_percentile_df, avg_wgea_quartile_salaries['Q1'], avg_wgea_quartile_salaries['Q2'])
q3_examples = find_example_roles(levels_percentile_df, avg_wgea_quartile_salaries['Q2'], avg_wgea_quartile_salaries['Q3'])
q4_examples = find_example_roles(levels_percentile_df, avg_wgea_quartile_salaries['Q3'], np.inf)

picked_1 = random.sample(q1_examples, 4)
picked_2 = random.sample(q2_examples, 4)
picked_3 = random.sample(q3_examples, 4)
picked_4 = random.sample(q4_examples, 4)

# Create the final conceptual mapping DataFrame with a more narrative style
mapping_data = pd.DataFrame({
    'WGEA Quartile': [f'Q{i} ({"Lower" if i == 1 else "Lower-middle" if i == 2 else "Upper-middle" if i == 3 else "Upper"} quartile)' for i in range(1, 5)],
    'Tech Sector (TCA Members) Salaries - Average': [f"${avg_wgea_quartile_salaries[f'Q{i}']:,.0f}" for i in range(1, 5)],
    'Gender - Average': [
        f"{tech_council_wgea_salary_df['Lower quartile % women'].mean():.1%} women, {1 - tech_council_wgea_salary_df['Lower quartile % women'].mean():.1%} men",
        f"{tech_council_wgea_salary_df['Lower-middle quartile % women'].mean():.1%} women, {1 - tech_council_wgea_salary_df['Lower-middle quartile % women'].mean():.1%} men",
        f"{tech_council_wgea_salary_df['Upper-middle quartile % women'].mean():.1%} women, {1 - tech_council_wgea_salary_df['Upper-middle quartile % women'].mean():.1%} men",
        f"{tech_council_wgea_salary_df['Upper quartile % women'].mean():.1%} women, {1 - tech_council_wgea_salary_df['Upper quartile % women'].mean():.1%} men"
    ],
    'Tech Sector (TCA Members) - % Women - Average': [
        round(tech_council_wgea_salary_df['Lower quartile % women'].mean() * 100, 1),
        round(tech_council_wgea_salary_df['Lower-middle quartile % women'].mean() * 100, 1),
        round(tech_council_wgea_salary_df['Upper-middle quartile % women'].mean() * 100, 1),
        round(tech_council_wgea_salary_df['Upper quartile % women'].mean() * 100, 1)
    ],
    'Tech Sector (TCA Members) - % Men - Average': [
        100 - round(tech_council_wgea_salary_df['Lower quartile % women'].mean() * 100, 1),
        100 - round(tech_council_wgea_salary_df['Lower-middle quartile % women'].mean() * 100, 1),
        100 - round(tech_council_wgea_salary_df['Upper-middle quartile % women'].mean() * 100, 1),
        100 - round(tech_council_wgea_salary_df['Upper quartile % women'].mean() * 100, 1)
    ],
    'Tech Sector (TCA Members) - Example Roles': [
        f"Quartile 1 band include, on average, an  {picked_1[0].lower()}, a {picked_1[1].lower()}, a {picked_1[2].lower()}, or a {picked_2[3].lower()}.",
        f"Low-mid quartile roles include, on average, an {picked_2[0].lower()}, a {picked_2[1].lower()}, a {picked_2[2].lower()}, or a {picked_2[3].lower()}.",
        f"Mid-upper quartile roles include, on average, an {picked_3[0].lower()}, a {picked_3[1].lower()}, a {picked_3[2].lower()}, or a {picked_3[3].lower()}.",
        f"This top-paying quartile would contain high-level leadership roles including C-Suite and Vice President-level executives as well as top-tier technical roles like a {picked_4[0].lower()}, a {picked_4[1].lower()}."
        ]
})

mapping_data['Tech Sector (TCA Members) - Example Roles'] = mapping_data['Tech Sector (TCA Members) - Example Roles'].str.replace(r"\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])", r"a \1", regex=True)
mapping_data['Tech Sector (TCA Members) - Example Roles'] = mapping_data['Tech Sector (TCA Members) - Example Roles'].str.replace(r"(\d+th)\s(?!p)(\w+)", r"\1 percentile \2", regex=True, case=False)

mapping_data_total= pd.DataFrame({
    'WGEA Quartile': ['Total workforce'],
    'Tech Sector (TCA Members) Salaries - Average': [f"${avg_wgea_total_salaries['Total workforce']:,.0f}"],
    'Gender - Average': [
        f"{tech_council_wgea_salary_df['Total workforce % women'].mean():.1%} women, {1 - tech_council_wgea_salary_df['Total workforce % women'].mean():.1%} men"
    ],
    'Tech Sector (TCA Members) - % Women - Average': [
        round(tech_council_wgea_salary_df['Total workforce % women'].mean() * 100, 1),
    ],
    'Tech Sector (TCA Members) - % Men - Average': [
        100 - round(tech_council_wgea_salary_df['Total workforce % women'].mean() * 100, 1),
    ],
    'Tech Sector (TCA Members) - Example Roles': ['']
})

mapping_data = pd.concat([mapping_data, mapping_data_total])

direct_tech_sector =  wgea_salary_df[wgea_salary_df['Industry (ANZSIC Class)'].isin(anzsic_class)]
direct_tech_sector['Sector'] = 'Direct Tech Sector'
direct_tech_avg = direct_tech_sector.groupby('Sector').mean(numeric_only=True).reset_index()[[
        'Sector', 
        'Total workforce % women', 'Upper quartile % women', 'Upper-middle quartile % women',
       'Lower-middle quartile % women', 'Lower quartile % women',
       'Total workforce - average total remuneration ($)*',
       'Upper quartile - average total remuneration ($)',
       'Upper-middle quartile - average total remuneration ($)',
       'Lower-middle quartile  - average total remuneration ($)',
       'Lower quartile - average total remuneration ($)']]

direct_tech_avg_long = direct_tech_avg.melt(
    id_vars="Sector",
    var_name="Metric",
    value_name="Value"
)

direct_tech_avg_long["Metric"] = direct_tech_avg_long["Metric"].str.replace("*", "", regex=False)

# Extract group (everything before "% women" or "average total remuneration")
direct_tech_avg_long['Group'] = direct_tech_avg_long['Metric'].str.extract(r'^(.*?(?:workforce|quartile))', expand=False).str.strip()

# Extract measure (everything after the group)
direct_tech_avg_long['Measure'] = direct_tech_avg_long['Metric'].str.replace(r'^.*?(workforce|quartile)\s*-?\s*', '', regex=True).str.strip()
# Drop original Metric if desired
direct_tech_avg_long = direct_tech_avg_long.drop(columns='Metric')
direct_tech_avg = direct_tech_avg_long.pivot(index = ['Sector','Group'], columns='Measure', values = 'Value').reset_index()
direct_tech_avg['WGEA Quartile'] = direct_tech_avg['Group']
direct_tech_avg['WGEA Avg Salary (avg across group)'] = direct_tech_avg['average total remuneration ($)']
direct_tech_avg['Gender Split (avg across group)'] = direct_tech_avg['% women']


direct_tech_avg = direct_tech_avg[['WGEA Quartile', 'WGEA Avg Salary (avg across group)', 'Gender Split (avg across group)']]

quartile_map = {
    "Lower quartile": "Q1 (Lower quartile)",
    "Lower-middle quartile": "Q2 (Lower-middle quartile)",
    "Upper-middle quartile": "Q3 (Upper-middle quartile)",
    "Upper quartile": "Q4 (Upper quartile)",
    "Total workforce": "Total workforce"
}

direct_tech_avg['WGEA Quartile'] = direct_tech_avg['WGEA Quartile'].map(quartile_map)
direct_tech_avg['Direct Tech Salary - Average'] = direct_tech_avg['WGEA Avg Salary (avg across group)'].apply(lambda x: f"${x:,.0f}")

direct_tech_avg['Direct Tech Gender - Average'] = direct_tech_avg['Gender Split (avg across group)'].apply(
    lambda x: f"{x:.1%} women, {1 - x:.1%} men"
)
direct_tech_avg['Direct Tech - % Women - Average'] = direct_tech_avg['Gender Split (avg across group)'].apply(
    lambda x: f"{x*100:.3}"
)

direct_tech_avg['Direct Tech - % Men - Average'] = direct_tech_avg['Gender Split (avg across group)'].apply(
    lambda x: f"{100-(x*100):.3}"
)
direct_tech_avg.drop(columns=['WGEA Avg Salary (avg across group)', 'Gender Split (avg across group)'], inplace=True)
quartile_order = ["Q1 (Lower quartile)", "Q2 (Lower-middle quartile)", 
                  "Q3 (Upper-middle quartile)", "Q4 (Upper quartile)"]
total_row = direct_tech_avg[direct_tech_avg['WGEA Quartile'] == "Total workforce"]
quartile_rows = direct_tech_avg[direct_tech_avg['WGEA Quartile'] != "Total workforce"]
quartile_rows['WGEA Quartile'] = pd.Categorical(quartile_rows['WGEA Quartile'], categories=quartile_order, ordered=True)
quartile_rows = quartile_rows.sort_values('WGEA Quartile')
direct_tech_avg_sorted = pd.concat([quartile_rows, total_row], ignore_index=True)

mapping_data = mapping_data.merge(direct_tech_avg_sorted, left_on="WGEA Quartile", right_on = "WGEA Quartile")

mapping_data.to_csv('Data/output/dashboard/tech_pay_quartiles.csv', index = False)

# WOMEN IN TECH
#--------------#
#--------------#
## WOMENS PAY SCALES IN TECH

### See tech pay quartiles above. 

## WOMEN IN LEADERSHIP BY SECTOR
wgea_workforce_composition.loc[wgea_workforce_composition['manager_category'] == 'Non-manager', 'occupation'] = 'Non-managers'

wgea_workforce_composition['occupation'] = wgea_workforce_composition['occupation'].replace({'Key Management Personnel': 'Other managers', 'Overseas Reporting Managers': 'Other managers'})

other_sectors_comp = wgea_workforce_composition[wgea_workforce_composition['anzsic_division'].isin([
    'Financial and Insurance Services',
    'Professional, Scientific and Technical Services'
])].copy()

direct_tech_sector_comp =  wgea_workforce_composition[wgea_workforce_composition['anzsic_class'].isin(anzsic_class)]

wgea_workforce_composition['employer_abn'] = wgea_workforce_composition['employer_abn'].astype(str).str.rstrip('.0')
tca_sector_comp = wgea_workforce_composition[wgea_workforce_composition['employer_abn'].isin(abn_df['ABN'])].copy()

tca_sector_comp['Sector'] = 'Tech Sector (TCA Members)'
other_sectors_comp['Sector'] = wgea_workforce_composition['anzsic_division']
direct_tech_sector_comp['Sector'] = 'Direct Tech Sector (All Companies)'

comp_sum = pd.concat([tca_sector_comp, direct_tech_sector_comp, other_sectors_comp])[['Sector', 'occupation', 'gender', 'n_employees']].groupby(['Sector', 'occupation', 'gender']).sum(numeric_only=True)

comp_sum['pct'] = round((
    comp_sum['n_employees'] /
    comp_sum.groupby(['Sector', 'occupation'])['n_employees'].transform('sum')
) * 100, 1)

comp_pct = comp_sum.reset_index().pivot(columns='gender', values='pct', index=['Sector', 'occupation']).reset_index()
comp_pct.to_csv('Data/output/dashboard/workplace_leadership_comp_pct.csv', index = False)

## WOMEN PROMOTION RATES BY MANAGERIAL LEVEL & SECTOR

wgea_mgmt_promotions = wgea_mgmt[wgea_mgmt['movement_type'] == 'Promotions']

other_sectors_mgmt_promotions = wgea_mgmt_promotions[wgea_mgmt_promotions['anzsic_division'].isin([
    'Financial and Insurance Services',
    'Professional, Scientific and Technical Services'
])].copy()

direct_tech_sector_mgmt_promotions =  wgea_mgmt_promotions[wgea_mgmt_promotions['anzsic_class'].isin(anzsic_class)]

wgea_mgmt_promotions['employer_abn'] = wgea_mgmt_promotions['employer_abn'].astype(str)
tca_sector_mgmt_promotions =  wgea_mgmt_promotions[wgea_mgmt_promotions['employer_abn'].isin(abn_df['ABN'])].copy()

tca_sector_mgmt_promotions['Sector'] = 'Tech Sector (TCA Members)'
other_sectors_mgmt_promotions['Sector'] = wgea_mgmt_promotions['anzsic_division']
direct_tech_sector_mgmt_promotions['Sector'] = 'Direct Tech Sector (All Companies)'

promos_sum = pd.concat([tca_sector_mgmt_promotions, direct_tech_sector_mgmt_promotions, other_sectors_mgmt_promotions])[['Sector', 'manager_type', 'gender', 'n_employees']].groupby(['Sector', 'manager_type', 'gender']).sum(numeric_only=True)

promos_sum['pct'] = round((
    promos_sum['n_employees'] /
    promos_sum.groupby(['Sector', 'manager_type'])['n_employees'].transform('sum')
) * 100, 1)

promos_pct = promos_sum.reset_index().pivot(columns='gender', values='pct', index=['Sector', 'manager_type']).reset_index()
promos_pct.to_csv('Data/output/dashboard/mgmt_promotions_pct.csv', index = False)
promos_n = promos_sum.reset_index().pivot(columns='gender', values='n_employees', index=['Sector', 'manager_type']).reset_index()
promos_n.to_csv('Data/output/dashboard/mgmt_promotions_n.csv', index = False)

# INTERNATIONAL BENCHMARKING
#---------------------------#
#---------------------------#
## AI

## R&D

## SCIENCE & INNOVATION - GREEN ECONOMY

## SKILLS

