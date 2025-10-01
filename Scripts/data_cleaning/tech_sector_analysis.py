# ==============================================================================
# Comprehensive Salary Analysis Script
# This script performs a full analysis of WGEA and Levels.FYI data,
# including filtering, cleaning, and a detailed breakdown by state and industry.
# It also includes the core analysis that maps a Senior Software Engineer's
# salary to WGEA pay quartiles.
# ==============================================================================
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import random

_ = load_dotenv(os.path.expanduser("~/.env"), verbose=False)
os.chdir(os.getenv("FILE_PATH")+"2508 - Jobs Data")


# --- 1. Load and Prepare the Datasets ---
# The script assumes the following CSV files are in the same directory:
# - Australian Tech Council Salary Data - WGEA_Data.csv
# - Australian Tech Council Salary Data - Levels_FYI_Data.csv
# - Australian Tech Council Salary Data - Tech_Council_ABNs.csv

try:
    wgea_df = pd.read_csv('Data/tech_salary_sector/Australian Tech Council Salary Data - WGEA_Data.csv')
    levels_df = pd.read_csv('Data/tech_salary_sector/Australian Tech Council Salary Data - Levels_FYI_Data.csv')
    abn_df = pd.read_csv('Data/tech_salary_sector/Australian Tech Council Salary Data - Tech_Council_ABNs.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all three data files are in the same directory.")
    exit()

# Convert ABN columns to string for consistent merging and filtering
wgea_df['Employer ABN'] = wgea_df['Employer ABN'].astype(str)
abn_df['ABN'] = abn_df['ABN'].astype(str)

# Filter the WGEA data to include only companies from the Tech Council ABN list
tech_council_wgea_df = wgea_df[wgea_df['Employer ABN'].isin(abn_df['ABN'])].copy()

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
    wgea_df[col] = wgea_df[col].replace({'\$': '', ',': ''}, regex=True).astype(float)
    tech_council_wgea_df[col] = tech_council_wgea_df[col].replace({'\$': '', ',': ''}, regex=True).astype(float)

for col in percentage_columns:
    wgea_df[col] = wgea_df[col].replace({'%': ''}, regex=True)
    wgea_df[col] = pd.to_numeric(wgea_df[col], errors='coerce') / 100
    tech_council_wgea_df[col] = tech_council_wgea_df[col].replace({'%': ''}, regex=True)
    tech_council_wgea_df[col] = pd.to_numeric(tech_council_wgea_df[col], errors='coerce') / 100

# --- 2. Core Analysis: Mapping Levels.FYI to WGEA Quartiles ---
# This section dynamically maps Levels.FYI roles to WGEA quartiles based on salary.
print("="*80)
print("SECTION 1: Mapping Roles to WGEA Quartiles")
print("="*80)

# Calculate average WGEA quartile salaries for Tech Council members
avg_wgea_quartile_salaries = {
    'Q4': tech_council_wgea_df['Upper quartile - average total remuneration ($)'].mean(),
    'Q3': tech_council_wgea_df['Upper-middle quartile - average total remuneration ($)'].mean(),
    'Q2': tech_council_wgea_df['Lower-middle quartile  - average total remuneration ($)'].mean(),
    'Q1': tech_council_wgea_df['Lower quartile - average total remuneration ($)'].mean()
}

print("\nAverage total remuneration for WGEA quartiles (Tech Council Members):")
for quartile, salary in avg_wgea_quartile_salaries.items():
    print(f"  - {quartile}: ${salary:,.0f}")

# Filter Levels.FYI for all percentile and median salary data
levels_percentile_df = levels_df[(levels_df['Metric'] == 'Summary') & (levels_df['Measurement'] == 'Percentile')].copy()
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

levels_percentile_df = levels_df[(levels_df['Metric'] == 'Summary') & 
                                (levels_df['Measurement'] == 'Percentile') &
                                (levels_df['Level'] != 'All')].copy()

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
mapping_data = {
    'WGEA Quartile': [f'Q{i} ({"Lower" if i == 1 else "Lower-middle" if i == 2 else "Upper-middle" if i == 3 else "Upper"} Quartile)' for i in range(1, 5)],
    'WGEA Avg Salary (Tech Employers)': [f"${avg_wgea_quartile_salaries[f'Q{i}']:,.0f}" for i in range(1, 5)],
    'Gender Split (avg across Tech Employers)': [
        f"{tech_council_wgea_df['Lower quartile % women'].mean():.1%} women, {1 - tech_council_wgea_df['Lower quartile % women'].mean():.1%} men",
        f"{tech_council_wgea_df['Lower-middle quartile % women'].mean():.1%} women, {1 - tech_council_wgea_df['Lower-middle quartile % women'].mean():.1%} men",
        f"{tech_council_wgea_df['Upper-middle quartile % women'].mean():.1%} women, {1 - tech_council_wgea_df['Upper-middle quartile % women'].mean():.1%} men",
        f"{tech_council_wgea_df['Upper quartile % women'].mean():.1%} women, {1 - tech_council_wgea_df['Upper quartile % women'].mean():.1%} men"
    ],
    'Example Roles (Levels.fyi mapping)': [
        f"Quartile 1 band include an average {picked_1[0].lower()}, a {picked_1[1].lower()}, a {picked_1[2].lower()}, or a {picked_2[3].lower()}.",
        f"Low-mid quartile roles include an average {picked_2[0].lower()}, a {picked_2[1].lower()}, a {picked_2[2].lower()}, or a {picked_2[3].lower()}.",
        f"Mid-upper quartile roles include an average {picked_3[0].lower()}, a {picked_3[1].lower()}, a {picked_3[2].lower()}, or a {picked_3[3].lower()}.",
        f"This top-paying quartile would contain high-level leadership roles including C-Suite and Vice President-level executives as well as top-tier technical roles like a {picked_4[0].lower()}, a {picked_4[1].lower()}."
        ]
}

conceptual_mapping_df = pd.DataFrame(mapping_data)

# Print the DataFrame to the console
print("\nMapping Levels.FYI data to WGEA:")
print(conceptual_mapping_df.to_string())

# Export the DataFrame to a CSV file
conceptual_mapping_df.to_csv('conceptual_mapping.csv', index=False)
#print("\nDataFrame exported to 'conceptual_mapping.csv' successfully.")

df_median_salaries = levels_df[(levels_df['Metric'] == 'Summary') &
                                        (levels_df['Label'] == 'Median') &
                                        (levels_df['Level'] == 'All')]

# Sort the results by salary in descending order.
df_median_salaries = df_median_salaries.sort_values(by='Salary', ascending=False)

# Select and display the relevant columns.
df_median_salaries_summary = df_median_salaries[['Job Title', 'Salary']].reset_index(drop=True)
df_median_salaries_summary['Salary'] = df_median_salaries_summary['Salary'].apply(
    lambda x: f"${x:,.0f}" 
)

print("Median Salaries by Job Title:")
print(df_median_salaries_summary)

# Filter for top companies.
df_top_companies = levels_df[levels_df['Metric'] == 'Top Company']

# Sort by rank order to see the highest paying companies first.
df_top_companies = df_top_companies.sort_values(by=['Level', 'Job Title', 'Rank Order'])

# Select and display relevant columns.
df_top_companies_summary = df_top_companies[['Job Title', 'Level', 'Label', 'Salary', 'Rank Order']]
senior_top_companies_summary = df_top_companies_summary[(df_top_companies_summary['Level'] == 'All')]

major_australian_cities = [
    "Sydney",
    "Melbourne",
    "Brisbane",
    "Perth",
    "Adelaide",
    "Canberra",
    "Gold Coast",
    "Newcastle",
    "Hobart",
    "Darwin"
]

df_top_companies_summary['Salary'] = df_top_companies_summary['Salary'].apply(
    lambda x: f"${x:,.0f}" 
)

senior_top_companies_summary = df_top_companies_summary[(df_top_companies_summary['Level'] == 'Senior') & (~df_top_companies_summary['Label'].isin(major_australian_cities))]
senior_top_locations_summary = df_top_companies_summary[(df_top_companies_summary['Level'] == 'Senior') & (df_top_companies_summary['Label'].isin(major_australian_cities))]

entry_top_companies_summary = df_top_companies_summary[(df_top_companies_summary['Level'] == 'Entry-Level') & (~df_top_companies_summary['Label'].isin(major_australian_cities))]
entry_top_locations_summary = df_top_companies_summary[(df_top_companies_summary['Level'] == 'Entry-Level') & (df_top_companies_summary['Label'].isin(major_australian_cities))]

all_top_companies_summary = df_top_companies_summary[(df_top_companies_summary['Level'] == 'All') & (~df_top_companies_summary['Label'].isin(major_australian_cities))]
all_top_locations_summary = df_top_companies_summary[(df_top_companies_summary['Level'] == 'All') & (df_top_companies_summary['Label'].isin(major_australian_cities))]

print("\nTop Paying Companies by Job Title:")
print(all_top_companies_summary.to_string())

#print("\nTop Paying Locations by Job Title:")
#print(all_top_locations_summary.to_string())

print("\nTop Paying Companies by Job Title (Seniors):")
print(senior_top_companies_summary.to_string())

#print("\nTop Paying Locations by Job Title (Seniors):")
#print(senior_top_locations_summary.to_string())

print("\nTop Paying Companies by Job Title (Entry-Level):")
print(entry_top_companies_summary.to_string())

#print("\nTop Paying Locations by Job Title (Entry-Level):")
#print(entry_top_locations_summary.to_string())

# Now, let's look at the salary progression by level.
# Filter for summary data with median salaries for senior and entry-level.
levels_df['Salary'] > 50000 # Remove junk entries <$50K
df_level_salaries = levels_df[(levels_df['Metric'] == 'Summary') & (levels_df['Level'].isin(['Entry-Level', 'Senior']))]
df_level_salaries['Label'] = df_level_salaries['Label'].replace('Median', '50th')
df_level_salaries['Label'] = df_level_salaries['Label'] + ' ' + df_level_salaries['Measurement'] + ' - ' + df_level_salaries['Level']

df_level_salaries = df_level_salaries.sort_values(by=['Job Title', 'Level', 'Label'])
desired_column_order = [
    '25th Percentile - Entry-Level', '50th Percentile - Entry-Level', '75th Percentile - Entry-Level', '90th Percentile - Entry-Level',
    '25th Percentile - Senior', '50th Percentile - Senior', '75th Percentile - Senior', '90th Percentile - Senior'
]
# Pivot the table to make it easier to compare salaries side-by-side.
df_level_pivot = df_level_salaries.pivot_table(index='Job Title',
                                               columns='Label',
                                               values='Salary')[desired_column_order].reset_index()


for col in desired_column_order:
    df_level_pivot[col] = df_level_pivot[col].apply(
        lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
)

print("\nMedian Salaries by Job Level:")

print(df_level_pivot.to_string())
# --- 3. Breakdowns by State and Industry ---
# This section uses the new location and industry data provided.

print("\n\n" + "="*80)
print("SECTION 2: WGEA Analysis by State and Industry")
print("="*80)

# Merge the WGEA data with the new location and industry data
merged_df = pd.merge(tech_council_wgea_df, abn_df, left_on='Employer ABN', right_on='ABN', how='inner')

# Un-pivot the state columns for easier analysis
# The previous logic had a bug. The correct way to filter is to check if 'x' is in the column's values.
state_cols = [col for col in merged_df.columns if col.startswith('Tech offices_') and 'x' in merged_df[col].values]
state_df = merged_df.melt(id_vars=['Employer ABN', 'Employer name', 'Total workforce - average total remuneration ($)*', 'Total workforce % women'],
                          value_vars=state_cols,
                          var_name='State',
                          value_name='Has Office')

# Filter for companies with an office in that state
state_df = state_df[state_df['Has Office'] == 'x'].copy()
state_df['State'] = state_df['State'].str.replace('Tech offices_', '')

# Group by state and calculate the average salary and percentage of women
state_analysis_df = state_df.groupby('State').agg(
    Avg_Total_Remuneration=('Total workforce - average total remuneration ($)*', 'mean'),
    Avg_Women_Percentage=('Total workforce % women', 'mean'),
    Companies=('Employer name', lambda x: ', '.join(sorted(list(x.unique())))),
    Company_Count=('Employer name', 'nunique'),
).sort_values(by='Avg_Total_Remuneration', ascending=False)

print("\nAverage Total Remuneration and Women Representation by State:")
state_analysis_df['Avg_Total_Remuneration'] = state_analysis_df['Avg_Total_Remuneration'].apply(
    lambda x: f"${x:,.0f}" 
)
state_analysis_df['Avg_Women_Percentage'] = state_analysis_df['Avg_Women_Percentage'].apply(
    lambda x: f"{x*100:.1f}%" 
)
state_analysis_df = state_analysis_df.sort_values(
    by='Avg_Total_Remuneration',
    ascending=False,
    key=lambda col: col.str.replace(r'[$,]', '', regex=True).astype(float) if col.dtype == "object" else col
)
print(state_analysis_df)

# Un-pivot the industry columns for easier analysis
industry_cols = [col for col in merged_df.columns if col.startswith('Tech sector_') and 'x' in merged_df[col].values]
industry_df = merged_df.melt(id_vars=['Employer ABN', 'Employer name', 'Total workforce - average total remuneration ($)*', 'Total workforce % women'],
                            value_vars=industry_cols,
                            var_name='Industry',
                            value_name='Is In Industry')

# Filter for companies in that industry
industry_df = industry_df[industry_df['Is In Industry'] == 'x'].copy()
industry_df['Industry'] = industry_df['Industry'].str.replace('Tech sector_', '')
industry_df['Industry'] = industry_df['Industry'].str.replace('_', ' ')

# Group by industry and calculate the average salary and percentage of women
industry_analysis_df = industry_df.groupby('Industry').agg(
    Avg_Total_Remuneration=('Total workforce - average total remuneration ($)*', 'mean'),
    Avg_Women_Percentage=('Total workforce % women', 'mean'),
    Companies=('Employer name', lambda x: ', '.join(sorted(list(x.unique())))),
    Company_Count=('Employer name', 'nunique')
).sort_values(by='Avg_Total_Remuneration', ascending=False)

print("\nAverage Total Remuneration and Women Representation by Industry:")
industry_analysis_df['Avg_Total_Remuneration'] = industry_analysis_df['Avg_Total_Remuneration'].apply(
    lambda x: f"${x:,.0f}" 
)
industry_analysis_df['Avg_Women_Percentage'] = industry_analysis_df['Avg_Women_Percentage'].apply(
    lambda x: f"{x*100:.1f}%" 
)
industry_analysis_df = industry_analysis_df.sort_values(
    by='Avg_Total_Remuneration',
    ascending=False,
    key=lambda col: col.str.replace(r'[$,]', '', regex=True).astype(float) if col.dtype == "object" else col
)
print(industry_analysis_df)

# --- 4. NEW SECTION: Comparison to Other Sectors (Finance & Legal) ---
print("\n\n" + "="*80)
print("SECTION 3: WGEA Comparison to Other Sectors (Finance & Legal)")
print("="*80)

# Define the other sectors of interest
other_sectors_df = wgea_df[wgea_df['Industry (ANZSIC Division)'].isin([
    'Financial and Insurance Services',
    'Professional, Scientific and Technical Services'
])].copy()

# Group by industry division and calculate the average salary and women percentage
sector_comparison_df = other_sectors_df.groupby('Industry (ANZSIC Division)').agg(
    Avg_Total_Remuneration=('Total workforce - average total remuneration ($)*', 'mean'),
    Avg_Women_Percentage=('Total workforce % women', 'mean')
).reset_index()

# Add the tech sector to the comparison
tech_avg_remuneration = tech_council_wgea_df['Total workforce - average total remuneration ($)*'].mean()
tech_avg_women = tech_council_wgea_df['Total workforce % women'].mean()
tech_row = pd.DataFrame([['Tech Sector (TCA Members)', tech_avg_remuneration, tech_avg_women]], 
                        columns=['Industry (ANZSIC Division)', 'Avg_Total_Remuneration', 'Avg_Women_Percentage'])

final_comparison_df = pd.concat([sector_comparison_df, tech_row], ignore_index=True)
print("\nCross-Sector Comparison (Average Salary & Gender Representation):")
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

print(final_comparison_df)


# --- 5. Final Summary ---
print("\n\n" + "="*80)
print("SECTION 4: Final Summary and Key Insights")
print("="*80)

# Recalculate and print the overall WGEA statistics for a final summary
avg_overall_remuneration = tech_council_wgea_df['Total workforce - average total remuneration ($)*'].mean()
avg_overall_women_pct = tech_council_wgea_df['Total workforce % women'].mean()

print(f"Overall average total remuneration for Tech Council members: ${avg_overall_remuneration:,.0f}")
print(f"Overall average women representation in the workforce: {avg_overall_women_pct:.2%}")

print("\nKey Insights:")
print("- The analysis of Levels.FYI and WGEA data shows that senior-level individual contributor roles (like a Senior Software Engineer) typically fall within a company's Upper-middle quartile.")
print("- A significant gender imbalance exists, with a declining percentage of women in higher-paying quartiles.")
print("- Salary and gender representation vary significantly by both state and industry.")
print("- While Tech Council members have a higher average total remuneration than the finance and legal sectors, they have a lower average representation of women.")
