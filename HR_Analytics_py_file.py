import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


# Load the dataset
file_path = 'HR Analytics Assignment.csv'
df = pd.read_csv(file_path)
# Define a mapping for categorical ratings to numeric values
rating_map = {
    "Very Dissatisfied": 1,
    "Dissatisfied": 2,
    "Neutral": 3,
    "Satisfied": 4,
    "Very Satisfied": 5
}
# Apply the mapping to relevant columns
rating_columns = ['Overall Rating', 'Career Growth', 'Compensation and Benefits', 'Work-life Balance']
for col in rating_columns:
    df[col] = df[col].map(rating_map)

df.describe()
df.isnull().sum()
df.set_index("EmpID",inplace=True)
df['Date Of Joining'] = pd.to_datetime(df['Date Of Joining'])
df.head()
df_missing_percent = df.isnull().sum()/len(df)
df_missing_percent
# 1. Overall rating w.r.t location
plt.figure(figsize=(20, 12))
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='Location', y='Overall Rating')
plt.title('Overall Ratings by Location')
plt.xlabel('Location')
plt.ylabel('Overall Rating')
# 2. Career growth ratings w.r.t gender
plt.figure(figsize=(20, 12))
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='Gender', y='Career Growth')
plt.title('Career Growth Ratings by Gender')
plt.xlabel('Gender')
plt.ylabel('Career Growth')

# 3. Work-life balance w.r.t employee level
plt.figure(figsize=(20, 12))
plt.subplot(2, 2, 3)
sns.barplot(data=df, x='Level', y='Work-life Balance', ci=None, estimator=sum)
plt.title('Work-Life Balance by Employee Level')
plt.xlabel('Employee Level')
plt.ylabel('Sum of Work-Life Balance Ratings')      ##################CHECK HERE WHY 2 TIMES MIDDLE IS COMING################33
# 4. Average ratings by year of joining
plt.figure(figsize=(20, 12))
df['Year Of Joining'] = df['Date Of Joining'].dt.year
yearly_ratings = df.groupby('Year Of Joining')[rating_columns].mean().reset_index()

plt.subplot(2, 2, 4)
for col in rating_columns:
    plt.plot(yearly_ratings['Year Of Joining'], yearly_ratings[col], label=col)

plt.title('Average Ratings by Year of Joining')
plt.xlabel('Year of Joining')
plt.ylabel('Average Rating (Numeric)')
plt.legend(title='Rating Categories')

plt.tight_layout()
plt.show()


# Pivot Tables for Additional Insights
# Pivot table for overall ratings by location and gender
pivot_table = df.pivot_table(values='Overall Rating', index='Location', columns='Gender', aggfunc='mean')
print("Pivot Table: Overall Ratings by Location and Gender")
print(pivot_table)
pivot_table.to_csv('pivot_table_overall_ratings.csv')

correlation_matrix = df[['Overall Rating', 'Career Growth', 'Compensation and Benefits', 'Work-life Balance']].corr()
correlation_matrix
highest_correlation = correlation_matrix['Overall Rating'][1:].idxmax()
highest_value = correlation_matrix['Overall Rating'][1:].max()

# Display results
print("Correlation Matrix:")
print(correlation_matrix)
print(f"\nThe factor with the highest correlation with Overall Rating is: {highest_correlation} with a value of {highest_value:.2f}")

# Save a summarized dataset for LLM input
summary_data = {
    'Factor': ['Career Growth', 'Compensation and Benefits', 'Work-life Balance'],
    'Correlation with Overall Rating': correlation_matrix['Overall Rating'][1:].values
}
summary_df = pd.DataFrame(summary_data)
summary_file_path = 'correlation_summary.csv'
summary_df.to_csv(summary_file_path, index=False)

# Define tenure groups
df['Tenure (Years)'] = (datetime.now() - df['Date Of Joining']).dt.days / 365.25

# Categorize associates into high-tenured and Covid-era joiners
df['Tenure Category'] = df['Tenure (Years)'].apply(
    lambda x: 'High Tenured (15+ Years)' if x >= 15 else ('Covid-Era Joiner (2020+)' if x <= 3 else 'Others')
)

# Count dissatisfaction levels in each group
dissatisfaction_levels = ['Very Dissatisfied', 'Dissatisfied']

def is_dissatisfied(overall_rating):
    return overall_rating in dissatisfaction_levels

df['Dissatisfied'] = df['Overall Rating'].apply(is_dissatisfied)

# Group by tenure category and calculate dissatisfaction rate
dissatisfaction_summary = df.groupby('Tenure Category').agg(
    # Total_Associates=('EmpID', 'count'),
    Dissatisfied_Associates=('Dissatisfied', 'sum'),
    Dissatisfaction_Rate=('Dissatisfied', 'mean')
).reset_index()

dissatisfaction_summary['Dissatisfaction_Rate'] = dissatisfaction_summary['Dissatisfaction_Rate'] * 100

# Save the results to a CSV file
output_file = 'Dissatisfaction_By_Tenure.csv'
dissatisfaction_summary.to_csv(output_file, index=False)

dissatisfaction_summary


