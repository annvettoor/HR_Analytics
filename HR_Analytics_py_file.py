##Importing all the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from scipy.stats import chi2_contingency
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
##Reading input file
file_path = 'HR Analytics Assignment.csv'
df = pd.read_csv(file_path)
##Checking for dupliacte rows if any
duplicates = df[df.duplicated()]
print("Duplicate Rows:")
print(duplicates)
num_duplicates = len(duplicates)
df.describe()
##Checking for null values
df.isnull().sum()
df['Date Of Joining'] = pd.to_datetime(df['Date Of Joining'])
df_missing_percent = df.isnull().sum()/len(df)
df_missing_percent
##Question 1
contingency_table = pd.crosstab(df['Location'], df['Overall Rating'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)
##Plot for Location vs Overall Rating
plt.figure(figsize=(20, 12))
ax = sns.countplot(data=df, x='Location', hue='Overall Rating')
for container in ax.containers:  
    heights = [bar.get_height() for bar in container]
    
    max_height = max(heights)
    max_index = heights.index(max_height)
    
    x = container[max_index].get_x() + container[max_index].get_width() / 2
    ax.plot([x - 0.2, x + 0.2], [max_height + 1, max_height + 1], color='red', lw=2)
    ax.text(
        x,
        max_height + 1.5,
        f"{int(max_height)}",
        ha="center",
        fontsize=12,
        color="black",
        weight="bold",
    )
plt.title('Location vs Overall Rating', fontsize=16)
plt.xlabel('Location', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(title='Overall Rating', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
##Heatmap for location vs overall rating
heatmap_data = pd.crosstab(df['Location'], df['Overall Rating'])
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,          
    fmt="d",             
    cmap="YlGnBu",       
    linewidths=0.5,      
    cbar_kws={'label': 'Frequency'} 
)
plt.title("Heatmap: Location vs Overall Rating", fontsize=16)
plt.xlabel("Overall Rating", fontsize=14)
plt.ylabel("Location", fontsize=14)
plt.tight_layout()
plt.show()
##Plot for "very dissatisfied" with respect to location
very_dissatisfied_data = df[df['Overall Rating'] == 'Very Dissatisfied']
plt.figure(figsize=(10, 6))
sns.countplot(data=very_dissatisfied_data, x='Location', palette='Reds')
plt.title('Frequency of "Very Dissatisfied" Ratings by Location', fontsize=16)
plt.xlabel('Location', fontsize=14)
plt.ylabel('Frequency of Very Dissatisfied Ratings', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
##Plot for "very satisfied" with respect to location
very_satisfied_data = df[df['Overall Rating'] == 'Very Satisfied']
plt.figure(figsize=(10, 6))
sns.countplot(data=very_satisfied_data, x='Location', palette='Reds')
plt.title('Frequency of "Very Satisfied" Ratings by Location', fontsize=16)
plt.xlabel('Location', fontsize=14)
plt.ylabel('Frequency of Very Satisfied Ratings', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
##Gender vs Overall Rating
contingency_table = pd.crosstab(df['Gender'], df['Overall Rating'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)
##Plot for Gender vs Overall Rating   
plt.figure(figsize=(20,12))
sns.countplot(data=df, x='Gender', hue='Overall Rating')
plt.title('Gender vs Overall Rating', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(title='Overall Rating', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  
plt.show()
##Plot for "Very Satisfied" with respect to Gender
very_satisfied_data = df[df['Overall Rating'] == 'Very Satisfied']
plt.figure(figsize=(10, 6))
sns.countplot(data=very_satisfied_data, x='Gender', palette='Reds')
plt.title('Frequency of "Very Satisfied" Ratings by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Frequency of Very Satisfied Ratings', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
##Plot for "Very Dissatisfied" with respect to Gender
very_dissatisfied_data = df[df['Overall Rating'] == 'Very Dissatisfied']
plt.figure(figsize=(10, 6))
sns.countplot(data=very_dissatisfied_data, x='Gender', palette='Reds')
plt.title('Frequency of "Very Dissatisfied" Ratings by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Frequency of Very Dissatisfied Ratings', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
##Location vs Career Growth
contingency_table = pd.crosstab(df['Location'], df['Career Growth'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)
##Plot for location vs Career Growth
plt.figure(figsize=(20,12))
sns.countplot(data=df, x='Location', hue='Career Growth')
plt.title('Location vs Career Growth', fontsize=16)
plt.xlabel('Location', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(title='Career Growth', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  
plt.show()
##Location vs Career Growth
contingency_table = pd.crosstab(df['Location'], df['Compensation and Benefits'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)
##Plot for Location vs Compensation and Benefits
plt.figure(figsize=(20,12))
sns.countplot(data=df, x='Location', hue='Compensation and Benefits')
plt.title('Location vs Compensation and Benefits', fontsize=16)
plt.xlabel('Location', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(title='Compensation and Benefits', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  
plt.show()
##Location vs Work-life balance

contingency_table = pd.crosstab(df['Location'], df['Work-life Balance'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)
##Plot for Location vs Work-lief balance

plt.figure(figsize=(20,12))
sns.countplot(data=df, x='Location', hue='Work-life Balance')
plt.title('Location vs Work-life Balance', fontsize=16)
plt.xlabel('Location', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(title='Work-life Balance', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()  
plt.show()
##Label Encoding for Overall Rating, Career Growth, Compensation and Benefits, Work-life Balance
rating_map = {
    "Very Dissatisfied": 1,
    "Dissatisfied": 2,
    "Neutral": 3,
    "Satisfied": 4,
    "Very Satisfied": 5
}
rating_cols = ['Overall Rating', 'Career Growth', 'Compensation and Benefits', 'Work-life Balance']
for col in rating_cols:
    df[col] = df[col].map(rating_map)
##Pivot table for Overall Ratings by Location and Gender
pivot_table = df.pivot_table(values='Overall Rating', index='Location', columns='Gender', aggfunc='mean')
print("Pivot Table: Overall Ratings by Location and Gender")
print(pivot_table)
##Question 2 - calculation of correlation of overall rating w.r.t Career Growth, Compensation and Benefits, Work-life Balance
spearman_corr = df[['Career Growth', 'Compensation and Benefits', 'Work-life Balance', 'Overall Rating']].corr(method='spearman')
print("Spearman Correlation:")
print(spearman_corr)
##Plot for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Spearman Correlation Heatmap")
plt.show()
##Career Growth vs Gender
contingency_table = pd.crosstab(df['Career Growth'], df['Gender'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)
##Plot for Career Growth vs Gender
reverse_rating_map = {v: k for k, v in rating_map.items()}
for col in rating_cols:
    df[f"{col}_decoded"] = df[col].map(reverse_rating_map)
plt.figure(figsize=(20, 12))
sns.countplot(data=df, x='Gender', hue='Career Growth_decoded') 
plt.title('Gender vs Career Growth', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Career Growth', fontsize=14)
plt.legend(title='Career Growth', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  
plt.show()

##Overall Rating vs Gender
contingency_table = pd.crosstab(df['Overall Rating'], df['Gender'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)
##Plot forOverall Rating vs Gender
reverse_rating_map = {v: k for k, v in rating_map.items()}
for col in rating_cols:
    df[f"{col}_decoded"] = df[col].map(reverse_rating_map)
plt.figure(figsize=(20, 12))
sns.countplot(data=df, x='Gender', hue='Overall Rating_decoded')  
plt.title('Gender vs Overall Rating', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Overall Rating', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  
plt.show()

df['Level'] = df['Level'].str.strip().str.title() 
##Level vs Gender
contingency_table = pd.crosstab(df['Level'], df['Gender'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)
##Plot for Gender Distribution by Level
sns.countplot(x='Level', hue='Gender', data=df)
plt.title('Gender Distribution by Level')
plt.xlabel('Level')
plt.ylabel('Count')
plt.show()
##Question 3 - To calculate which category has more dissatisfaction percentage
df['Date Of Joining'] = pd.to_datetime(df['Date Of Joining'], format='%d-%b-%y', errors='coerce')
df['Duration'] = (datetime.now() - df['Date Of Joining']).dt.days / 365.25
df['Class'] = df['Duration'].apply(
    lambda x: 'High Tenured (15+ Years)' if x >= 15 else ('Covid Joiner(2020+)' if x <= 4 else 'Others')
)
df['Very Dissatisfied'] = df['Overall Rating'] == 1
df['Dissatisfied'] = df['Overall Rating'] == 2
df['Dissatisfied_Associates'] = df['Very Dissatisfied'] | df['Dissatisfied']
total_associates = df['EmpID'].count()
total_dissatisfied = df['Dissatisfied_Associates'].sum()
dissatisfaction_percentage = (total_dissatisfied / total_associates) * 100
very_dissatisfied_percentage = (df['Very Dissatisfied'].sum() / total_associates) * 100
dissatisfied_percentage = (df['Dissatisfied'].sum() / total_associates) * 100
print(f"Overall Dissatisfaction Percentage: {dissatisfaction_percentage:.2f}%")
print(f"Very Dissatisfied Percentage: {very_dissatisfied_percentage:.2f}%")
print(f"Dissatisfied Percentage: {dissatisfied_percentage:.2f}%")
dissatisfaction_summary = df.groupby('Class').agg(
    Total_Associates=('EmpID', 'count'),
    Very_Dissatisfied_Associates=('Very Dissatisfied', 'sum'),
    Dissatisfied_Associates=('Dissatisfied', 'sum'),
    Dissatisfied_Associates_Combined=('Dissatisfied_Associates', 'sum')
).reset_index()
dissatisfaction_summary['Very_Dissatisfied_Rate'] = (dissatisfaction_summary['Very_Dissatisfied_Associates'] / dissatisfaction_summary['Total_Associates']) * 100
dissatisfaction_summary['Dissatisfied_Rate'] = (dissatisfaction_summary['Dissatisfied_Associates'] / dissatisfaction_summary['Total_Associates']) * 100
dissatisfaction_summary['Dissatisfaction_Rate'] = (dissatisfaction_summary['Dissatisfied_Associates_Combined'] / dissatisfaction_summary['Total_Associates']) * 100

aggregated_summary = {
    'Total_Associates': total_associates,
    'Total_Dissatisfied_Associates': total_dissatisfied,
    'Dissatisfaction_Percentage': dissatisfaction_percentage,
    'Very_Dissatisfied_Percentage': very_dissatisfied_percentage,
    'Dissatisfied_Percentage': dissatisfied_percentage
}
dissatisfaction_summary
#Question 4 - Sentiment Analysis
df['Comments'] = df['Comments'].astype(str)
def sentimentanalysis(comment):
    blob = TextBlob(comment)
    return blob.sentiment.polarity
df['Sentiment Polarity'] = df['Comments'].apply(sentimentanalysis)
df['Categorry of sentiment'] = df['Sentiment Polarity'].apply(
    lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
)
sentiment_summary = df['Categorry of sentiment'].value_counts().reset_index()
sentiment_summary.columns = ['Sentiment', 'Count']
print(sentiment_summary)
##Pie chart for sentiment analysis
sentiment_summary = df['Categorry of sentiment'].value_counts().reset_index()
sentiment_summary.columns = ['Sentiment', 'Count']
plt.figure(figsize=(2,2))
plt.pie(sentiment_summary['Count'], labels=sentiment_summary['Sentiment'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999', '#99ff99'])
plt.title('Sentiment Distribution', fontsize=16)
plt.axis('equal')  
plt.show()

df['Comments'] = df['Comments'].astype(str)
words_to_remove = {'good', 'life', '0', 'better', 'job', 'happy', 'work', 'place', 'company',
                   '\n', 'companyx', 'nothing', 'working', 'like', 'great', 'career'}
stop_words = list(set(stopwords.words('english')).union(words_to_remove))

def preprocessing(text):
    text = text.lower() 
    text = text.translate(str.maketrans('', '', string.punctuation))  
    return text

df['Processed_Comments'] = df['Comments'].apply(preprocessing)
vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=stop_words)
X = vectorizer.fit_transform(df['Processed_Comments'])
ngram_counts = X.toarray().sum(axis=0)
ngram_list = vectorizer.get_feature_names_out()
ngram_freq = dict(zip(ngram_list, ngram_counts))
top_10_themes = Counter(ngram_freq).most_common(10)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_freq)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Top Themes", fontsize=16)
plt.show()
print("Top 10 Themes:")
for theme, count in top_10_themes:
    print(f"{theme}: {count}")

