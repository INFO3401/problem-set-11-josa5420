import numpy as np
import pandas as pd
import scipy

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sb
import operator

import altair as alt
alt.renderers.enable('notebook')

from vega_datasets import data


# Extract the data. Return both the raw data and dataframe
def generateDataset(filename):
    data = pd.read_csv(filename)
    df = data[0:]
    df = df.dropna()
    return data, df

def runTTest(ivA, ivB, dv):
    ttest = scipy.stats.ttest_ind(ivA[dv],ivB[dv])
    print(ttest)
    
def runAnova(data, formula):
    model = ols(formula, data).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    print(aov_table)
    
def dic(c):
    d = {}
    for i in c:
        try:
            d[i]+=1
        except:
            d[i]=1
    import operator
    d = sorted(d.items(), key=operator.itemgetter(1),reverse = True)
    return d

a = pd.read_csv("simpsons_paradox.csv")
dep = a.groupby('Gender').agg({'Admitted':np.sum,'Rejected':np.sum})
gender = a.groupby('Department').agg({'Admitted':np.sum,'Rejected':np.sum})
gender = gender.reset_index()

"""
1 What statistical test would you use for the following scenarios? 

    (a) Does a student's current year (e.g., freshman, sophomore, etc.) effect their GPA?
    
ANOVA

    (b) Has the amount of snowfall in the mountains changed over time? 
    
Generalized Regression

    (c) Over the last 10 years, have there been more hikers on average in Estes Park in the spring or summer? 
    
Ordinal Logistic Regression

    (d) Does a student's home state predict their highest degree level?
    
Chi-Squared

"""

"""

2 You've been given some starter code in class that shows you how to set up ANOVAs and Student's T-Tests in addition to the regression code from the last few weeks. Now, use this code to more deeply explore the simpsons_paradox.csv dataset. Compute new dependent variables that shows the percentage of students admitted and rejected for each row in the CSV. Use those rows to try to understand what significant correlations exist in this data. What factors appear to contribute most heavily to admissions? Do you think the admissions process is biased based on the available data? Why or why not?
"""

a['Percent Admitted'] = a['Admitted']/(a['Admitted']+a['Rejected'])
a['Percent Rejected'] = a['Rejected']/(a['Admitted']+a['Rejected'])
print(np.mean(a[a['Gender']=='Female']['Percent Admitted']))
print(np.mean(a[a['Gender']!='Female']['Percent Admitted']))

"""
Based strictly on the 'Percent Admitted' and 'Percent Rejected' columns, it appears as though the admissions process is biased towards favoring women. This is because, as printed above, the average percentages of accepted women is 3.6% higher than the rate of men.
"""

"""
3 There's a data quality issue hiding in the admissions dataset from Monday. Correct this issue and compare your new results. How are they the same? How do they differ?
"""
rawData, df = generateDataset('simpsons_paradox_bad.csv')
print("Does gender correlate with admissions?")
men = df[df['Gender']=='Male']#.append(df[df['Gender']=='Male '])
women = df[df['Gender']=='Female']
runTTest(men,women,'Admitted')

print("Does department correlate with admissions?")
simpleFormula = 'Admitted ~ C(Department)'
runAnova(rawData, simpleFormula)

print("Do gender and department correlate with admissions?")
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData, moreComplex)

# Above is bad csv, below is corrected

rawData, df = generateDataset('simpsons_paradox.csv')
print("Does gender correlate with admissions?")
men = df[df['Gender']=='Male']#.append(df[df['Gender']=='Male '])
women = df[df['Gender']=='Female']
runTTest(men,women,'Admitted')

print("Does department correlate with admissions?")
simpleFormula = 'Admitted ~ C(Department)'
runAnova(rawData, simpleFormula)

print("Do gender and department correlate with admissions?")
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData, moreComplex)

"""
In the csv file, several instances in the gender column for 'male' had a space after them. I went into the text file and corrected them.

In the results from the faulty dataset, we rejected the null hypothesis in the results of the complex ANOVA tests on department & gender correlation with admissions.

In the results from the correct dataset, we failed to reject the null hypothesis in the results of the complex ANOVA tests on department & gender correlation with admissions.

This difference represents a complete flip in the results. The null hypotheses should not be rejected, but in the faulty dataset they would have been. 
"""

"""
4 The data also represents an example of Simpson's Paradox. Use whatever visualization tools you'd like to illustrate the two possible perspectives. Make sure to include a screenshot of each and explain the perspective shown in each. 
"""

alt.Chart(a).mark_bar().encode( 
    x='Department',
    y='Percent Admitted',
    color = 'Gender'
) 

#This visulaization shows that females appear to have a higher admittance rate

plt.pie([np.sum(a[a['Gender']=='Female']['Admitted']/np.sum(a[a['Gender']=='Female']['Rejected'])),
                np.sum(a[a['Gender']=='Male']['Admitted']/np.sum(a[a['Gender']=='Male']['Rejected']))],
        labels=['Total Average Female Acceptance Rate','Total Average Male Acceptance Rate'])

#This visualization takes all of the numbers and sums them together, to show a different side: There is a disparity in acceptance rate based on gender.

"""
Some scholars contend that Shakespeare's early plays are actually collaborations that could be attributed to other authors. One such author is Christopher Marlow (1564-1593). Use a series of visualizations to compare data from Marlowe's plays and Shakespeare's plays. The zip file in Assignment Data contains CSV data looking at both sentiment and word counts comparing plays from the two authors. Use the data and Altair to work through the following problems (submit your notebook with additional documentation addressing each question): 

5 Build a visualization that allows you to compare the distribution of positive sentiment in both Marlowe and Shakespeare. What does this tell you about the styles of the two authors?

"""

sentiment = pd.read_csv('sentimentData.csv')
alt.Chart(sentiment).mark_point().encode( 
    x='Negative',
    y='Neutral'
    ,color = 'Author'
)

# This visualization shows us that Marlowe's plays appear to be less negative, while Shakespeare's plays appear to be less neutral. 

"""
6 Build a visualization that allows you to explore the correlation of word counts for Marlowe and Shakespeare. What does this tell you about the styles of the two authors? 
"""
counts = pd.read_csv('rawCounts.csv')

marlowe_d = {}
shakespeare_d = {}

for i in counts[counts['Author']=='Marlowe']['Count']:
    try:
        marlowe_d[i]+=1
    except:
        marlowe_d[i]=1
for i in counts[counts['Author']=='Shakespeare']['Count']:
    try:
        shakespeare_d[i]+=1
    except:
        shakespeare_d[i]=1
m_df = pd.DataFrame.from_dict(marlowe_d,orient='index').reset_index()
m_df['Author']=['Marlowe' for i in range(len(m_df))]
s_df = pd.DataFrame.from_dict(shakespeare_d,orient='index').reset_index()
s_df['Author']=['Shakespeare' for i in range(len(s_df))]
df = m_df.append(s_df)
df = df.rename(columns={'index':'Times a word Appears',0:'Frequency of word Appearance'})
alt.Chart(df).mark_point().encode( 
    x='Times a word Appears',
    y=alt.Y('Frequency of word Appearance',scale=alt.Scale(type='log'))
    ,color = 'Author'
)

"""
From this we are able to see that Shakespeare has more words that appear more frequently, although marlowe has both extremes: Marlowe has the highest count of words used only once, and uses a single word more times than the word shakespeare uses the most.
"""

"""
7 Generate three additional visualizations using this data (please use different visualization techniques for each visualization). What do these visualizations tell you about potential collaborations between Shakespeare and Marlowe?

"""
d={}
tf_df = pd.DataFrame(columns=['Text','TFIDF Avg','Author','Average Count'])
m = 0
s = 0
for i in counts['Text'].unique():
    a = counts[counts['Text']==i]
    d[i]=np.mean(a['TFIDF']*a['Count'])
    xxx = pd.DataFrame(data=[[str(i),float(d[i]),str(list(a['Author'])[0]),np.mean(a['Count'])]],columns=[str('Text'),str('TFIDF Avg'),str('Author'),'Average Count'])
    tf_df = tf_df.append(xxx)
alt.Chart(tf_df.sort_values('Author')).mark_bar().encode( 
   alt.X('Text', sort=alt.EncodingSortField(field="Author", op="count", order='ascending')),
y='TFIDF Avg'
,color = 'Author'
)  

"""
This graph actually shows how Shakespeare tends to be more consistent with the TFIDF Averages across his works than Marlowe, who tends to be more sporadic.
"""

plt.pie([np.mean(counts[counts['Author']=='Marlowe']['Count']),
         np.mean(counts[counts['Author']=='Shakespeare']['Count'])],
        labels=['Marlowe Average Word Count','Shakespeare Average Word Count'])

"""
This graph shows that Marlowe and Shakespeare have relatively similar averages of words in their texts.
"""
md={}
m=counts[counts['Author']=='Marlowe']
for i in range(len(m)):
    try:
        md[m.loc[i]['Word']]+=m.loc[i]['Count']
    except:
        md[m.loc[i]['Word']]=m.loc[i]['Count']
md = sorted(md.items(), key=operator.itemgetter(1),reverse = True)

sd={}
s=counts[counts['Author']=='Shakespeare'].reset_index()
for i in range(len(s)):
    try:
        sd[s.loc[i]['Word']]+=s.loc[i]['Count']
    except:
        sd[s.loc[i]['Word']]=s.loc[i]['Count']
sd = sorted(sd.items(), key=operator.itemgetter(1),reverse = True)
word_freqs = pd.DataFrame(columns=['Word','Position','Author'])
for i in range(20):
    word_freqs = word_freqs.append(pd.DataFrame(data=[[sd[i][0],i+1,'Shakespeare']],columns = ['Word','Position','Author']))
    word_freqs = word_freqs.append(pd.DataFrame(data=[[md[i][0],i+1,'Marlowe']],columns = ['Word','Position','Author']))
alt.Chart(word_freqs).mark_bar().encode( #mark_point
    x='Word',
    y='Position',
    color = 'Author'
)
"""
This graph shows two things: one, that Marlowe and Shakespeare share the same identical top twenty words between all of their plays, but also they are very similar in their ordering of top word.
"""
