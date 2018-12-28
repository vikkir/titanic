#Created 28/12/18 by Vikki Richardson
#importing libraries for dealing with csv and pandas

import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid',color_codes=True)

###################################################

#BRING IN THE TEST FILE
#creates a dataframe with the training info
titanicDF = pd.read_csv('train.csv', index_col = "PassengerId")

#####FUNCTIONS#####################################

def ageDefine(x):
    if x['Age'] < 16: 
        return 'Child'
    elif x['Age'] <= 50:
        return 'Adult'
    elif x['Age'] > 50:
        return 'Elderly'

def hasFamily(x):
    if x['Parch'] == 0 & x['SibSp'] == 0:
        return 'Alone'
    else:
        return 'Travelling with Family'

####DESCRIBE THE DATA  ################################

#find missing values
print(titanicDF.isnull().sum())
#age has a lot of missing values so try to fill those in

#get the average age
meanAge = titanicDF['Age'].mean()
print(meanAge)

#find all the entries with no age whose name indicates they are children and make them 10 years old
titanicDF.loc[titanicDF['Age'].isnull() & (titanicDF['Name'].str.contains('Miss') | titanicDF['Name'].str.contains('Master')), 'Age'] = 10
#now find all others with no age and make them the average age
titanicDF.loc[titanicDF['Age'].isnull(), 'Age'] = meanAge

#check the missing values now
print(titanicDF.isnull().sum())

#####GENDER VISUALISATIONS################

#bar chart for gender on the titanic
genderBarChart = sns.countplot(x='Sex', data=titanicDF)
plt.title('Gender of Passengers on Board')
plt.xlabel('Gender')
plt.ylabel('Count')
genderFig = genderBarChart.get_figure()
genderFig.savefig("GenderBarChart.png")
plt.show()
plt.close()

#bar chart for survivors by gender on the titanic
survivorGenderBarChart = sns.countplot(x='Survived', data=titanicDF, hue='Sex')
survivorGenderBarChart.set(xticklabels=['No','Yes'])
plt.title("Survived By Gender")
plt.xlabel("Survived")
survivorGenderFig = survivorGenderBarChart.get_figure()
survivorGenderFig.savefig("SurvivorGenderBarChart.png")
plt.show()
plt.close()

######AGE VISUALISATIONS####################

#create age category to seperate into child, adult and elderly
titanicDF['AgeCategory'] = titanicDF.apply(ageDefine, axis=1)

#bar chart for passengers on board by age
ageBarChart = sns.countplot(x='AgeCategory', data=titanicDF)
plt.title("Age category of passengers")
plt.xlabel('Age Category')
plt.ylabel("Count")
ageFig = ageBarChart.get_figure()
ageFig.savefig("AgeBarChart.png")
plt.show()
plt.close()


#bar chart for survivors by age
survivorAgeBarChart = sns.countplot(x='Survived', data=titanicDF, hue='AgeCategory')
survivorAgeBarChart.set(xticklabels=['No','Yes'])
plt.title("Survived By Age")
plt.xlabel("Survived")
survivorAge = survivorAgeBarChart.get_figure()
survivorAge.savefig("SurvivorAgeBarChart.png")
plt.show()
plt.close()

#box plot for actual age to check for outliers
ageBoxPlot = sns.boxplot(x=titanicDF['Age'], data=titanicDF)
plt.title("Age")
plt.xlabel('Age')
ageBoxFig = ageBoxPlot.get_figure()
ageBoxFig.savefig("AgeBoxPlot.png")
plt.show()
plt.close()

#histogram to check for normal distribution
ageHist = sns.distplot(titanicDF['Age'])
plt.title("Age of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
agefig = ageHist.get_figure()
agefig.savefig("AgeHistogram.png")
plt.show()
plt.close()

#####RELATIVES INFORMATION#########

titanicDF['Alone'] = titanicDF.apply(hasFamily, axis=1)

aloneBarChart = sns.countplot(x='Alone', data=titanicDF)
plt.title("Passengers Travelling with Family or Alone")
#plt.xlabel('Relatives')
plt.ylabel("Count")
#parchFig = parchBarChart.get_figure()
#parchFig.savefig("AgeBarChart.png")
plt.show()
plt.close()

survivorAloneBarChart = sns.countplot(x='Survived', data=titanicDF, hue="Alone")
plt.title("Passengers Travelling with Family or Alone")
#plt.xlabel('Relatives')
plt.ylabel("Count")
#parchFig = parchBarChart.get_figure()
#parchFig.savefig("AgeBarChart.png")
plt.show()
plt.close()

##########CLASS INFORMATION#########

classBarChart = sns.countplot(x='Pclass', data=titanicDF)
plt.title("Passengers By Class of Cabin")
plt.xlabel('Class of Passengers')
plt.ylabel("Count")
#parchFig = parchBarChart.get_figure()
#parchFig.savefig("AgeBarChart.png")
plt.show()
plt.close()

survivorClassBarChart = sns.countplot(x='Survived', data=titanicDF, hue="Pclass")
plt.title("Survivors by Class")
#plt.xlabel('Relatives')
plt.ylabel("Count")
#parchFig = parchBarChart.get_figure()
#parchFig.savefig("AgeBarChart.png")
plt.show()
plt.close()

#histogram to check for normal distribution
fareHist = sns.distplot(titanicDF['Fare'])
plt.title("Fare paid")
plt.xlabel("Fare")
plt.ylabel("Frequency")
#agefig = ageHist.get_figure()
#agefig.savefig("AgeHistogram.png")
plt.show()
plt.close()

corr = titanicDF.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#labels = ['PropGenFunding','PropGovtFunding','NoOfTweets']
sns.heatmap(corr, mask=mask,cmap=cmap, vmax=.3,linewidths=.5)
plt.rcParams["axes.labelsize"] = 6
plt.title('Diagonal correlation matrix for Final Data Set',fontsize=10)
ax = plt.gca()
#ax.set_xticklabels(labels)
#ax.set_yticklabels(labels)
plt.tight_layout()
plt.show()
plt.close()