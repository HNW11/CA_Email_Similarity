from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#EXPLORING THE DATA
#emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'])
#print(emails.target_names)
#print(emails.data[5])
#print(emails.target[5]) #1
#print(emails.target_names) #Hockey

#MAKING THE TRAINING AND TEST SETS
#Training set
train_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware','rec.sport.hockey', 'sci.crypt', 'talk.religion.misc'], subset='train', shuffle=True, random_state=108)
#Test Set 
test_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware','rec.sport.hockey', 'sci.crypt', 'talk.religion.misc'], subset='test', shuffle=True, random_state=108)
#Transform into list of word counts
counter = CountVectorizer()
#Tell counter what possible words can exist
counter.fit(test_emails.data + train_emails.data)
#Make a list of the counts of words in training
train_counts = counter.transform(train_emails.data)
#Make a list of the counts of words in test
test_counts = counter.transform(test_emails.data)

#MAKING A NAIVE BAYES CLASSIFIER
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
#Test Naive Bayes Classifier
print(classifier.score(test_counts, test_emails.target))


#TESTING OTHER DATASETS
#The classifier was 99% accurate when trying to classify hockey and tech emails.

#Percentage went down when I added 2 more categories 