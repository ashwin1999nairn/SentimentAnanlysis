import pandas as pd
import re
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report



#READING AND RE-ORGANISATION OF DATA

data = open('C:/Users/Ashwin/Downloads/corpus', encoding='utf-8').read()
labels , texts = [], []
for i, line in enumerate(data.split('\n')):
    info = line.split()
    labels.append(info[0])
    texts.append("".join(info[1:]))



# CREATING A DATAFRAME USING PANDAS

dataframe = pd.DataFrame()
dataframe['texts'] = texts
dataframe['label'] = labels
print(dataframe.shape)

sns.countplot(dataframe.label)


def custom_encoder(df):
       df.replace(to_replace ="__label__1", value =0, inplace=True)
       df.replace(to_replace ="__label__2", value =1, inplace=True)

custom_encoder(dataframe)





# SPLITTING THE DATASET INTO TRAINING AND TESTING SETS



trainDF = dataframe.iloc[:8000,:]
testDF = dataframe.iloc[2000:,:]


# PREPROCESSING THE DATASETS USING VARIOUS  CLEANSING AND NORMALISATION TECHNIQUES

lem = WordNetLemmatizer()
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lem.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

corp1 = text_transformation(trainDF['texts'])

word_cloud = ""
for row in corp1:
    for word in row:
        word_cloud+=" ".join(word)
wordcloud = WordCloud(width = 1000, height = 500,background_color ='white',min_font_size = 10).generate(word_cloud)
plt.imshow(wordcloud)



cv = TfidfVectorizer(ngram_range=(1,2))
traindata = cv.fit_transform(corp1)

train_label = trainDF.label

def trainmodel(classifier , text, label):
    classifier.fit(text,label)

nb = naive_bayes.MultinomialNB()

trainmodel(nb,traindata,train_label)

corp2 = text_transformation(testDF['texts'])
testdata = cv.transform(corp2)

prediction = nb.predict(testdata)
accuracy = metrics.accuracy_score(testDF.label,prediction)
precision = metrics.precision_score(testDF.label,prediction)
recall = metrics.recall_score(testDF.label,prediction)

print("Accuracy of the model : ", accuracy)
print("Precision_score of the model :" , precision)
print("recall_score of the model : ", recall)

plot_confusion_matrix(testDF.label,prediction)
cr = classification_report(testDF.label,prediction)
print(cr)
























def expression_check(prediction_input):
    if prediction_input == 0:
        print("Input statement has Negative Sentiment.")
    elif prediction_input == 1:
        print("Input statement has Positive Sentiment.")
    else:
        print("Invalid Statement.")


def sentiment_predictor(input):
    print('input statement is : ',input)
    input = text_transformation(input)
    transformed_input = cv.transform(input)
    prediction = nb.predict(transformed_input)
    expression_check(prediction)
    


ans = 'yes'

while(ans == 'yes'):
    print('Enter your review : ')
    input_review = input()
    li = []
    li.append(input_review)
    sentiment_predictor(li)

    print('Would you like to enter more reviews ? (yes/no)')
    ans = input()
    if ans == 'no':
        print('exit succeesful')





# input2 = ["This product is the worst."]
# input3 = ['this is a good book.']

# sentiment_predictor(input2)
# sentiment_predictor(input3)