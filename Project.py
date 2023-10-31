
'''#required for running the code on Google Colab
!pip install pyLDAvis
import pyLDAvis
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')'''

from ast import main
import pandas as pd
import glob
import nltk
import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensim_models
import numpy as np
import seaborn as sns
import csv
import re
import matplotlib.pyplot as plt
from this import d
from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from itertools import chain

'''# Collect data from datset based on keywords
keywords = ['indigenous', 'bipoc', 'racist', 'racists', 'racism', 'minority', 'minorities', 'facist', 'facists', 'white supremacist',
            'black people', 'blacks', 'immigrant', 'immigrants', 'white', 'whites', 'asian', 'asians','white supremacists', 'black',
            'xenophobic', 'xenophobia', 'non-white', 'non-whites', 'hispanic', 'hispanics', 'bigot', 'bigots', 'white privilege']
column = 2

with open('rawTweets.csv', 'r', encoding='utf-8') as input_file, open('keywordTweets.csv', 'w', newline='', encoding='utf-8') as output_file:
    # Create a CSV reader and writer
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    headers = next(reader)
    writer.writerow(headers)

    for row in reader:
        if any(keyword in row[column] for keyword in keywords):
            writer.writerow(row)'''

'''# fetch rows for manual analyzing
df = pd.read_csv("keywordTweets.csv")
df.head()
selected_rows = df.iloc[0::2]
selected_rows.to_csv('analyze.csv', index=False)'''

# load dataset
df = pd.read_csv("keywordTweets.csv", usecols=['text'])
df.head()

# clean dataset
df = df.drop_duplicates(subset='text', keep="first")
df = df.reset_index(drop=True)
df["clean_text"] = df["text"].apply(lambda x: re.sub('http[s]?://\S+', '', str(x))) # Remove links
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
df["clean_text"] = df["clean_text"].apply(lambda s: deEmojify(s)) # Remove emojis
df["clean_text"] = df["clean_text"].apply(lambda s: ' '.join(re.sub("[.,!?:;-='_]", " ", s).split())) # Remove punctuations
df["clean_text"] = df["clean_text"].apply(lambda s: ' '.join(re.sub("@[A-Za-z0-9_]+","",s).split()))  # Remove @
#df["clean_text"] = df["clean_text"].apply(lambda s: ' '.join(re.sub("#[A-Za-z0-9_]+","",s).split())) # Remove hashtags

# tokenize dataset
def tokenize(text: str) -> str:
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation]
    text = preprocess_string(text, CUSTOM_FILTERS)

    return text

doc_token = df['clean_text'].apply(lambda x: tokenize(x))

bigram = gensim.models.Phrases(doc_token, threshold=100, min_count=5)
doc_bigram = [bigram[doc_token[i]] for i in range(len(doc_token))]

# list of stop words
stopwords = stopwords.words('english')

# create lemmatizer object
lem = WordNetLemmatizer()

# lemmatize words if they are not stop words or short words
doc_list = []
for sent in doc_bigram:
    new_doc = [lem.lemmatize(word) for word in sent if word not in stopwords and len(word)>3]
    doc_list.append(new_doc)

'''# Plot word frequency
doc_list = list(chain.from_iterable(doc_list))
sns.set_style('darkgrid')
nlp_words=nltk.FreqDist(doc_list)
with open('Words.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for word, freq in nlp_words.items():
        writer.writerow([word, freq])
nlp_words.plot(100)'''

# preview of dataset
df['Text (Preprocessed)'] = doc_list
df.head()
print(df)

# create dictionary
dictionary = corpora.Dictionary(doc_list)

# create document term matrix
bow = [dictionary.doc2bow(text) for text in doc_list]

'''#TF-IDF method
tfIdf = models.TfidfModel(bow, normalize=True)
newBow = tfIdf[bow]'''

"""# compute the coherence scores for each number of topics
best_num = float('NaN')
best_score = 0
coherence_scores = []
for i in range(2,20):
    # create lda model with i topics
    lda = LdaModel(corpus=bow, num_topics=i, id2word=dictionary, random_state=42)

    # obtain the coherence score
    coherence_model = CoherenceModel(model=lda, texts=doc_list, dictionary=dictionary, coherence='c_v')
    coherence_score = np.round(coherence_model.get_coherence(),2)
    coherence_scores.append((i, coherence_score))
    if coherence_score > best_score:
        best_num = i
        best_score = coherence_score

df = pd.DataFrame(coherence_scores, columns=['Number of Topics', 'Coherence Score'])
df.to_csv('coherence_scores.csv', index=False)

print(f'The coherence score is highest ({best_score}) with {best_num} topics.')"""

'''# obtain the coherence score and plot the results
def compute_coherence_values(dictionary, corpus, texts, limit=20, start=2, step=1):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(cm.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=bow, texts=doc_list, start=2, limit=20, step=1)
# Show graph
limit=20; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num of Topics")
plt.ylabel("Coherence Score")
plt.legend(("coherence_values"), loc='best')
plt.savefig('myplot.png', dpi=300, bbox_inches='tight')
plt.show()'''

# build the lda model
lda_model = gensim.models.ldamodel.LdaModel(corpus=bow, id2word=dictionary, num_topics=6, random_state=42)

# show the words most strongly associated with each topic
for topic in lda_model.print_topics():
    print(topic)

# obtain topic distributions for each document
topic_dist = lda_model[bow]

# store distributions ina list
dist = []
for t in topic_dist:
    dist.append(t)

# add list to the data frame
df['Topic Distribution'] = topic_dist

# dataset preview
df[['clean_text', 'Topic Distribution']].head()
#df[['text', 'clean_text', 'Topic Distribution']].to_csv('analyze6.csv')
print(df)

#calculating model perplexity
perplexity = lda_model.log_perplexity(bow)
print(perplexity)

'''# visualize LDA model results
pyLDAvis.enable_notebook()
gensim_models.prepare(lda_model, dictionary=dictionary, corpus=bow)'''