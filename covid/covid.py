
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
from rank_bm25 import BM25Okapi
from sklearn.metrics import euclidean_distances, pairwise_distances
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df_covid=pd.read_pickle('/Users/iqbal/Desktop/coviddata.pkl') 
df_covid.dropna(inplace=True)
  
number_of_article = 1000 # 12500
df_covid = df_covid.head(number_of_article)
 

import re
df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
 
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))
df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))
   
text = df_covid.drop(["paper_id", "abstract", "abstract_word_count", "body_word_count", "authors", "title", "journal", "abstract_summary"], axis=1)

 
tokenized_corpus = [doc.split(" ") for doc in text["body_text"]]
bm25 = BM25Okapi(tokenized_corpus)

def get_matrix(fname,bm25):
    f=open(fname,'r')
    keywords=f.read()
    tokenized_query = keywords.split(",")
    tokenized_query=[ lower_case(a.strip()) for a in tokenized_query]
    print(tokenized_query)
    doc_scores1 = bm25.get_scores(tokenized_query)
    D=pairwise_distances(np.array(doc_scores1).reshape(-1,1))
    f.close()
    return D

D1=get_matrix("dist1keywords_transmission.txt", bm25)

D2=get_matrix("dist2keywords_risk.txt", bm25)

D3=get_matrix("dist3keywords_genetics.txt", bm25)

\

text_arr = text.stack().tolist()
len(text_arr)
 
words = []
for ii in range(0,len(text)):
    words.append(str(text.iloc[ii]['body_text']).split(" "))
   
n_gram_all = []

for word in words:
    # get n-grams for the instance
    n_gram = []
    for i in range(len(word)-2+1):
        n_gram.append("".join(word[i:i+2]))
    n_gram_all.append(n_gram)
 
from sklearn.feature_extraction.text import HashingVectorizer

# hash vectorizer instance
hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)

# features matrix X
X = hvec.fit_transform(n_gram_all)
 
from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print("X_test size:", len(X_test), "\n")
 
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1, perplexity=5)
X_embedded = tsne.fit_transform(X_train)

from matplotlib import pyplot as plt
import seaborn as sns
 
sns.set(rc={'figure.figsize':(15,15)})
 
palette = sns.color_palette("bright", 1)
 
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)

plt.title("t-SNE Covid-19 Articles")
 
plt.show()
 
from sklearn.cluster import KMeans

k = 10
kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)
y_pred = kmeans.fit_predict(X_train)
 
y_train = y_pred
 
y_test = kmeans.predict(X_test)

 
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered")
# plt.savefig("plots/t-sne_covid19_label.png")
plt.show()
