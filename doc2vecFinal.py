import gensim.models.word2vec as w2v
import numpy as np
import pandas as pd
import spacy, os, re, io
import multiprocessing
from gensim.corpora import Dictionary




############################# Reading poems #############################
poem_dir = 'poem'
documents = []

docMap_dict = {}

i=0

files = os.listdir(data_dir)
for file in files:
	with io.open(poem_dir+'/'+file, errors='ignore',encoding='utf-8') as fid:
		txt = fid.read()
		txt = re.sub('\s', ' ', txt)
		documents.append(txt)
		docMap_dict[int(file)] = i
		i +=1

############################# Preprocessing Poems #############################

#create word_list
nlp = spacy.load('en')
processed_docs = []
for document in nlp.pipe(documents, n_threads=4, batch_size=10):
	ents = document.ents
	document = [token.lemma_ for token in document if token.is_alpha and not token.is_stop and len(token)>2]
	document.extend([str(entity) for entity in ents if len(entity)>1])
	processed_docs.append(document)

#each poem is processed to document which contains list of words
#documents contains list of list of words
documents = processed_docs
del processed_docs

############################# Create Dictionary #############################

#Dictionary contains the vocab/word list of the all poems along with its number of total number of occurrances

dictionary = Dictionary(documents)
max_freq = 0.5
min_wordcount = 5
dictionary.filter_extremes(no_below = min_wordcount, no_above = max_freq)

_ = dictionary[0]

############################# Create Corpus #############################
#Bag of words representation of documents/poems

corpus = [dictionary.doc2bow(document) for document in documents]


#print "len of words/vocab", len(dictionary)
#print "len of corpus", len(corpus)
#print "number of documents", len(documents)

############################# Create Word2vec #############################

#Hyper Parameters for Word2vec Model
seed = 1
num_features = 5
num_workers = multiprocessing.cpu_count()
min_word_Count = 4
context_size = 7
downsampling = 1e-3


#Initialize Word2vec Model
word2vec = w2v.Word2Vec(
	sg=1, 
	seed = seed, 
	workers = num_workers, 
	size=num_features, 
	min_count = min_word_Count,
	sample = downsampling
	)

#Build vocab for Word2Vec
word2vec.build_vocab(documents)
#print "Word2Vec vocabulary length:", len(word2vec.wv.vocab)

#Train Word2Vec Model
word2vec.train(documents)


#We can also save the model and use it for later use
#save the model
#if not os.path.exist("peoms2vecModel"):
#	os.makedirs("poems2vecModel")
#word2vec.save(os.path.join("poems2vecModel", "word2vec.w2v"))

#sample command to check word2vec results
#print word2vec.most_similar("love")
#print word2vec.most_similar("death")
#print word2vec.most_similar("night")
#print word2vec.most_similar("day")

#Similarity between two word using Word2vec Model
#print word2vec.similarity("day", "night")
#print word2vec.similarity("day", "tree")

#Sample operations which can be performed using Word2Vec
#print word2vec.wv['day']
#print "a", word2vec.wv['day']*word2vec.wv['night']
#print "b", np.sqrt(word2vec.wv['day'].T.dot(word2vec.wv['day']))
#print "c", (word2vec.wv['day']+word2vec.wv['night'])/2


################################ Create Doc2vec using Word2vec #################################

#For each document, calculated weighted mean of all word2vec vector of words present in the document to create Doc2vec


documents2vec = []

#create doc2Vec for each poem
for document in documents:
	document2vec = np.array([])
	total_frequency = 0.
	for word_id, frequency in dictionary.doc2bow(document):
		wordVec = word2vec.wv[dictionary.id2token.get(word_id)]
		if len(document2vec)==0:
			document2vec = wordVec*frequency
		else:
			document2vec += wordVec*frequency
		total_frequency +=frequency
	document2vec = document2vec/total_frequency
	documents2vec.append(document2vec)


################################### Create Cosine Similarity Matrix ################################

#Precompute Cosine Similarity Matrix and store it in memory to give similarity for a poem/documents
length = len(documents2vec)
csm = np.eye(length, length)

absolute_vec = [np.sqrt(a.T.dot(a)) for a in documents2vec]

for i in range(1,length):
	for j in range(i):
		#print i, j
		a = documents2vec[i]
		b = documents2vec[j]
		numer = a.T.dot(b)
		denom = absolute_vec[i] * absolute_vec[j]
		csm[i][j] = numer/denom
		csm[j][i] = csm[i][j]

################################ Take input and calculate Doc2Vec Similarity ###########################

sim_score = []

choice = 'yes'
while choice=='yes' or choice=='y' :
	print "enter poem number either between [1-35] or [40-69] for which you want to find similarity"
	input1 = int(raw_input())
	if docMap_dict.has_key(input1) == False:
		print " please enter poem number either between 1-35 or 40-69. Your input was out of index"
	else:
		print "enter vale of N for top documents similar to input"
		n = min(int(raw_input()), 64)
		i = docMap_dict[input1]
		print documents[i]
		#similarity for i document
		a = documents2vec[i]
		abs = np.sqrt(documents2vec[i].T.dot(documents2vec[i]))
		sim_score = csm[i]
		sim_score = sorted(enumerate(sim_score), key=lambda item: -item[1])

		for i, sim in enumerate(sim_score):
			a , b = sim
			print "Document no.", list(docMap_dict.keys())[list(docMap_dict.values()).index(a)], " score: ", b
			if(i>=(n-1)):
				break;
	print "want to check again (y/n)"
	choice = raw_input()
		