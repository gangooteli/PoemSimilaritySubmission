from gensim.models import TfidfModel
from gensim import corpora, similarities
import io, re, spacy, os
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim.models.ldamodel import LdaModel


############################# Reading poems #############################

poem_dir = 'poem'
documents = []

docMap_dict = {}

i=0

files = os.listdir(poem_dir)

for file in files:
	with io.open(poem_dir+'/'+file, errors='ignore',encoding='utf-8') as fid:
 		txt = fid.read()
		#preprocessing
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
#Bag of words representation of documents
corpus = [dictionary.doc2bow(document) for document in documents]

#print "Number of unique tokens", len(dictionary)
#print "Number of documents", len(corpus)

#data_dir = 'dataAnalysis' 

#corpus_name = 'corpus1'
#dictionary_name = 'dictionary1.txt'
#we can save and load dictionary to reduce time while re-execution
#dictionary.save_as_text('dataAnalysis/dictionary1.txt', sort_by_word = True)
#dictionary = corpora.Dictionary.load_from_text(data_dir+'/'+dictionary_name)


#we can save and load corpus to reduce time while re-execution

#from gensim.corpora.mmcorpus import MmCorpus
#save corpus
#MmCorpus.serialize('dataAnalysis/corpus1', corpus)
#corpus = corpora.MmCorpus(data_dir+'/'+ corpus_name)


############################# Create TF-IDF Model #############################


#initialize tfidf Model
tfidfModel = TfidfModel(corpus)

#convert corpus to tfidf
corpus_tfidf = tfidfModel[corpus]


############################# Create LSI Model #############################


#intialize lsi Model
lsi = LsiModel(corpus, id2word=dictionary, num_topics=10)




############################# Create LDA Model #############################

num_topics=10

#initialise lda Model
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=10, passes=50)



############################# Create Similarity Matrix Using TfIdf, LSI, LDA corpus #############################

#similarity
#create tfidf similarity matrix of tfidf corpus created above 
poemTfidfSimilarityIndex = similarities.MatrixSimilarity(corpus_tfidf)

#create LSI similarity matrix of corpus created above 
poemLsiSimilarityIndex = similarities.MatrixSimilarity(lsi[corpus])

#create LDA similarity matrix of corpus created above
poemLdaSimilarityIndex = similarities.MatrixSimilarity(lda[corpus])

#save and load operation to reduce time while re-execution
#index.save('poemTfidfSimilarityIndex.index')
#index = similarities.MatrixSimilarity.load('poemTfidfSimilarityIndex.index')


############################# Take input and calculate TfIDF, LSI, LDA Similarity #############################

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
		##########similarity for i document##############
		doc_bow = dictionary.doc2bow(documents[i])

		doc_tfidf = tfidfModel[doc_bow]
		sims_tfidf = poemTfidfSimilarityIndex[doc_tfidf]
		sims_tfidf = sorted(enumerate(sims_tfidf), key=lambda item: -item[1])

		doc_lsi = lsi[doc_bow]
		sims_lsi = poemLsiSimilarityIndex[doc_lsi]
		sims_lsi = sorted(enumerate(sims_lsi), key=lambda item: -item[1])
		
		doc_lda = lda[doc_bow]
		sims_lda = poemLdaSimilarityIndex[doc_lda]
		sims_lda = sorted(enumerate(sims_lda), key=lambda item: -item[1])

		print "Similarity Using TFIDF model"
		for i, sim in enumerate(sims_tfidf):
			a , b = sim
			print "Document no.", list(docMap_dict.keys())[list(docMap_dict.values()).index(a)], " score: ", b
			if(i>=(n-1)):
				break;
		print "##########################################################"
		print "Similarity Using LSI model"
		for i, sim in enumerate(sims_lsi):
			a , b = sim
			print "Document no.", list(docMap_dict.keys())[list(docMap_dict.values()).index(a)], " score: ", b
			if(i>=(n-1)):
				break;

		print "##########################################################"
		print "Similarity Using LDA model"
		for i, sim in enumerate(sims_lda):
			a , b = sim
			print "Document no.", list(docMap_dict.keys())[list(docMap_dict.values()).index(a)], " score: ", b
			if(i>=(n-1)):
				break;
	
	print "want to check again (y/n)"
	choice = raw_input()

################################################# End #################################################
		