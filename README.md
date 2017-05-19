# PoemSimilaritySubmission

The dataset contains list of peom/documents. task was that given a document will return a list of n most similar documents.
This project contains two implementation of above problem in python.
A.Doc2vec implementation:
  1. Train word2Vec model using corpus.
  2. Find weighted mean of word vetors in document to create doc2vec representation of the document/poem.
  3. Compute and create Cosine Similarity Matrix of all doc2vec vectors with others.
  4. For DocX, return top n documents who cosine score is top n with Doc X in the Cosine Similarity Matrix(CSM)
  
B.Mix Model implementation:
  1. Trained LDA, LSA, TF-IDF model using corpus of all documents.
  2. This will give LDA, LSA, TF_IDF representation of documents/poems.
  3. Compute and create Cosine Similarity Matrix of all doc2vec vectors with others.
  4. For DocX, return top n documents who cosine score is top n with Doc X in the Cosine Similarity Matrix(CSM)
  
  
 Results of tf-idf is good. LDA and LSI is not able to give good results. Same is the case with Doc2Vec
 
 Further Extension(Have Some Idea in Mind):
 Can create word2vec matrix of document by application of padding for different length of document to represent. 
 Then use this matrix to feed into multi layer CNN.
 Need to think more.
