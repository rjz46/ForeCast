import numpy as np
import pickle
import os
def getGloveEmbeddings(word):

        if os.path.isfile("embeddings.pkl"):
                embeddings = pickle.load( open( "embeddings.pkl", "rb" ) )
        else:
                
                path = "glove_word_embeds.txt"
                embeddings = {}
                toxicData = []
                with open(path,'r') as f:
                        toxicData  = f.readlines()

                for d in toxicData:
                        t = d.split(' ')
                        t[-1] = t[-1].strip('\n')
                        embeddings[t[0]] = np.array(t[1:], dtype=float)
                f = open("embeddings.pkl","wb")
                pickle.dump(embeddings,f)
                f.close()

        l = len(embeddings['and'])
        print(l)
        
        if word in embeddings:
                return embeddings[word]
        else:
                return np.zeros(l)
#print(getGloveEmbeddings('ans'))
