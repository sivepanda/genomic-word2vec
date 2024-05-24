from numpy import triu
from gensim.models import Word2Vec
import numpy as np
import os

# Create a sequence mapping so that we can re-match the sequence to its corresponding location
sequence_map = dict()

# Create a list of lists containing a list of "sentences" (genes that code similar tissue) 
all_sequences = []

# Project structure contains a dir named "fasta" that contains .FASTA sequences 
for filename in os.listdir('./fasta'):
    sequences = [] # Contruct a new "sentence" for each set of related sequence
    f = os.path.join('./fasta', filename)
    if (os.path.isfile(f)):
        key = ''
        with open(f, 'r') as file:
            for index, line in enumerate(file):
                if (index % 2 != 0):
                    sequences.append(line.strip())
                    sequence_map[line.strip()] = key + '_' + filename[:-3]
                else:
                    key = line.strip()
    all_sequences.append(sequences)

# Train model
model = Word2Vec(sentences=all_sequences, min_count=1, sg=1)


def get_region_set_embedding(rs, mdl):
    vectors = [mdl.wv[region] for region in rs if region in mdl.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(mdl.vector_size)

# For a list of new regions, place it into the vector space of the model, and return the closest neighbors
def get_region_similar_embedding(new_region, mdl, tn):
    mdl.build_vocab([new_region], update=True)
    return mdl.wv.most_similar(new_region, topn=tn)

runset = []
with open('./seq.txt', 'r') as file:
    runset = file.readlines()

s = get_region_similar_embedding(runset, model, 2)

print(sequence_map[s[0][0]])
