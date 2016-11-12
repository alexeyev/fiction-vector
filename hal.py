import gensim
import pymystem3
import nltk
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from pymystem3.mystem import Mystem

TEXT_FILE_PATH = "data/anya.txt"

size = 160
window = 4
min_count = 3
sg = 1  # 0 for CBOW, 1 for skip-gram
iter = 15

MODEL_FILE_PATH = TEXT_FILE_PATH + "-size" + str(size) + \
                  "-window" + str(window) + "-mincount" + str(min_count) + \
                  "-sg" + str(sg) + "-iter" + str(iter) + "-model.bin"

mystem = Mystem()

try:
    model = Word2Vec.load(TEXT_FILE_PATH)
except:  # todo

    print("Splitting...")

    splitted_sentences = []

    with open(TEXT_FILE_PATH, "r") as input_file:

        whole_text = input_file.read().replace("\n", " ").lower()
        whole_text_sentences = sent_tokenize(whole_text, language='english')

        for sent in whole_text_sentences:
            splitted_sentences.append(mystem.lemmatize(sent))

    print("Training...")

    model = Word2Vec(splitted_sentences,
                     iter=iter, sg=sg, size=size, window=window,
                     min_count=min_count, workers=4)

    print("Training done.")

    model.save(MODEL_FILE_PATH)

print(model.most_similar(["аня"]))
# print(model.index2word)
