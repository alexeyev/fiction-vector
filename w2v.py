from gensim.models.word2vec import Word2Vec
from nltk.tokenize import sent_tokenize
from pymystem3.mystem import Mystem

DATA_PATH = "data/"
MODEL_PATH = "models/"
TEXT_FILE_PATH = "anya.txt"
mystem = Mystem()


# size = 160
# window = 4
# min_count = 3
# sg = 1  # 0 for CBOW, 1 for skip-gram
# iter = 15


def train_model(text_file_path, size=30, window=7, min_count=3, sg=0, iter=15):
    """
        Training models with caching
    """

    model_file_path = MODEL_PATH + text_file_path + "-size" + str(size) + \
                      "-window" + str(window) + \
                      "-mincount" + str(min_count) + \
                      "-sg" + str(sg) + \
                      "-iter" + str(iter) + \
                      ".bin"

    try:
        model = Word2Vec.load(MODEL_PATH + text_file_path)
    except Exception as exc:

        print("Model not cached, retraining.", exc)
        print("Splitting...")

        splitted_sentences = []

        with open(DATA_PATH + text_file_path, "r") as input_file:

            whole_text = input_file.read().replace("\n", " ").lower()
            whole_text_sentences = sent_tokenize(whole_text, language='english')

            for sent in whole_text_sentences:
                splitted_sentences.append(mystem.lemmatize(sent))

        print("Training...")

        model = Word2Vec(splitted_sentences,
                         iter=iter, sg=sg, size=size, window=window,
                         min_count=min_count, workers=4)

        print("Training done.")

        model.save(model_file_path)

    return model


if __name__ == "__main__":
    # experiments

    model_nabokov = ("Nabokov", train_model("anya.txt"))
    model_demurova = ("Demurova", train_model("alisa.txt"))

    all_models = [model_nabokov, model_demurova]
    words_list = list(model_nabokov[1].vocab.keys() | model_demurova[1].vocab.keys())
    print("Len of vocab", len(words_list))

    for checked_word in words_list:
        for name, model in all_models:
            print(name, ":", checked_word, ":", end=" ")
            try:
                for word, distance in model.most_similar([checked_word], topn=5):
                    print(word, end="\t")  # "\t", distance)
            except KeyError as ke:
                print("Error", ke, end=" ")
            print()
        print()
