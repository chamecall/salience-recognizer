import gensim.downloader as api
import pandas as pd
from fse import IndexedList
from fse.models import SIF
from multiprocessing import cpu_count
from tqdm import tqdm

class JokePicker:
    def __init__(self, jokes_path, model_path):
        self.jokes = pd.read_csv(jokes_path)
        self.model = SIF.load(model_path)
        self.prev_jokes = []

    def pick_jokes_by_context(self, sentence):

        res = self.model.sv.similar_by_sentence(sentence.split(), model=self.model, topn=3)
        joke_nums = [joke[0] for joke in res]
        jokes = self.jokes.iloc[joke_nums].Joke
        jokes = [tpl[1] for tpl in jokes.iteritems()]
        self.prev_jokes = jokes
        return jokes