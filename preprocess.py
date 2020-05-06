from nltk.corpus import stopwords
import spacy
import re
from tools import logging




class Tokenizer():

    def __init__(self, tokenizer_type='normal', remove_stop_words=False):
        self.is_remove_stop_words = remove_stop_words
        if tokenizer_type == 'normal':
            self.tokenizer = self.normal_token
        elif tokenizer_type == 'spacy':
            self.nlp = spacy.load('en_core_web_sm')
            self.tokenizer = self.spacy_token
        else:
            raise RuntimeError(f'Tokenizer type is error, do not have type {tokenizer_type}')
        self.token_type = tokenizer_type
        self.stop_words = set(stopwords.words('english'))
        for w in ["<br />", '!', ',', '.', '?', '-s', '-ly', '</s>', 's', '</', '>', '/>', 'br', '<']:
            self.stop_words.add(w)
        logging(f'using tokenizer {tokenizer_type}, is_remove_stop_words={remove_stop_words}')


    def pre_process(self, text: str)->str:
        text = text.lower().strip()
        text = re.sub(r"<br />", "", text)
        text = re.sub(r'(\W)(?=\1)', '', text)
        text = re.sub(r"([.!?,])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
        return text.strip()

    def normal_token(self, text: str, is_word=True)->[str]:
        if is_word:
            return [tok for tok in text.split() if not tok.isspace()]
        else:
            return [tok for tok in text]


    def spacy_token(self, text: str, is_word=True)->[str]:
        if is_word:
            text = self.nlp(text)
            return [token.text for token in text if not token.text.isspace()]
        else:
            return [tok for tok in text]


    def stop_words_filter(self, words: [str]):
        return [word for word in words if word not in self.stop_words]


    def __call__(self, text: str, is_word=True)->[str]:
        text = self.pre_process(text)
        words = self.tokenizer(text, is_word=is_word)
        if self.is_remove_stop_words:
            return self.stop_words_filter(words)
        return words





if __name__ == '__main__':
    pass