# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    time.sleep(0.50)
    txt = requests.get(url).text
    clean_txt = (txt
                .replace('\r\n', '\n')
                )
    # finding the beginning of the text
    start_idx = re.search(r"\*{3}(\n){3}", clean_txt).start() + 3
    end_idx = re.search(r"(\n)+(\*){3} END OF THE PROJECT GUTENBERG EBOOK", clean_txt).start()

    return clean_txt[start_idx:end_idx]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    starting_ending = re.sub(r"\n{2,}", "\x03 \x02", book_string.strip())
    cleaned = '\x02' + starting_ending + '\x03'
    tokens = re.findall(r"\x02|\x03|\w+|[^w+\s]", cleaned)
    return tokens   

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        # find the probabilities that the tokens occur
        probs_df = pd.DataFrame(data = {'token' : pd.Series(tokens).unique(),
                                         'probability' : 1/pd.Series(tokens).nunique()})
        probs_srs = probs_df.set_index('token')['probability']
        return probs_srs
        # drop the /x02 and /x03 
      
    def probability(self, words):
        # takes a non-empty sequnce of tokens returns the probability of the string given under the language model
        running_prob = 1
        
        for word in words:
            if word in self.mdl.index: 
                running_prob *= self.mdl.loc[word]
            else: 
                return 0      
        return running_prob
        
    def sample(self, M):
        # takes in a positive integer M and generates a string made up of M tokens using the language model. This method generates random sentences!
        token_word = np.array(self.mdl.index)
        token_probs = np.array(self.mdl)
        chosen_words = list(np.random.choice(token_word, size =M, p=token_probs))
        output = " ".join(chosen_words)
        return output

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        trained_srs = pd.Series(tokens).value_counts() / len(tokens)
        return trained_srs
    
    def probability(self, words):
        # takes a non-empty sequnce of tokens returns the probability of the string given under the language model
        running_prob = 1
        for word in words: 
            if word in self.mdl.index: 
                running_prob *= self.mdl.loc[word]
            else: 
                return 0 
        return running_prob
        
    def sample(self, M):
        chosen_words = list(np.random.choice(self.mdl.index, size = M, p=self.mdl))
        output = " ".join(chosen_words)
        return output

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        self.N = N
        ngrams = self.create_ngrams(tokens)
        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        t0 = time.time()
        output_tups = []
        token_length = len(tokens)
        start = 0 # starts from the front of the tokens list
        end = self.N # end of the window we want to look 
        
        while end <= token_length: 
            output_tups.append(tuple(tokens[start:end]))
            start += 1 
            end += 1
        return output_tups
            
    def train(self, ngrams):
        # N-Gram counts C(w_1, ..., w_n)
        n_grams_lst = ngrams
        
        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        n1_grams_lst = [tuple(tup[0:len(tup)-1]) for tup in ngrams]      

        # Make a DF
        data = {'ngram': n_grams_lst,
                'n1gram': n1_grams_lst
                }
        df = pd.DataFrame(data)
        ngram_counts = df.groupby('ngram').size().reset_index(name='count')
        n1gram_counts = df.groupby('n1gram').size().reset_index(name='n1gram_count')

        df = pd.merge(df, ngram_counts, on='ngram', how='left')
        df = pd.merge(df, n1gram_counts, on='n1gram', how='left')
        df['prob'] = df['count'] / df['n1gram_count']

        df = df.drop_duplicates()
        df = df.drop(columns=['count', 'n1gram_count'])

        return df
    def probability(self, words):
        # self.mdl contains the df we want 
        
        # words is tuple of words in the sentence
        runningProb = 1
        # how many models will we have to go through? # Example we have N=3 means that we'll need to go through 3 models (Tri, Bi, and Unigram)
        end_tup = len(words) 
         
        start_idx = end_tup - self.N
        while end_tup > (self.N - 1):
            big_tup = tuple(words[start_idx:end_tup])
            small_tup = tuple(words[start_idx:end_tup-1])
            # find the probs: 
            query_df = self.mdl[(self.mdl['ngram'] == big_tup) & (self.mdl['n1gram'] == small_tup)]
            if len(query_df) == 0:
                return 0 
            else: 
                prob = query_df['prob'].iloc[0]

            runningProb *= prob
            
            end_tup -= 1 
            start_idx -= 1
            
        return runningProb * self.prev_mdl.probability(words[:self.N-1])

    def sample(self, M): # M is the length
        def generate_token(curr_mdl, last_n, curr_n, M, output=''): 
            if M == 0: 
                return output + ' \x03'

            target_mdl = curr_mdl
            if curr_n < curr_mdl.N: 
                for _ in range(curr_mdl.N - curr_n): 
                    target_mdl = target_mdl.prev_mdl
            target_mdl = target_mdl.mdl

            if target_mdl['n1gram'].isin([tuple(last_n)]).any(): 
                possible_df = target_mdl.loc[target_mdl['n1gram'] == tuple(last_n)]
                possible_tups = possible_df['ngram']
                probs = possible_df['prob'].tolist()
                chosen_tup = np.random.choice(possible_tups, p=probs)
                chosen = str(chosen_tup[-1])
            else:
                chosen = '\x03'

            output = f'{output} {chosen}'
            last_n.append(chosen)
            
            if len(last_n) >= curr_mdl.N: 
                last_n = last_n[-curr_mdl.N + 1:]

            return generate_token(curr_mdl, last_n, curr_n + 1 if curr_mdl.N else curr_n, M-1, output)

        start = '\x02'
        initial_last_n = ['\x02']
        return generate_token(self, initial_last_n, 2, M-1, start)

                    