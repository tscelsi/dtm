"""
    -- analysis.py -- 
    
    This file contains logic and functionality for analysing
    the results of a model created by the Dynamic Topic Model (DTM). We use this
    model to understand what sorts of topics appear in a corpus of text, and
    also to understand how the prevalence of these topics fluctuates over time. 
"""

import pandas as pd
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import re
import csv
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import gensim.downloader
from collections import Counter, defaultdict
import matplotlib.pylab as plt
from visualisation import time_evolution_plot, plot_word_ot, plot_word_topic_evolution_ot
from pprint import pprint
import seaborn as sns
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
import spacy
from eurovoc import Eurovoc
from auto_labelling import AutoLabel

WHITELIST_EUROVOC_LABELS_PATH = os.path.join(os.environ['DTM_ROOT'], "dtm", "eurovoc_labels_merged.txt")
EUROVOC_PATH = os.path.join(os.environ['DTM_ROOT'], "dtm", "eurovoc_export_en.csv")

class DTMAnalysis:
    
    eurovoc_label_correction_map = {
        "6626 soft energy": "6626 renewable energy",
        "6616 oil industry": "6616 oil and gas industry"
    }

    # default remapping, can be overridden in __init__
    eurovoc_label_remapping = {
        "1621 economic structure": "1611 economic conditions",
        "2006 trade policy": "2016 business operations and trade",
        "2421 free movement of capital": "2016 business operations and trade",
        "2016 trade": "2016 business operations and trade",
        "4006 business organisation": "2016 business operations and trade",
        "4016 legal form of organisations" : "2016 business operations and trade",
        "2426 financing and investment": "2016 business operations and trade",
        "2026 consumption": "2016 business operations and trade",
    }


    def __init__(
        self, 
        ndocs, 
        ntopics, 
        model_root, 
        doc_year_map_file_name="model-year.dat",
        seq_dat_file_name="model-seq.dat",
        vocab_file_name="vocab.txt",
        model_out_dir="model_run",
        thesaurus=None,
        spacy_lang="en_core_web_sm"
        ):
        self.nlp = spacy.load(spacy_lang)
        self.ndocs = ndocs
        self.ntopics = ntopics
        if thesaurus:
            self.thesaurus = thesaurus
        else:
            # defaults to Eurovoc as per original paper
            self.thesaurus = Eurovoc(eurovoc_whitelist=True).eurovoc
        self.auto_labelling = AutoLabel(self.thesaurus, phrase_col="TERMS (PT-NPT)", label_col="MT", spacy_lang=spacy_lang)
        self.model_root = model_root
        self.model_out_dir = model_out_dir
        self.gam_path = os.path.join(self.model_root, self.model_out_dir, "lda-seq", "gam.dat")
        self.doc_year_map_path = os.path.join(self.model_root, doc_year_map_file_name)
        self.seq_dat = os.path.join(self.model_root, seq_dat_file_name)
        self.topic_prefix = "topic-"
        self.topic_suffix = "-var-e-log-prob.dat"
        
        vocab_file = os.path.join(self.model_root, vocab_file_name)

        vocab = open(vocab_file, "r").read().splitlines()
        self.vocab = [x.split("\t")[0] for x in vocab]
        self.index_to_word = {i:w for i, w in enumerate(self.vocab)}

        # load the doc-year mapping, which is just a list of length(number of documents) in the same order as
        # the -mult.dat file.
        self.doc_year_mapping = [int(x) for x in open(self.doc_year_map_path, "r").read().splitlines()]
        assert len(self.doc_year_mapping) == ndocs

        self.years = sorted(list(set(self.doc_year_mapping)))

        # load the counts of years file

        self.docs_per_year = [int(x) for x in open(self.seq_dat, "r").read().splitlines()[1:]]

        # load the models gammas

        self.gammas = open(self.gam_path, "r").read().splitlines()
        assert len(self.gammas) == ndocs * ntopics

        # let's change the gammas into a nicer form, from a 1d array of length ndocs * ntopics
        # to a 2d array of shape (ndocs, ntopics)

        self.gammas = np.reshape(self.gammas, (ndocs, ntopics)).astype(np.double)
        assert len(self.gammas[0]) == ntopics

        # let's create a dataframe where each row is a document, with its topic
        # distribution and year of publication
        self.doc_topic_gammas = pd.DataFrame(zip(self.gammas, self.doc_year_mapping), columns=["topic_dist", "year"])

        # check to see that we have the same counts of yearly docs as the seq-dat file
        assert self.docs_per_year == self.doc_topic_gammas.groupby('year').count()['topic_dist'].tolist()

    def save_gammas(self, save_path, split=True):
        doc_ids = pd.read_csv(os.path.join(self.model_root, "doc_ids.csv"))
        assert len(doc_ids) == len(self.doc_topic_gammas)
        if split:
            tmp_df = pd.DataFrame(self.doc_topic_gammas['topic_dist'].tolist(), columns=[i for i in range(self.ntopics)])
            tmp_df['year'] = self.doc_topic_gammas['year']
            tmp_df['doc_id'] = doc_ids['doc_id']
            tmp_df.to_csv(save_path)
            del tmp_df
        else:
            self.doc_topic_gammas['doc_id'] = doc_ids['doc_id']
            self.doc_topic_gammas.to_csv(save_path)

    def _get_topic_proportions_per_year(self, logged=False):
        """This function returns a pandas DataFrame of years and their corresponding
        topic proportions such that for a year, the topic proportions sum to one. 
        i.e. how probable is it that topic X appears in year Y.
        """
        def get_topic_proportions(row, logged):
            total = np.sum(row)
            if logged:
                return [np.log(topic / total) for topic in row]
            else:
                return [topic / total for topic in row]
        grouping = self.doc_topic_gammas.groupby('year')
        x = grouping.topic_dist.apply(np.sum)
        # assign self the list of years 
        topic_proportions_per_year = x.apply(lambda x: get_topic_proportions(x, logged))
        return topic_proportions_per_year

    def create_topic_proportions_per_year_df(self, remove_small_topics=False, threshold=0.01, merge_topics=False, include_names=False, **kwargs):
        """
        This function creates a dataframe which eventually will be used for
        plotting topic proportions over time. Similar to the visualisations used
        in the coal-discourse (Muller-hansen) paper. The dataframe contains a
        row for each year-topic combination along with its proportion of
        occurrence that year.
        """
        topic_names = self.get_topic_names(**kwargs)
        # topic_names = str([[re.sub(r"\d{4} ", "", x) for x,_ in topic_name] for topic_name in topic_names])
        # Here I have begun working on retrieving the importance of topics over
        # time. That is, the Series topic_proportions_per_year contains the
        # importance of each topic for particular years.
        topic_proportions_per_year = self._get_topic_proportions_per_year()
        for_df = []
        for year, topic_props in zip(self.years, topic_proportions_per_year):
            if merge_topics:
                merged_topics = Counter()
                for topic_idx, topic in enumerate(topic_props):
                    curr_topic_name = re.search(r"^\d{4} (.*)$", topic_names[topic_idx][0][0]).group(1)
                    merged_topics.update({curr_topic_name : topic})
                for topic_idx, [topic_name, proportion] in enumerate(merged_topics.items()):
                    for_df.append([year, topic_idx, proportion, topic_name])
            else:
                for topic_idx, topic in enumerate(topic_props):
                    if include_names:
                        curr_topic_name = re.search(r"^\d{4} (.*)$", topic_names[topic_idx][0][0]).group(1)
                        for_df.append([year, topic_idx, topic, curr_topic_name + "(" + str(topic_idx) + ")"])
                    else:
                        for_df.append([year, topic_idx, topic, str(topic_idx)])
        topic_proportions_df = pd.DataFrame(for_df, columns=["year", "topic", "proportion", "topic_name"])
        if remove_small_topics:
            m = topic_proportions_df.groupby('topic')['proportion'].mean().apply(lambda x: x > threshold)
            topic_prop_mask = topic_proportions_df.apply(lambda row: m[row.topic] == True, axis=1)
            topic_proportions_df = topic_proportions_df[topic_prop_mask]
        # topic_proportions_df = pd.DataFrame(zip(self.years, topic_proportions_per_year), columns=["year", "topic", "proportion", "topic_name"])
        return topic_proportions_df

    def create_plottable_topic_proportion_ot_df(self, remove_small_topics=False, threshold=0.01, merge_topics=False, include_names=False, limit=2020):
        df = self.create_topic_proportions_per_year_df(remove_small_topics, threshold, merge_topics=merge_topics, include_names=include_names)
        df = df[df['year'] <= limit]
        df = df.pivot(index='year', columns='topic_name', values='proportion')
        return df
    
    def get_single_topic_proportions_ot(self, topic_idx):
        df = self.create_plottable_topic_proportion_ot_df(include_names=False)
        max_val = df.max().max()
        df = df.loc[:,str(topic_idx)]
        df = df / df.sum() * 100
        return df

    def get_label_names_for_topic_ot(self, topic_idx, return_type="matrix", normalised=True, n=10, **kwargs):
        word_dist_arr_ot = self.get_topic_word_distributions_ot(topic_idx)
        top_words_ot = self.get_words_for_topic(word_dist_arr_ot, over_time=True, n=n)
        label_names = []
        for tw in top_words_ot:
            topic_scores = self.auto_labelling._get_auto_topic_name(tw, topic_idx, score_type="embedding", raw=True, **kwargs)
            total_score = sum(topic_scores.values())
            if return_type == "matrix":
                    if normalised:
                        label_names.append(np.array([topic_scores[label]/total_score for label in self.auto_labelling.sorted_labels]))
                    else:
                        label_names.append(np.array([topic_scores[label] for label in self.auto_labelling.sorted_labels]))
            else:
                if normalised:
                    label_names.append(dict([(l, score / total_score) for l,score in topic_scores.items()]))
                else:
                    label_names.append(topic_scores)
        if return_type == "matrix":
            return np.array(label_names)                
        return label_names

    def get_topic_names(self, _type="tfidf", raw=False, n=10, **kwargs):
        """
        """
        topic_names = []
        self.top_word_arr = []
        proportions = np.array(self._get_topic_proportions_per_year(logged=True).tolist())
        for i in range(self.ntopics):
            word_dist_arr_ot = self.get_topic_word_distributions_ot(i)
            topic_proportions_ot = np.array(proportions[:,i])
            # we want to weight our top words by the topic proportions, so weighted=True
            top_words = self.get_words_for_topic(word_dist_arr_ot, n=n, with_prob=True, weighted=True, timestep_proportions=topic_proportions_ot)
            # add top words to class object
            self.top_word_arr.append(top_words)
        topic_names = self.auto_labelling.get_topic_names(self.top_word_arr, top_n=4, score_type=_type, raw=raw, **kwargs)
        self.topic_names = topic_names
        return topic_names
    
    def get_top_words(self, **kwargs):
        """
        Gets the top words for each topic within the DTM model being analysed.
        """
        self.top_word_arr = []
        proportions = np.array(self._get_topic_proportions_per_year(logged=True).tolist())
        for i in range(self.ntopics):
            word_dist_arr = self.get_topic_word_distributions_ot(i)
            _proportions = np.array(proportions[:,i])
            self.top_word_arr.append(self.get_words_for_topic(word_dist_arr, timestep_proportions=_proportions, weighted=True, **kwargs))
        return self.top_word_arr

    def get_words_for_topic(self, word_dist_arr_ot, n=10, with_prob=True, weighted=True, timestep_proportions=None, rescaled=True, over_time=False):
        """
        This function takes in an NUM_YEARSxLEN_VOCAB array/matrix that
        represents the vocabulary distribution for each year a topic is
        fit. It returns a list of the n most probable words for that
        particular topic and optionally their summed probabilities over all
        years. These probabilities can be weighted or unweighted.

        Args: 

        word_dist_arr_ot (np.array): This is the array containing a topics word
            distribution for each year. It takes the shape: ntimes x len(vocab)

        n (int): The number of word probabilities to return descending (bool):
            Whether to return the word probabilities in ascending or descending
            order. i.e ascending=lowest probability at index 0 and vice-versa.

        with_prob (bool): Whether to return just the word text, or a tuple of word
        text and the word's total probability summed over all time spans

        Returns: Either a list of strings or a list of tuples (float, string)
        representing the summed probability of a particular word.
        """
        if over_time:
            top_words = []
            for timestep_vocab in word_dist_arr_ot:
                timestep_sorted = timestep_vocab.argsort()
                timestep_sorted = np.flip(timestep_sorted)
                timestep_top_words = [self.index_to_word[i] for i in timestep_sorted[:n]]
                timestep_top_pw = [timestep_vocab[i] for i in timestep_sorted[:n]]
                if with_prob and rescaled:
                    total = sum(timestep_top_pw)
                    rescaled_probs = [x/total for x in timestep_top_pw]
                    top_words.append([(i, j) for i,j in zip(rescaled_probs, timestep_top_words)])
                else:
                    top_words.append(timestep_top_words)
            return top_words
        if weighted and not type(timestep_proportions) == np.ndarray:
            print("need to specify the timestep proportions to use in weighting with timestep_proportions attribute.")
            sys.exit(1)
        elif weighted:
            assert timestep_proportions.shape[0] == word_dist_arr_ot.shape[0]
            weighted_word_dist_acc = np.zeros(word_dist_arr_ot.shape)
            logged_word_dist_arr_ot = np.log(word_dist_arr_ot)
            for i in range(len(logged_word_dist_arr_ot)):
                np.copyto(weighted_word_dist_acc[i], np.exp(timestep_proportions[i] + logged_word_dist_arr_ot[i]))
            acc = np.sum(weighted_word_dist_acc, axis=0)
        else:
            acc = np.sum(word_dist_arr_ot, axis=0)
        word_dist_arr_sorted = acc.argsort()
        word_dist_arr_sorted = np.flip(word_dist_arr_sorted)
        top_pw = [acc[i] for i in word_dist_arr_sorted[:n]]
        top_words = [self.index_to_word[i] for i in word_dist_arr_sorted[:n]]
        if with_prob and rescaled:
            total = sum(top_pw)
            rescaled_probs = [x/total for x in top_pw]
            return [(i, j) for i,j in zip(rescaled_probs, top_words)]
        elif with_prob:
            return [(i, j) for i,j in zip(top_pw, top_words)]
        else:
            return [i for i in top_words]
    
    def get_topic_word_distributions_ot(self, topic_idx):
        """
        k shouldn't be over 99
        """
        if topic_idx<10:
            k = f"00{topic_idx}"
        else:
            k = f"0{topic_idx}"
        topic_file_name = self.topic_prefix + k + self.topic_suffix
        topic_file_path = os.path.join(self.model_root, self.model_out_dir, "lda-seq", topic_file_name)
        try:
            topic_word_distributions = open(topic_file_path).read().splitlines()
        except:
            print("Can't open", topic_file_path)
            raise Exception
        assert len(topic_word_distributions) == len(self.vocab) * len(self.years), print("WRONG VOCAB!!")
        
        word_dist_arr = np.exp(np.reshape([float(x) for x in topic_word_distributions], (len(self.vocab), len(self.years))).T)
        return word_dist_arr

    def create_top_words_df(self, n=10):
        # Here we want to visualise the top 10 most pertinent words to a topic over each timestep
        # similar to figure 8 in the Muller-hansen paper
        topw_df = pd.DataFrame(columns=["topic_idx", "year", "top_words"])
        for knum in range(100):
            try:
                word_dist_arr = self.get_topic_word_distributions_ot(knum)
            except:
                break
            top_words_per_year = []
            for year_idx in range(0,len(self.years)):
                # find top n most pertinent
                topws = word_dist_arr[year_idx].argsort()[-n:]
                topws = np.flip(topws)
                # np.take_along_axis(word_dist_arr[0], topws, axis=0)
                top_words = [self.index_to_word[i] for i in topws]
                top_words_per_year.append(top_words)
            assert len([knum]*len(self.years)) == len(self.years)
            assert len(self.years) == len(top_words_per_year)
            topic_df = pd.DataFrame(zip([knum]*len(self.years),self.years,top_words_per_year), columns=["topic_idx", "year", "top_words"])
            topw_df = topw_df.append(topic_df)
        return topw_df
    
    def print_topic_ot(self, topic_idx, topw_df):
        topw_df.groupby('year').top_words.apply(np.array)[0][0]
        top_words_per_topic = topw_df.groupby('topic_idx').top_words.apply(np.array)[topic_idx]
        years_per_topic = topw_df.groupby('topic_idx').year.apply(np.array)[topic_idx]
        print(f"TOPIC: {topic_idx}\n")
        for year, top_words in zip(years_per_topic, top_words_per_topic):
            print(f"{year}\t{top_words}")
            print("-----")
    
    def _get_sorted_columns(self, df, sort_by="peak_pos"):
        # sort according to position of peak
        sel2 = df.copy()
        if sort_by == 'peak_pos':
            sel2.loc['peak_pos'] = [sel2[topic].idxmax() for topic in sel2.columns]
            sel2 = sel2.sort_values(by='peak_pos', axis=1)
            sel2 = sel2.drop('peak_pos')
            return sel2.columns
        try:
            columns = sorted([int(x) for x in df.columns])
        except:
            columns = sorted(df.columns)
        return pd.Index([str(y) for y in columns])
    
    def plot_topics_ot(self, save_path, save=True, sort_by="peak_pos", keep=None, include_names=False, scale=0.75, merge_topics=False):
        df_scores = self.create_plottable_topic_proportion_ot_df(include_names=include_names, merge_topics=merge_topics)
        for i in df_scores.index:
            df_scores.loc[i] = df_scores.loc[i] / df_scores.loc[i].sum() * 100
        sorted_selection = self._get_sorted_columns(df_scores, sort_by)
        if keep:
            df_scores = df_scores[sorted_selection].loc[:,[str(x) for x in keep]]
        else:
            df_scores = df_scores[sorted_selection]
        plt = time_evolution_plot(df_scores, save_path, scale=scale, save=save)
        return plt

    def plot_words_ot_from_topic(self, topic_idx, words, title, save_path=None, plot=True):
        try:
            word_indexes = []
            for word in words:
                ind = self.vocab.index(word)
                assert self.vocab[ind] == word
                word_indexes.append(ind)
        except:
            print("word not in vocab")
            sys.exit(1)
        topic_word_distribution = self.get_topic_word_distributions_ot(topic_idx)
        word_ot = topic_word_distribution[:, word_indexes]
        plot_df = pd.DataFrame(word_ot, columns=words)
        plot_df['year'] = self.years
        plot_df = plot_df.set_index('year')
        if plot:
            plt = plot_word_ot(plot_df, title, save_path=save_path)
            return plot_df, plt
        else:
            return plot_df
    
    def plot_labels_ot_from_topic(self, topic_idx, labels, title, save_path=None, plot=True, n=10):
        plot_df = pd.DataFrame(columns=["year", "label", "value"])
        label_names_ot = self.get_label_names_for_topic_ot(topic_idx, normalised=True, n=n)
        for label in labels:
            index = self.eurovoc_topics.index(label)
            subbed_label = re.sub(r"\d{4} ", "", label)
            labels = [subbed_label] * len(self.years)
            tmp_df = pd.DataFrame(data={"year": self.years, "label": labels, "value": label_names_ot[:,index]})
            plot_df = plot_df.append(tmp_df, ignore_index=True)
        return plot_df

    
    def plot_words_ot_with_proportion(self, topics, wordlist, titles, figsize=None, save_path=None):
        assert len(topics) == len(wordlist) == len(titles)
        dfs_to_plot = []
        for topic, words in zip(topics, wordlist):
            words_df = self.plot_words_ot_from_topic(topic, words, None, plot=False)
            props_df = self.get_single_topic_proportions_ot(topic)
            words_df['_prop'] = props_df
            dfs_to_plot.append(words_df)
        plot_word_topic_evolution_ot(dfs_to_plot, titles, figsize=figsize, save_path=save_path)


if __name__ == "__main__":
    NDOCS = 2686 # number of lines in -mult.dat file.
    NTOPICS = 30
    whitelist_label_path = os.path.join(os.environ['DTM_ROOT'], "dtm", "eurovoc_labels_merged.txt")
    analysis = DTMAnalysis(
        2686, 
        30,
        model_root=os.path.join(os.environ['DTM_ROOT'], "dtm", "final_datasets", "greyroads_aeo_min_freq_40_1997_2020_ngram"),
        doc_year_map_file_name="model-year.dat",
        seq_dat_file_name="model-seq.dat",
        vocab_file_name="vocab.txt",
        model_out_dir="k30_a0.01_var0.05",
        # eurovoc_whitelist=True,
        # whitelist_path=whitelist_label_path,
        )
    # topic_names = analysis.get_topic_names()
    emb_topic_names = analysis.get_topic_names(_type="embedding")
