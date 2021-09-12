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
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import gensim.downloader
from collections import Counter, defaultdict
import matplotlib.pylab as plt
from visualisation import time_evolution_plot, plot_word_ot
from pprint import pprint
import seaborn as sns
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
import spacy

WHITELIST_EUROVOC_LABELS_PATH = os.path.join(os.environ['DTM_ROOT'], "dtm", "eurovoc_labels_merged.txt")
EUROVOC_PATH = os.path.join(os.environ['DTM_ROOT'], "dtm", "eurovoc_export_en.csv")

class DTMAnalysis:

    topic_cluster_mapping = {
        "0421 parliament": "Legislation",
        "0431 politics and public safety": "Legislation",
        "0806 international affairs": "Global",
        "0816 international security": "Global",
        "1611 economic conditions": "Economic",
        "1616 regions and regional policy": "Other",
        "2016 trade": "Economic",
        "2021 international trade": "Economic",
        "2411 monetary economics": "Economic",
        "5206 environmental policy": "Environment",
        "5211 natural environment": "Environment",
        "5216 deterioration of the environment": "Environment",
        # "6606 energy policy",
        "6611 coal and mining industries": "Fossil Fuels",
        "6616 oil industry": "Fossil Fuels",
        "6621 electrical and nuclear industries": "Industrial",
        "6816 iron, steel and other metal industries": "Industrial"
    }

    eurovoc_label_correction_map = {
        "6626 soft energy": "6626 renewable energy",
        "6616 oil industry": "6616 oil and gas industry"
    }

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
        model_root="/data/greyroads/energy-roadmap/DTM/greyroads_steo", 
        doc_year_map_file_name="model-year.dat",
        seq_dat_file_name="model-seq.dat",
        vocab_file_name="vocab.txt",
        model_out_dir="model_run",
        eurovoc_whitelist=True,
        **kwargs
        ):
        self.nlp = spacy.load("en_core_web_sm")
        self.ndocs = ndocs
        self.ntopics = ntopics
        if eurovoc_whitelist:
            self.whitelist_eurovoc_labels = [x.strip() for x in open(WHITELIST_EUROVOC_LABELS_PATH, "r").readlines()]
        self.eurovoc_whitelist = eurovoc_whitelist
        self.model_root = model_root
        self.model_out_dir = model_out_dir
        self.gam_path = os.path.join(self.model_root, self.model_out_dir, "lda-seq", "gam.dat")
        self.doc_year_map_path = os.path.join(self.model_root, doc_year_map_file_name)
        self.seq_dat = os.path.join(self.model_root, seq_dat_file_name)
        self.eurovoc = None
        self.eurovoc_topics = None
        self.embeddings = None
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

    def change_model(self, model_out_dir, n_topics):
        self.model_out_dir = model_out_dir
        self.ntopics = n_topics
        self.gam_path = os.path.join(self.model_root, self.model_out_dir, "lda-seq", "gam.dat")
        self.gammas = open(self.gam_path, "r").read().splitlines()
        assert len(self.gammas) == self.ndocs * self.ntopics

        # let's change the gammas into a nicer form, from a 1d array of length self. * ntopics
        # to a 2d array of shape (ndocs, ntopics)

        self.gammas = np.reshape(self.gammas, (self.ndocs, self.ntopics)).astype(np.double)
        assert len(self.gammas[0]) == self.ntopics
        # let's create a dataframe where each row is a document, with its topic
        # distribution and year of publication
        self.doc_topic_gammas = pd.DataFrame(zip(self.gammas, self.doc_year_mapping), columns=["topic_dist", "year"])

        # check to see that we have the same counts of yearly docs as the seq-dat file
        assert self.docs_per_year == self.doc_topic_gammas.groupby('year').count()['topic_dist'].tolist()

    def save_gammas(self, save_path, split=True):
        if split:
            tmp_df = pd.DataFrame(self.doc_topic_gammas['topic_dist'].tolist(), columns=[i for i in range(self.ntopics)])
            tmp_df['year'] = self.doc_topic_gammas['year']
            tmp_df.to_csv(save_path)
            del tmp_df
        else:
            self.doc_topic_gammas.to_csv(save_path)

    def _create_eurovoc_embedding_matrix(self):
        """This function creates an K x T_k x gloVedims embedding matrix where K is the number of eurovoc labels in the thesaurus,
        T_k is the number of terms for a eurovoc label k and gloVedims is the dimensions of the pre-trained gloVe word embeddings as per gensim docs.

        To create this matrix this function iterates through the terms of a eurovoc label 
        """
        self.label_term_map = {}
        self.embedding_matrix = []
        for topic in self.eurovoc_topics:
            mask = self.eurovoc['MT'].apply(lambda x: x.lower()) == topic
            terms = self.nlp.pipe([x.lower() for x in self.eurovoc[mask]['TERMS (PT-NPT)']], n_process=11)
            term_vec_matrix = []
            term_list = []
            for term in terms:
                vec = self._get_vector_from_tokens(term)
                # take average of vectors to achieve embedding for term
                if type(vec) == np.ndarray:
                    term_vec_matrix.append(vec)
                    term_list.append(term.text)
            self.label_term_map[topic] = term_list
            self.embedding_matrix.append(term_vec_matrix)
        self.embedding_matrix = np.array(self.embedding_matrix)

    def _get_vector_from_tokens(self, tokens):
        vec = []
        for tok in tokens:
            if tok.lemma_ in self.embeddings:
                vec.append(self.embeddings[tok.lemma_])
        if vec:
            vec = np.array(vec).mean(axis=0)
        return vec

    def _init_embeddings(self, load, save=True, label_term_map_path="whitelist_label_to_term_map.pickle", embedding_matrix_path="whitelist_embedding_matrix.pickle", emb_type='glove-wiki-gigaword-50'):
        print("Initialising gloVe embeddings...")
        self.embeddings = gensim.downloader.load(emb_type)
        if not load:
            self._create_eurovoc_embedding_matrix()
            if save and (not label_term_map_path or not embedding_matrix_path):
                print("If you want to save the embeddings, you need to provide label_term_map_path and embedding_matrix_path values.")
            elif save and label_term_map_path and embedding_matrix_path:
                with open(embedding_matrix_path, "wb+") as fp:
                    pickle.dump(self.embedding_matrix, fp)
                with open(label_term_map_path, "wb+") as fp:
                    pickle.dump(self.label_term_map, fp)
        elif load and (not label_term_map_path or not embedding_matrix_path):
            print("If loading the eurovoc embeddings matrix, you need to provide a path for the label term list and embedding matrix path.")
            sys.exit(1)
        elif load:
            with open(embedding_matrix_path, "rb") as fp:
                self.embedding_matrix = np.array(pickle.load(fp))
            with open(label_term_map_path, "rb") as fp:
                self.label_term_map = pickle.load(fp)
        
    def _init_eurovoc(self, eurovoc_path):
        print("Initialising EuroVoc...")
        def preproc(label):
            lowered_label = label.lower()
            if lowered_label in self.eurovoc_label_remapping:
                lowered_label = self.eurovoc_label_remapping[lowered_label]
            if lowered_label in self.eurovoc_label_correction_map:
                lowered_label = self.eurovoc_label_correction_map[lowered_label]
            return lowered_label
        self.eurovoc = pd.read_csv(eurovoc_path)
        self.eurovoc['MT'] = self.eurovoc['MT'].apply(preproc)
        # self.eurovoc['MT'] = self.eurovoc['MT'].apply(lambda x: self.eurovoc_label_remapping[x] if x in self.eurovoc_label_remapping else x)
        # self.eurovoc['MT'] = self.eurovoc['MT'].apply(lambda x: self.eurovoc_label_correction_map[x] if x in self.eurovoc_label_correction_map else x)
        # remove non-whitelisted topic terms
        if self.eurovoc_whitelist:
            m = self.eurovoc.apply(lambda x: x.MT.lower() in self.whitelist_eurovoc_labels, axis=1)
            self.eurovoc = self.eurovoc[m]
        self.eurovoc['index'] = [i for i in range(len(self.eurovoc))]
        self.eurovoc = self.eurovoc.set_index('index')
        self._create_eurovoc_label_term_map()
        # self.eurovoc_terms = [doc for doc in self.nlp.pipe(self.eurovoc['TERMS (PT-NPT)'], disable=['tok2vec', 'ner', 'parser', 'tagger'], batch_size=256, n_process=11)]

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

    def create_topic_proportions_per_year_df(self, remove_small_topics=False, threshold=0.01, merge_topics=False, include_names=False):
        """
        This function creates a dataframe which eventually will be used for
        plotting topic proportions over time. Similar to the visualisations used
        in the coal-discourse (Muller-hansen) paper. The dataframe contains a
        row for each year-topic combination along with its proportion of
        occurrence that year.
        """
        topic_names = self.get_topic_names()
        # Here I have begun working on retrieving the importance of topics over
        # time. That is, the Series topic_proportions_per_year contains the
        # importance of each topic for particular years.
        topic_proportions_per_year = self._get_topic_proportions_per_year()
        for_df = []
        for year, topic_props in zip(self.years, topic_proportions_per_year):
            if merge_topics:
                merged_topics = Counter()
                for topic_idx, topic in enumerate(topic_props):
                    curr_topic_name = re.search(r"(\d{4} .*?)\d+", topic_names[topic_idx]).group(1)
                    merged_topics.update({self.topic_cluster_mapping[curr_topic_name] : topic})
                for topic_idx, [topic_name, proportion] in enumerate(merged_topics.items()):
                    for_df.append([year, topic_idx, proportion, topic_name])
            else:
                for topic_idx, topic in enumerate(topic_props):
                    if include_names:
                        for_df.append([year, topic_idx, topic, str(topic_names[topic_idx])])
                    else:
                        for_df.append([year, topic_idx, topic, str(topic_idx)])
        topic_proportions_df = pd.DataFrame(for_df, columns=["year", "topic", "proportion", "topic_name"])
        if remove_small_topics:
            m = topic_proportions_df.groupby('topic')['proportion'].mean().apply(lambda x: x > threshold)
            topic_prop_mask = topic_proportions_df.apply(lambda row: m[row.topic] == True, axis=1)
            topic_proportions_df = topic_proportions_df[topic_prop_mask]
        # topic_proportions_df = pd.DataFrame(zip(self.years, topic_proportions_per_year), columns=["year", "topic", "proportion", "topic_name"])
        return topic_proportions_df

    def create_plottable_topic_proportion_ot_df(self, remove_small_topics=False, threshold=0.01, merge_topics=False, include_names=False):
        df = self.create_topic_proportions_per_year_df(remove_small_topics, threshold, merge_topics=merge_topics, include_names=include_names)
        df = df.pivot(index='year', columns='topic_name', values='proportion')
        return df

    def _create_eurovoc_label_term_map(self):
        eurovoc_label_term_map = {}
        self.eurovoc_topic_docs = {}
        for term, topic in zip(self.nlp.pipe(self.eurovoc['TERMS (PT-NPT)'], disable=['tok2vec', 'ner'], batch_size=256, n_process=11), self.eurovoc['MT'].apply(lambda x: x.lower())):
            if topic in eurovoc_label_term_map:
                eurovoc_label_term_map[topic].append(term)
            else:
                eurovoc_label_term_map[topic] = [term]
        for topic in eurovoc_label_term_map:
            curr_topic_list = eurovoc_label_term_map[topic]
            c_doc = Doc.from_docs(curr_topic_list, ensure_whitespace=True)
            self.eurovoc_topic_docs[topic] = c_doc
        self.eurovoc_topics = sorted(np.array(list(self.eurovoc_topic_docs.keys())))
        # self.eurovoc_topic_indices = sorted(self.eurovoc_topics)

    def get_topic_tfidf_scores(self, top_terms, tfidf_enabled=False):
        """
        Returns a matrix for a DTM topic where the rows represent a top term for the dtm topic, and the columns
        represent each EuroVoc topic. Each cell is the tfidf value of a particular term-topic
        combination. This will be used when calculating the automatic EuroVoc topic labels for the
        DTM topics.

        i.e. shape is | top_terms | x | EuroVoc Topics |
        """
        self.tfidf_mat = np.zeros((len(top_terms), len(self.eurovoc_topics)))
        # number of documents containing a term
        N = Counter()
        tfs = {}
        doc_lens = {}
        # ensure to get rid of the underscores in bigram terms and then rejoin with space
        # i.e. 'greenhouse_gas' becomes 'greenhouse gas'
        spacy_terms = [t for t in self.nlp.pipe([" ".join(t.split("_")) for _, t in top_terms], disable=['tok2vec', 'ner'], n_process=11)]
        self.raw_term_list = [t.text for t in spacy_terms]
        self.eurovoc_term_matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA", validate=True)
        for term in spacy_terms:
            self.eurovoc_term_matcher.add(term.text, [term])
        ## term freq
        ## total terms in each topic (doc)
        ## number of docs that match term
        ## total number of docs
        for topic in self.eurovoc_topics:
            term_freq = Counter()
            curr_doc = self.eurovoc_topic_docs[topic]
            doc_len = len(self.eurovoc_topic_docs[topic])
            doc_lens[topic] = doc_len
            terms_contained_in_topic = set()
            matches = self.eurovoc_term_matcher(curr_doc)
            if matches:
                for match_id, _, _ in matches:
                    matched_term = self.nlp.vocab.strings[match_id]
                    terms_contained_in_topic.add(matched_term)
                    term_freq[matched_term] += 1
                tfs[topic] = term_freq
            N.update(terms_contained_in_topic)
        # calculate tfidfs for each term in each topic
        for i, topic in enumerate(self.eurovoc_topics):
            try:
                curr_tfs = tfs[topic]
                doc_len = doc_lens[topic]
                for term in curr_tfs:
                    ind = self.raw_term_list.index(term)
                    tf = curr_tfs[term] / doc_len
                    idf = np.log(len(self.eurovoc_topic_docs.keys()) / N[term])
                    tfidf = tf * idf
                    if tfidf_enabled:
                        self.tfidf_mat[ind][i] = tfidf
                    else:
                        # just idf
                        self.tfidf_mat[ind][i] = idf
            except KeyError as e:
                continue
        return self.tfidf_mat
    
    def _get_eurovoc_scores(self, top_words, tfidf_enabled=True):
        c = Counter()
        terms = Counter()
        top_word_dict = dict([(" ".join(y.split("_")),x) for (x,y) in top_words])
        for i, topic in enumerate(self.eurovoc_topics):
            score = 0
            # topic_ind = np.where(self.eurovoc_topics == topic)[0][0]
            matches = self.eurovoc_term_matcher(self.eurovoc_topic_docs[topic])
            for match_id,_,_ in matches:
                term = self.nlp.vocab.strings[match_id]
                terms[term] += 1
                weight = top_word_dict[term]
                term_ind = self.raw_term_list.index(term)
                tfidf = self.tfidf_mat[term_ind][i]
                # weighting and tf-idf 
                if tfidf_enabled:
                    tmp = weight * tfidf
                    score = score + (weight * tfidf)
                else:
                    score = score + weight
            c.update({topic: score})
        return c

    def _get_embedding_scores(self, top_words):
        c = Counter()
        if isinstance(self.eurovoc, type(None)):
            self._init_eurovoc(EUROVOC_PATH)
        if not self.embeddings:
            self._init_embeddings(True, save=False)
        words = [self.nlp(" ".join(w.split("_"))) for _,w in top_words]
        word_probs = np.array([x for (x,_) in top_words])
        word_vecs = []
        for i,w in enumerate(words):
            vec = self._get_vector_from_tokens(w)
            if type(vec) != list:
                word_vecs.append(vec*word_probs[i])
        # word_vecs = np.array(word_vecs)*word_probs[:,np.newaxis]
        word_vec = np.array(word_vecs).mean(axis=0).reshape(1,-1)
        for i, topic in enumerate(self.embedding_matrix):
            # pairwise cosine sim between top words and topic term vectors
            topic_mat = np.array(topic)
            topic_vec = topic_mat.mean(axis=0).reshape(1,-1)
            score = cosine_similarity(word_vec, topic_vec)[0][0]
            # score = np.multiply(scores.squeeze(), word_probs).sum()
            topic_name = self.eurovoc_topics[i]
            c.update({topic_name: score})
        return c

    def get_auto_topic_name(self, top_words, i, top_n=4, stringify=True, tfidf_enabled=True, return_raw_scores=False, score_type="tfidf"):
        if score_type == "tfidf":
            weighted_ev_topics = self._get_eurovoc_scores(top_words, tfidf_enabled=tfidf_enabled)
        elif score_type == "embedding":
            weighted_ev_topics = self._get_embedding_scores(top_words)
        else:
            print("score_type needs to be one of either tfidf|embedding")
            sys.exit(1)
        if stringify:
            return str([(k, round(v,2)) for k, v in weighted_ev_topics.most_common(top_n)]) + str(i)
        elif return_raw_scores:
            return weighted_ev_topics
        else:
            return [(k, round(v,2)) for k, v in weighted_ev_topics.most_common(top_n)]

    def _get_baseline_topic_vectors(self, simple=True, zeroed=False, indices=None):
        """
        This function returns random scores normalised with a standard scaler. _init_eurovoc needs to run before this function.
        """
        topic_vectors = []
        topic_indices = indices if indices else sorted(self.eurovoc_topics)
        topic_proportions = np.array(self._get_topic_proportions_per_year(logged=True).tolist())
        for i in range(self.ntopics):
            word_dist_arr_ot = self.get_topic_word_distributions_ot(i)
            topic_proportions_ot = np.array(topic_proportions[:,i])
            top_words = self.get_words_for_topic(word_dist_arr_ot, n=30, with_prob=True, weighted=True, timestep_proportions=topic_proportions_ot)
            self.get_topic_tfidf_scores(top_words, tfidf_enabled=False)
            model_ev_scores = self.get_auto_topic_name(top_words, i, stringify=False, tfidf_enabled=False, return_raw_scores=True)
            topic_vectors.append([model_ev_scores[i] for i in topic_indices])
        rng = np.random.default_rng()
        baseline_topic_vectors = np.array(topic_vectors)
        if simple:
            # we want to shuffle everything
            for vec in baseline_topic_vectors:
                # print(f"before: {vec}")
                rng.shuffle(vec)
                # print(f"after: {vec}")
                # print("-----")
            return baseline_topic_vectors
        else:
            print("INTELLIGENT BASELINE")
            # we want to only shuffle elements that have score above threshold
            # shuffle top 5/10 with highest score
            threshold = np.quantile(baseline_topic_vectors, 0.7)
            threshold_matrix = []
            top_ten_matrix = []
            for topic in baseline_topic_vectors:
                # get top 10 score indices
                top_ten_indices = np.argsort(topic)[::-1][:10]
                top_ten_indices_shuffled = np.copy(top_ten_indices)
                rng.shuffle(top_ten_indices_shuffled)
                above_thresh_scores = []
                for score in topic:
                    if score > threshold:
                        above_thresh_scores.append(score)
                rng.shuffle(above_thresh_scores)
                if zeroed:
                    top_ten = np.zeros(shape=topic.shape)
                else:    
                    top_ten = np.copy(topic)
                for i in range(len(top_ten_indices)):
                    top_ten[top_ten_indices[i]] = topic[top_ten_indices_shuffled[i]] 
                i = 0
                # print(f"before: {topic}")
                threshold_vector = []
                for score in topic:
                    if score > threshold:
                        threshold_vector.append(above_thresh_scores[i])
                        i += 1
                    else:
                        if zeroed:
                            threshold_vector.append(0)
                        else:
                            threshold_vector.append(score)
                threshold_matrix.append(threshold_vector)
                top_ten_matrix.append(top_ten)
                # print(f"after: {np.array(new_arr)}")
                # print("-----")
            return np.array(threshold_matrix), np.array(top_ten_matrix)

    def generate_baselines(self, **kwargs):
        simple_baseline = self._get_baseline_topic_vectors(simple=True, **kwargs)
        threshold_baseline, top_ten_baseline = self._get_baseline_topic_vectors(simple=False, **kwargs)
        threshold_baseline_zeroed, top_ten_baseline_zeroed = self._get_baseline_topic_vectors(simple=False, zeroed=True, **kwargs)
        return simple_baseline, threshold_baseline, top_ten_baseline, threshold_baseline_zeroed, top_ten_baseline_zeroed

    def get_baseline_vec_topic_names(self, vec, top_n=4):
        """
        This function generates the topic names for a baseline vector which represents the scores
        of each of the EuroVoc topics for a particular DTM topic k. That is, how relevant each EuroVoc
        topic is to k. EuroVoc needs to be initialised for this to function correctly.
        """
        if not self.eurovoc_topics:
            print("need to init eurovoc before calling this function by _init_eurovoc")
            sys.exit(1)
        all_ev_topics = []
        for dtm_topic in vec:
            ev_topics = [(self.eurovoc_topics[x], round(dtm_topic[x],2)) for x in np.argsort(dtm_topic)[::-1]][:top_n]
            all_ev_topics.append(ev_topics)
        return all_ev_topics

    def get_topic_names(self, detailed=False, stringify=True, tfidf_enabled=True, _type="tfidf", raw=False, n=10):
        """
        
        """
        if isinstance(self.eurovoc, type(None)):
            self._init_eurovoc(EUROVOC_PATH)
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
            if _type == "tfidf":
                self.get_topic_tfidf_scores(top_words, tfidf_enabled=False)
            curr_topic_name = self.get_auto_topic_name(top_words, i, stringify=stringify, tfidf_enabled=tfidf_enabled, score_type=_type, return_raw_scores=raw)
            if detailed:
                topic_names.append((curr_topic_name, top_words))
            else:
                topic_names.append(curr_topic_name)
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

    def get_words_for_topic(self, word_dist_arr_ot, n=10, descending=True, with_prob=True, weighted=True, timestep_proportions=None, rescaled=True):
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
        if descending:
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
        else:
            return pd.Index([str(y) for y in sorted([int(x) for x in df.columns])])
    
    def plot_topics_ot(self, save_path, save=True, sort_by="peak_pos"):
        df_scores = self.create_plottable_topic_proportion_ot_df(include_names=False)
        for i in df_scores.index:
            df_scores.loc[i] = df_scores.loc[i] / df_scores.loc[i].sum() * 100
        sorted_selection = self._get_sorted_columns(df_scores, sort_by)
        plt = time_evolution_plot(df_scores[sorted_selection], save_path, scale=0.75, save=save)
        return plt
    
    def plot_words_from_topic(self, topic_idx, words, title, save_path=None):
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
        plt = plot_word_ot(plot_df, title, save_path=save_path)
        return plt
        

def compare_coherences(dataset_root, analysis_folder):
    pmi_c = Counter()
    npmi_c = Counter()
    with open(os.path.join(dataset_root, analysis_folder, "coherences.txt"), "r") as fp:
        reader = csv.reader(fp, delimiter="\t")
        for i, row in enumerate(reader):
            if i != 0:
                topic_num = row[0].split("_")[2].split("topics")[1]
                pmi = float(row[1])
                npmi = float(row[2])
                if topic_num in pmi_c:
                    pmi_c[topic_num].append(pmi)
                else:
                    pmi_c[topic_num] = [pmi]
                if topic_num in npmi_c:
                    npmi_c[topic_num].append(npmi)
                else:
                    npmi_c[topic_num] = [npmi]
    return pmi_c, npmi_c

def compare_dataset_coherences():
    datasets = [
        # os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_steo_all_bigram"),
        # os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all_bigram"),
        # os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all_bigram"),
        os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all_bigram"),
        # os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_coal_bigram"),
        # os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_biofuels_bigram"),
        # os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_solar_bigram")
    ]
    pmi = Counter()
    npmi = Counter()
    from statistics import median
    for dataset in datasets:
        pmi_c, npmi_c = compare_coherences(dataset, "analysis_all_eurovoc_topics")
        pmi.update(pmi_c)
        npmi.update(npmi_c)
    for topic_num in pmi:
        pmi_vals = pmi[topic_num]
        npmi_vals = npmi[topic_num]
        print(f"For models with {topic_num} topics\n\tAverage pmi is: {sum(pmi_vals)/len(pmi_vals)}\n\tMedian pmi is: {median(pmi_vals)}")
        print(f"For models with {topic_num} topics\n\tAverage npmi is: {sum(npmi_vals)/len(npmi_vals)}\n\tMedian npmi is: {median(npmi_vals)}")
        print("-----")
    print("==========")

if __name__ == "__main__":
    NDOCS = 15457 # number of lines in -mult.dat file.
    NTOPICS = 30
    tdma = DTMAnalysis(
        NDOCS, 
        NTOPICS,
        model_root=os.path.join(os.environ['HANSARD'], "coal_output", "dtm", "general_run_18Aug", "2a_ngram", "raw"),
        doc_year_map_file_name="model-year.dat",
        seq_dat_file_name="model-seq.dat",
        vocab_file_name="vocab.txt",
        model_out_dir="k30_a0.01_var0.1",
        eurovoc_whitelist=True,
        )
    topic_names = tdma.get_topic_names(_type="embedding", stringify=False)
    # names = tdma.get_topic_names(_type="embedding", raw=True, stringify=False)
    # tdma._init_eurovoc(EUROVOC_PATH)
    # tdma._init_embeddings(load=False, save=True)
    # tdma.plot_words_from_topic(1, ["greenhouse_gas"])
    # tdma.save_gammas()
    breakpoint()
    print("he")
