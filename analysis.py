"""
    -- analysis.py -- This file contains logic and functionality for analysing
    the results of a model created by the Dynamic Topic Model (DTM). We use this
    model to understand what sorts of topics appear in a corpus of text, and
    also to understand how the prevalence of these topics fluctuates over time. 
"""

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import re
from collections import Counter, defaultdict
import matplotlib.pylab as plt
from pprint import pprint
import seaborn as sns
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
import spacy

EUROVOC_PATH = os.path.join("eurovoc_export_en.csv")

class TDMAnalysis:

    whitelist_eurovoc_topics = [
        "0421 parliament",
        "0431 politics and public safety",
        "0806 international affairs",
        "0816 international security",
        "1611 economic conditions",
        "1616 regions and regional policy",
        "2016 trade",
        "2021 international trade",
        "2411 monetary economics",
        "5206 environmental policy",
        "5211 natural environment",
        "5216 deterioration of the environment",
        "6606 energy policy",
        "6611 coal and mining industries",
        "6616 oil industry",
        "6621 electrical and nuclear industries",
        "6816 iron, steel and other metal industries"
    ]

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


    def __init__(
        self, 
        ndocs, 
        ntopics, 
        model_root="/data/greyroads/energy-roadmap/DTM/greyroads_steo", 
        doc_year_map_file_name="eiatfidf-year.dat", 
        seq_dat_file_name="eiatfidf-seq.dat", 
        vocab_file_name="vocab_tfidf.txt",
        model_out_dir="model_run",
        eurovoc_whitelist=True
        ):
        self.nlp = spacy.load("en_core_web_sm")
        self.ndocs = ndocs
        self.ntopics = ntopics
        self.eurovoc_whitelist = eurovoc_whitelist
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
        self.doc_year_mapping = open(self.doc_year_map_path, "r").read().splitlines()
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

    def _init_eurovoc(self, eurovoc_path):
        self.eurovoc = pd.read_csv(eurovoc_path)
        # remove non-whitelisted topic terms
        if self.eurovoc_whitelist:
            m = self.eurovoc.apply(lambda x: x.MT.lower() in self.whitelist_eurovoc_topics, axis=1)
            self.eurovoc = self.eurovoc[m]
        self.eurovoc['index'] = [i for i in range(len(self.eurovoc))]
        self.eurovoc = self.eurovoc.set_index('index')
        # self.eurovoc_terms = [doc for doc in self.nlp.pipe(self.eurovoc['TERMS (PT-NPT)'], disable=['tok2vec', 'ner', 'parser', 'tagger'], batch_size=256, n_process=11)]

    def create_topic_proportions_per_year_df(self, remove_small_topics, threshold, merge_topics=False):
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
        def get_topic_proportions(row):
            total = np.sum(row)
            return [topic / total for topic in row]

        grouping = self.doc_topic_gammas.groupby('year')
        x = grouping.topic_dist.apply(np.sum)
        # assign self the list of years 
        topic_proportions_per_year = x.apply(get_topic_proportions)
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
                    for_df.append([year, topic_idx, topic, str(topic_names[topic_idx])])
        topic_proportions_df = pd.DataFrame(for_df, columns=["year", "topic", "proportion", "topic_name"])
        if remove_small_topics:
            m = topic_proportions_df.groupby('topic')['proportion'].mean().apply(lambda x: x > threshold)
            topic_prop_mask = topic_proportions_df.apply(lambda row: m[row.topic] == True, axis=1)
            topic_proportions_df = topic_proportions_df[topic_prop_mask]
        # topic_proportions_df = pd.DataFrame(zip(self.years, topic_proportions_per_year), columns=["year", "topic", "proportion", "topic_name"])
        return topic_proportions_df

    def create_plottable_topic_proportion_ot_df(self, remove_small_topics=False, threshold=0.01, merge_topics=False):
        df = self.create_topic_proportions_per_year_df(remove_small_topics, threshold, merge_topics=merge_topics)
        df = df.pivot(index='year', columns='topic_name', values='proportion')
        return df

    def create_eurovoc_topic_term_map(self):
        eurovoc_topic_term_map = {}
        self.eurovoc_topic_docs = {}
        for term, topic in zip(self.nlp.pipe(self.eurovoc['TERMS (PT-NPT)'], disable=['tok2vec', 'ner'], batch_size=256, n_process=11), self.eurovoc['MT'].apply(lambda x: x.lower())):
            if topic in eurovoc_topic_term_map:
                eurovoc_topic_term_map[topic].append(term)
            else:
                eurovoc_topic_term_map[topic] = [term]
        for topic in eurovoc_topic_term_map:
            curr_topic_list = eurovoc_topic_term_map[topic]
            c_doc = Doc.from_docs(curr_topic_list, ensure_whitespace=True)
            self.eurovoc_topic_docs[topic] = c_doc
        self.eurovoc_topics = np.array(list(self.eurovoc_topic_docs.keys()))
        print("finished.")

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
        for topic in self.eurovoc_topic_docs:
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


    def eurovoc_lookup(self, word, prob):
        """
        This function takes in a word and its associated probability of
        occurring within a particular DTM model topic (see get_words_for_topic).
        The word is searched for in the eurovoc terms. Each term that has this
        word as a substring is considered a MATCH. For each MATCH, the
        EUROVOC_TOPIC associated with that term is incremented in a counter,
        where the counter is incremented by the word's probability.

        The intuition is that words with higher probability will weight the
        EUROVOC_TOPIC more when deciding which EUROVOC_TOPIC to use to
        categorise a dtm topic.

        complicated explanation lol
        """
        c = Counter()
        for idx, term in enumerate(self.eurovoc['TERMS (PT-NPT)'].apply(lambda x: x.lower())):
            if re.search(f"(?:^|\\s){word}(?:\\s|$)", term):
                ev_topic = self.eurovoc.loc[idx]['MT'].lower()
                c[ev_topic] += prob
        return c
    
    def eurovoc_lookup_2(self, top_words):
        c = Counter()
        top_word_dict = dict([(" ".join(y.split("_")),x) for (x,y) in top_words])
        for topic in self.eurovoc_topics:
            score = 0
            topic_ind = np.where(self.eurovoc_topics == topic)[0][0]
            matches = self.eurovoc_term_matcher(self.eurovoc_topic_docs[topic])
            for match_id,_,_ in matches:
                term = self.nlp.vocab.strings[match_id]
                weight = top_word_dict[term]
                term_ind = self.raw_term_list.index(term)
                tfidf = self.tfidf_mat[term_ind][topic_ind]
                # weighting and tf-idf 
                score = score + (weight * tfidf)
            c.update({topic: score})
        return c
        

    def get_auto_topic_name(self, top_words, i, top_n=4):
        auto_topic_suggestions = Counter()
        weighted_ev_topics = self.eurovoc_lookup_2(top_words)
        # for prob, w in top_words:
        #     weighted_topics_for_w = self.eurovoc_lookup(w, prob)
        #     auto_topic_suggestions.update(weighted_topics_for_w)
        # print([(k, round(v,2)) for k, v in weighted_ev_topics.most_common(top_n)])
        # str([(k, round(v, 2)) for k, v in auto_topic_suggestions.most_common(top_n)]) + str(i), 
        return str([(k, round(v,2)) for k, v in weighted_ev_topics.most_common(top_n)]) + str(i)

    def get_topic_names(self, auto=True, detailed=False):
        """
        
        """
        self._init_eurovoc(EUROVOC_PATH)
        topic_names = {} if detailed else []
        self.create_eurovoc_topic_term_map()
        for i in range(self.ntopics):
            word_dist_arr_ot = self.get_topic_word_distributions_ot(i)
            top_words = self.get_words_for_topic(word_dist_arr_ot, n=30, with_prob=True)
            self.get_topic_tfidf_scores(top_words, tfidf_enabled=False)
            if auto:
                curr_topic_name = self.get_auto_topic_name(top_words, i)
            else:
                curr_topic_name = str([x[1]+str(i) for x in top_words[:3]])
            if detailed:
                topic_names[curr_topic_name] = top_words
            else:
                topic_names.append(curr_topic_name)
        return topic_names
    
    def get_words_for_topic(self, word_dist_arr_ot, n=10, descending=True, with_prob=True):
        """
        This function takes in an NUM_YEARSxLEN_VOCAB array/matrix that
        represents the vocabulary distribution for each year a topic is
        calculated. It returns a list of the n most probable words for that
        particular topic and optionally their summed probabilities over all
        years.

        Args: 

        word_dist_arr_ot (np.array): This is the array containing a topics word
            distribution for each year. It takes the shape: ntimes x len(vocab)

        n (int): The number of word probabilities to return descending (bool):
            Whether to return the word probabilities in ascending or descending
            order. i.e ascending=lowest probability at index 0 and vice-versa.

        tuples (bool): Whether to return just the word text, or a tuple of word
        text and the word's total probability summed over all time spans

        Returns: Either a list of strings or a list of tuples (float, string)
        representing the summed probability of a particular word.
        """
        acc = np.sum(word_dist_arr_ot, axis=0)
        word_dist_arr_sorted = acc.argsort()
        if descending:
            word_dist_arr_sorted = np.flip(word_dist_arr_sorted)
        top_pw = [acc[i] for i in word_dist_arr_sorted[:n]]
        top_words = [self.index_to_word[i] for i in word_dist_arr_sorted[:n]]
        if with_prob:
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
                # find top 10 most pertinent
                topws = word_dist_arr[year_idx].argsort()[-10:]
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
    
    def time_evolution_plot(self, dfs, filename, title=None, scale=1):
        sns.set_context("talk")
        sns.set_style("ticks")
        sns.set_style({'axes.spines.bottom': True,
                    'axes.grid':True,
                    'axes.spines.left': False,
                    'axes.spines.right': False,
                    'axes.spines.top': False,
                    'ytick.left': False,
                    'figure.facecolor':'w'})
        fig = plt.figure(figsize=(30, 1.7 * len(dfs.columns)))
        ax = fig.gca()
        #ax.autoscale(enable=False)
        #ax.set_ylim([0 - max_val, len(dfs.index) + max_val])
        #ax.set(ylim=(0 - max_val, len(dfs.index) + max_val))
        plt.yticks([])
        plt.xticks(range(1,len(dfs)))
        max_val = scale * dfs.max().max() + 5
        #print(max_val)

        for i, t in enumerate(reversed(dfs.columns)):
            plt.fill_between(dfs.index, dfs[t] + i*max_val, i*max_val - dfs[t], label=t)
            plt.text(20, (i+0.) *max_val, t)

        plt.xlabel('Year')
        if title:
            plt.title(title)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    
    def get_sorted_columns(self, df):
        
        # sort according to position of peak
        sel2 = df.copy()
        sel2.loc['peak_pos'] = [sel2[topic].idxmax() for topic in sel2.columns]
        sel2 = sel2.sort_values(by='peak_pos', axis=1)
        sel2 = sel2.drop('peak_pos')
        return sel2.columns
    
    def plot_topics_ot(self, save_path):
        df_scores = self.create_plottable_topic_proportion_ot_df()
        for i in df_scores.index:
            df_scores.loc[i] = df_scores.loc[i] / df_scores.loc[i].sum() * 100
        sorted_selection = self.get_sorted_columns(df_scores)
        self.time_evolution_plot(df_scores[sorted_selection], save_path, scale=1.3)
        return
        

if __name__ == "__main__":
    NDOCS = 19971 # number of lines in -mult.dat file.
    NTOPICS = 30
    tdma = TDMAnalysis(
        NDOCS, 
        NTOPICS,
        model_root=os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all_bigram"),
        doc_year_map_file_name="eiajournal-year.dat",
        seq_dat_file_name="eiajournal-seq.dat",
        vocab_file_name="vocab.txt",
        model_out_dir="model_run_topics30_alpha0.01_topic_var0.05",
        eurovoc_whitelist=False,
        )
    topic_names = tdma.get_topic_names(auto=True, detailed=False)
    # breakpoint()
    # df = tdma.create_top_words_df(n=20)
    # breakpoint()
    # for i in range(10):
    #     word_dist_arr_ot = tdma.get_topic_word_distributions_ot(i)
    #     top_words = tdma.get_words_for_topic(word_dist_arr_ot, n=6, with_prob=True)
    #     break
        # print(top_words)
    # df = tdma.create_topic_proportions_per_year_df(merge_topics=True)
    # breakpoint()