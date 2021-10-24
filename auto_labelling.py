from collections import Counter
from spacy.matcher import PhraseMatcher
import spacy
import numpy as np
import sys
import pandas as pd
from spacy.tokens import Doc
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import gensim.downloader
import pickle


class AutoLabel:
    DEFAULT_PHRASE_COL = "phrase"
    DEFAULT_LABEL_COL = "label"

    def __init__(self, thesaurus, phrase_col=None, label_col=None, spacy_lang="en_core_web_sm"):
        """Here we initialise the auto labelling class. It is passed a thesaurus-like object
        that maps phrases to overarching thematic labels. Either pass a dataframe and specify the
        phrase and label columns, or pass a dict with phrases mapped to labels.

        if dict, we expect:

                PHRASE      LABEL
            {
                "climate": "environment",
                "coal": "mining",
                "coke coal": "mining",
                "photovoltaic": "renewables",
                ...
            }
        
        if dataframe, we expect something like:

            phrase  label
            0   climate       environment
            1   coal          mining
            2   coke coal     mining
            3   photovoltaic  renewables

        Args:
            thesaurus (pd.DataFrame | dict): A thesaurus-like object mapping phrases to labels.
        """
        self.nlp = spacy.load(spacy_lang)
        if isinstance(thesaurus, dict):
            self.thes = pd.DataFrame(thesaurus.items(), columns=[self.DEFAULT_PHRASE_COL, self.DEFAULT_LABEL_COL])
            self.phrase_col = self.DEFAULT_PHRASE_COL
            self.label_col = self.DEFAULT_LABEL_COL
        elif isinstance(thesaurus, pd.DataFrame):
            if not phrase_col or not label_col:
                print("need to specify the phrase and label columns of the dataframe.")
                sys.exit(1)
            else:
                self.phrase_col = phrase_col
                self.label_col = label_col
                self.thes = thesaurus.loc[:, [self.phrase_col, self.label_col]]
        # creates sorted_labels, thes_label_docs
        self._create_label_phrase_docs()
        # init in case we use embedding approach for auto-labelling
        self.phrase_embeddings = None

    def _create_phrase_embeddings(self):
        """This function creates an K x p_k x gloVedims embedding matrix where K is the number of labels in the thesaurus,
        T_k is the number of phrases under a label k and gloVedims is the dimensions of the pre-trained gloVe word embeddings 
        as per gensim docs.
        """
        self.label_term_map = {}
        self.phrase_embeddings = []
        for label in self.sorted_labels:
            mask = self.thes[self.label_col].apply(lambda x: x.lower()) == label
            terms = self.nlp.pipe([x.lower() for x in self.thes[mask][self.phrase_col]], n_process=11)
            term_vec_matrix = []
            term_list = []
            for term in terms:
                vec = self._get_vector_from_tokens(term)
                # take average of vectors to achieve embedding for term
                if type(vec) == np.ndarray:
                    term_vec_matrix.append(vec)
                    term_list.append(term.text)
            self.label_term_map[label] = term_list
            self.phrase_embeddings.append(term_vec_matrix)
        self.phrase_embeddings = np.array(self.phrase_embeddings)

    def _init_embeddings(self, load_embeddings, save_embeddings=True, phrase_embedding_path="phrase_embeddings.pickle", emb_type='glove-wiki-gigaword-50'):
        """This function creates a class-accessible embedding matrix of the phrases under each label in the thesaurus. For more information
        see _create_phrase_embeddings function.

        Args:
            load_embeddings (bool): Whether to load the phrase embeddings from phrase_embedding_path
            save_embeddings (bool, optional): Whether to save the phrase embeddings to phrase_embedding_path. Only checked if load is False. Defaults to True.
            phrase_embedding_path (str, optional): The path with which to load or save the phrase embeddings. Defaults to "phrase_embeddings.pickle".
            emb_type (str, optional): string outlining which gensim pre-trained embeddings to use. Defaults to 'glove-wiki-gigaword-50'.
        """
        print("Initialising gloVe embeddings...")
        self.embeddings = gensim.downloader.load(emb_type)
        if not load_embeddings:
            self._create_phrase_embeddings()
            if save_embeddings and (not phrase_embedding_path):
                print("If you want to save the embeddings, you need to provide a phrase_embedding_path value.")
            elif save_embeddings and phrase_embedding_path:
                with open(phrase_embedding_path, "wb+") as fp:
                    pickle.dump(self.phrase_embeddings, fp)
        elif load_embeddings and not phrase_embedding_path:
            print("If loading the eurovoc embeddings matrix, you need to provide a path for the embedding matrix.")
            sys.exit(1)
        elif load_embeddings:
            try:
                with open(phrase_embedding_path, "rb") as fp:
                    self.phrase_embeddings = np.array(pickle.load(fp))
            except Exception as e:
                print("Try setting load_embeddings=False, save=True to save your initialised embeddings for future use")
                sys.exit(1)

    def _create_label_phrase_docs(self):
        """This function aggregates all the phrases for a particular label into a 'document'
        where a 'document' just represents each phrase concatenated together with spaces in between.
        This is a more feasible approach to match terms together with spaCy.
        """
        label_term_map = {}
        self.thes_label_docs = {}
        for term, label in zip(self.nlp.pipe(self.thes[self.phrase_col], disable=['tok2vec', 'ner'], batch_size=256, n_process=11), self.thes[self.label_col].apply(lambda x: x.lower())):
            if label in label_term_map:
                label_term_map[label].append(term)
            else:
                label_term_map[label] = [term]
        for label in label_term_map:
            curr_label_list = label_term_map[label]
            c_doc = Doc.from_docs(curr_label_list, ensure_whitespace=True)
            self.thes_label_docs[label] = c_doc
        self.sorted_labels = sorted(np.array(list(self.thes_label_docs.keys())))

    def _get_topic_tfidf_scores(self, top_terms):
        """
        Returns a matrix for a DTM topic where the rows represent a top term for the dtm topic, and the columns
        represent each EuroVoc topic. Each cell is the tfidf value of a particular term-topic
        combination. This will be used when calculating the automatic EuroVoc topic labels for the
        DTM topics.

        i.e. shape is | top_terms | x | EuroVoc Topics |
        """
        self.tfidf_mat = np.zeros((len(top_terms), len(self.sorted_labels)))
        # number of documents containing a term
        N = Counter()
        tfs = {}
        doc_lens = {}
        # ensure to get rid of the underscores in bigram terms and then rejoin with space
        # i.e. 'greenhouse_gas' becomes 'greenhouse gas'
        spacy_terms = [t for t in self.nlp.pipe([" ".join(t.split("_")) for _, t in top_terms], disable=['tok2vec', 'ner'], n_process=11)]
        self.raw_term_list = [t.text for t in spacy_terms]
        self.thes_term_matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA", validate=True)
        for term in spacy_terms:
            self.thes_term_matcher.add(term.text, [term])
        ## term freq
        ## total terms in each topic (doc)
        ## number of docs that match term
        ## total number of docs
        for label in self.sorted_labels:
            term_freq = Counter()
            curr_doc = self.thes_label_docs[label]
            doc_len = len(self.thes_label_docs[label])
            doc_lens[label] = doc_len
            terms_contained_in_label = set()
            matches = self.thes_term_matcher(curr_doc)
            if matches:
                for match_id, _, _ in matches:
                    matched_term = self.nlp.vocab.strings[match_id]
                    terms_contained_in_label.add(matched_term)
                    term_freq[matched_term] += 1
                tfs[label] = term_freq
            N.update(terms_contained_in_label)
        # calculate tfidfs for each term in each topic
        for i, label in enumerate(self.sorted_labels):
            try:
                curr_tfs = tfs[label]
                doc_len = doc_lens[label]
                for term in curr_tfs:
                    ind = self.raw_term_list.index(term)
                    tf = curr_tfs[term] / doc_len
                    idf = np.log(len(self.thes_label_docs.keys()) / N[term])
                    tfidf = tf * idf
                    self.tfidf_mat[ind][i] = tfidf
            except KeyError as e:
                continue
        return self.tfidf_mat

    def _get_vector_from_tokens(self, tokens):
        """Creates an embedding for a phrase with
        any number of tokens by taking the mean
        of each tokens embedding.

        Args:
            tokens ([spacy.Token]): list of spacy tokens

        Returns:
            np.array: embedding vector summarising a phrase
        """
        vec = []
        for tok in tokens:
            if tok.lemma_ in self.embeddings:
                vec.append(self.embeddings[tok.lemma_])
        if vec:
            vec = np.array(vec).mean(axis=0)
        return vec

    def _get_tfidf_scores(self, top_words, **kwargs):
        """This function calculates the scores between a topics top terms and
        each thesaurus label using the match-based approach. It returns a 
        dict-like object of labels with their associated scores. 
        The higher the score, the more similar that label to the topic.

        Args:
            top_words (list): The n top terms for a topic

        Returns:
            Counter: A Counter (dict-like) object with scores between the top 
            words of a topic and each thesaurus label.
        """
        c = Counter()
        terms = Counter()
        top_word_dict = dict([(" ".join(y.split("_")),x) for (x,y) in top_words])
        for i, label in enumerate(self.sorted_labels):
            score = 0
            matches = self.thes_term_matcher(self.thes_label_docs[label])
            for match_id,_,_ in matches:
                term = self.nlp.vocab.strings[match_id]
                terms[term] += 1
                weight = top_word_dict[term]
                term_ind = self.raw_term_list.index(term)
                tfidf = self.tfidf_mat[term_ind][i]
                # weighting and tf-idf
                score = score + (weight * tfidf)
            c.update({label: score})
        return c

    def _get_embedding_scores(self, top_words, **kwargs):
        """Uses the embedding approach to automatically label topics. Calculates
        scores between a topic and each label. It returns a dict-like object of 
        labels with their associated scores. The higher the score, the more similar
        that label to the topic.

        Args:
            top_words (list): The n top terms for a topic

        Returns:
            Counter: A Counter (dict-like) object with scores between the top 
            words of a topic and each thesaurus label.
        """
        c = Counter()
        if isinstance(self.phrase_embeddings, type(None)):
            self._init_embeddings(**kwargs)
        words = [self.nlp(" ".join(w.split("_"))) for _,w in top_words]
        word_probs = np.array([x for (x,_) in top_words])
        word_vecs = []
        unweighted_word_vecs = []
        for i,w in enumerate(words):
            vec = self._get_vector_from_tokens(w)
            if type(vec) != list:
                word_vecs.append(vec*word_probs[i])
        word_vecs = np.array(word_vecs)
        word_vec = word_vecs.mean(axis=0).reshape(1,-1)
        for i, topic in enumerate(self.phrase_embeddings):
            # pairwise cosine sim between top words and topic term vectors
            topic_mat = np.array(topic)
            topic_vec = topic_mat.mean(axis=0).reshape(1,-1)
            score = cosine_similarity(word_vec, topic_vec)[0][0]
            # score = np.multiply(scores.squeeze(), word_probs).sum()
            topic_name = self.sorted_labels[i]
            c.update({topic_name: score})
        return c

    def _get_auto_topic_name(self, top_words, top_n, raw, score_type, **kwargs):
        if score_type == "tfidf":
            weighted_ev_topics = self._get_tfidf_scores(top_words, **kwargs)
        elif score_type == "embedding":
            weighted_ev_topics = self._get_embedding_scores(top_words, **kwargs)
        else:
            print("score_type needs to be one of either tfidf|embedding")
            sys.exit(1)
        if raw:
            return weighted_ev_topics
        else:
            return [(k, round(v,2)) for k, v in weighted_ev_topics.most_common(top_n)]
    
    def get_topic_names(self, topics, top_n=1, score_type="embedding", raw=False, **kwargs):
        """Main class function. Pass in a list of topics represented as top-n terms and their probabilities
        and this function assigns automatic labels to each of the topics.

        A topic representation should look like, see auto_labelling_demo.ipynb for more examples:

        [
            (0.2814950666697298, 'power'), 
            (0.2414588962377678, 'system'),
            (0.11754153696260229, 'heat'), 
            (0.07225017488814223, 'generation'), 
            (0.06429135899845657, 'electricity'), 
            (0.057083478661588666, 'chp'), 
            (0.05036248076080362, 'energy'), 
            (0.04161238301685071, 'electric'), 
            (0.03957228287328841, 'district_heating'), 
            (0.03433234093076981, 'electrical')
        ]

        Args:
            topics (list): list of topics represented as top terms. i.e. a list of lists of topic terms
            top_n (int, optional): Number of labels to return as the topic label. Defaults to 1.
            score_type (str, optional): Automatic labelling approach to use. Defaults to "embedding".
            raw (bool, optional): Whether to return all labels with scores instead of just the top_n labels. Defaults to False.

        Returns:
            list: A list of automatically assigned topic names. If raw=True, then this is a list of 
            Counter objects where each object represents the label scores for the particular topic
            that it is assigning auto labels for.
        """
        topic_names = []
        for top_terms in topics:
            if score_type == "tfidf":
                self._get_topic_tfidf_scores(top_terms)
            curr_topic_name = self._get_auto_topic_name(top_terms, top_n, raw, score_type, **kwargs)
            topic_names.append(curr_topic_name)
        return topic_names