import numpy as np
import os
from sys import exit
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import visualisation as v

# local
from analysis import TDMAnalysis

EUROVOC_PATH = os.path.join(os.environ['DTM_ROOT'], "dtm", "eurovoc_export_en.csv")

class AcrossModelAnalysis:
    """ This class analyses the similarities and differences between two DTM
    models. Here we want to achieve:
        1. Clustering of topics from both models to see which topics are similar between models. Doing
            this in a reproducible and robust way will allow us to draw conclusions of how similar topics
            from different models are impacted by changes in the other. A good example of this is seeing how
            changes in research affect changes in grey literature in similar areas.
        2. Evaluating our clustering technique by:
            a. Silhouette Analysis(?)
            b. PCA plotting to assess effectiveness of clustering and potentially between clustering methods.
        3. This clustering technique can also provide a way to evaluate our auto-assigned EuroVoc
            topics. Where topics that are clustered together should also have reasonably similar
            EuroVoc topic labels.
    
    We group the topics from each model into clusters using kmeans.
    """
    def __init__(self, m1, m2, m1_alias="M1", m2_alias="M2"):
        """
        pass two model parameter dictionaries into this class in order to create analysis classes
        for each and begin the process of analysing between the two topics. 
        """
        self.m1_alias = m1_alias
        self.m2_alias = m2_alias
        m1_ntopics, m2_ntopics = self._get_models_n_topics(m1, m2)
        m1_ndocs, m2_ndocs = self._get_models_n_docs(m1, m2)
        self.m1 = TDMAnalysis(
            m1_ndocs, 
            m1_ntopics, 
            **m1
        )
        self.m2 = TDMAnalysis(
            m2_ndocs, 
            m2_ntopics, 
            **m2
        )
        self.m1_common = None
        self.m2_common = None
        self._create_common_vocabulary()
        self._update_model_distributions_to_common_vocab()

    def _get_models_n_topics(self, m1, m2):
        return (
            int(m1['model_out_dir'].split("_")[2].split("topics")[1]),
            int(m2['model_out_dir'].split("_")[2].split("topics")[1])
        )

    def _get_models_n_docs(self, m1, m2):
        return (
            sum([int(x) for x in open(os.path.join(m1['model_root'], m1['seq_dat_file_name'])).readlines()[1:]]),
            sum([int(x) for x in open(os.path.join(m2['model_root'], m2['seq_dat_file_name'])).readlines()[1:]])
        )
    
    def _create_common_vocabulary(self):
        """
        In this function we create a 'common' vocabulary between the two 
        models by taking the union of the two individual model vocabularies. 
        At the same time we also create a mapping between each word in the 
        common vocabulary and it's index.
        """
        self.common_vocabulary = np.array(list(set([*self.m1.vocab, *self.m2.vocab])))
        self.w_ind_map = {w:i for i,w in enumerate(self.common_vocabulary)}
    
    def _update_model_distributions_to_common_vocab(self):
        """
        In this function, since we now have a common vocabulary between
        the two models, we want to augment their existing per-topic word 
        distributions over their vocabularies, to per-topic word distributions 
        over the shared vocabulary so that they can be compared.

        Example:

        In a simple case, we may have two models m1 and m2 that have the
        respective vocabularies:
        
        v_m1 = ["cat", "hat", "door"]
        v_m2 = ["cat", "rat", "mouse", "jerry"]

        To simplify this example, we assume that a topic for a model is represented 
        as a distribution over its vocabulary (in reality it's the sum of a topic's 
        vocabulary distribution at each time step):
        
        k_m1 = [0.7, 0.3, 0]
        k_m2 = [0.2, 0, 0.4, 0.4]

        In order to compare how similar these topics are, we create a common vocabulary 
        by computing the union over the two vocabularies of the respective models. 
        Our next step is to augment their existing distributions so that they can be 
        compared. This is done by mapping their existing distributions to the new 
        common vocabulary.

        v_common = ["cat", "hat", "door", "rat", "mouse", "jerry"]
        k_m1_common = [0.7, 0.3, 0, 0, 0, 0]
        k_m2_common = [0.2, 0, 0, 0, 0.4, 0.4]

        we can now compute similarity measures of these same-dimension 'vectors'.
        """
        m1_common = np.zeros((self.m1.ntopics, len(self.common_vocabulary)))
        m2_common = np.zeros((self.m2.ntopics, len(self.common_vocabulary)))
        for topic_idx in range(self.m1.ntopics):
            pw_k = np.sum(self.m1.get_topic_word_distributions_ot(topic_idx), axis=0)
            for w, val in zip(self.m1.vocab, pw_k):
                m1_common[topic_idx][self.w_ind_map[w]] = val
        for topic_idx in range(self.m2.ntopics):
            pw_k = np.sum(self.m2.get_topic_word_distributions_ot(topic_idx), axis=0)
            for w, val in zip(self.m2.vocab, pw_k):
                m2_common[topic_idx][self.w_ind_map[w]] = val
        # reduce dimensionality of common vocabulary vectors
        # concat_arr = StandardScaler().fit_transform(np.concatenate((m1_common, m2_common)))
        concat_arr = np.concatenate((m1_common, m2_common))
        pca = PCA()
        dim_reduced_arr = pca.fit_transform(concat_arr)
        self.m1_common = dim_reduced_arr[:self.m1.ntopics]
        self.m2_common = dim_reduced_arr[self.m1.ntopics:self.m1.ntopics + self.m2.ntopics]
    
    def _get_similarity(self, X=None, Y=None, return_plottable=False, m1_title="Model 1 Topic", m2_title="Model 2 Topic"):
        """
        This function computes the similarity between each topic in Model 1 against each topic in Model 2. One can either
        choose to return a plottable dataframe which can be passed into seaborn heatmap function, or the raw similarity matrix
        as return by sklearn.

        Kwargs:
            return_plottable (bool, optional): Flag dictating whether to return the raw sklearn similarities or a plottable dataframe. Defaults to False.
            m1_title (str, optional): This variable outlines the title to use when referencing Model 1. This will show as the axes label on the seaborn heatmap. Defaults to "Model 1 Topic".
            m2_title (str, optional): This variable outlines the title to use when referencing Model 2. This will show as the axes label on the seaborn heatmap. Defaults to "Model 2 Topic".
        """
        if type(X) == np.ndarray and type(Y) == np.ndarray:
            sim = cosine_similarity(X, Y)
        else:
            # default is the common vocabulary vectors
            sim = cosine_similarity(self.m1_common, self.m2_common)
        if return_plottable:
            t1_val = []
            t2_val = []
            sim_val = []
            for t1_ind, matrix in enumerate(sim):
                for t2_ind, val in enumerate(matrix):
                    t1_val.append(t1_ind)
                    t2_val.append(t2_ind)
                    sim_val.append(val)
            df = pd.DataFrame({m1_title: t1_val, m2_title: t2_val, "Similarity": sim_val})
            self.heatmap_data = df.pivot(index=m1_title, columns=m2_title, values='Similarity')
            return self.heatmap_data
        else:
            return sim

    def _run_matcher(self, threshold_distance=-1):
        """
        DEPRECATED, but may be still useful

        This function simply runs a matching algorithm between the common vocabulary topics of model1
        and model2. It iterates through model1 and for each topic, finds it's most closely related topic
        by cosine similarity.

        One topic can have NONE or ONE matches.
        One topic cannot match another topic from the same model.

        Args: 
            threshold_distance (float): Must be between -1 and 1. This is the minimum distance two vectors must be similar in order to match.
        """
        if threshold_distance < -1 or threshold_distance > 1:
            print("threshold_distance must be between -1 <= threshold_distance <= 1")
            exit(1)
        matches = []
        sim = self._get_similarity(return_plottable=False)
        for topic1_ind, matrix in enumerate(sim):
            # get the highest valued topic2 match
            closest_topic2_ind = np.argmax(matrix)
            if closest_topic2_ind > threshold_distance:
                matches.append((topic1_ind, closest_topic2_ind))
            else:
                matches.append((topic1_ind, None))
        return matches
    
    def _get_eurovoc_topic_vectors(self, model, tfidf_enabled=True, scale=True):
        if type(model.eurovoc) != pd.DataFrame:
            model._init_eurovoc(EUROVOC_PATH)
            model.create_eurovoc_topic_term_map()
        topic_indices = sorted(model.eurovoc_topics)
        topic_vectors = []
        max = []
        for i in range(model.ntopics):
            word_dist_arr_ot = model.get_topic_word_distributions_ot(i)
            top_words = model.get_words_for_topic(word_dist_arr_ot, n=30, with_prob=True)
            model.get_topic_tfidf_scores(top_words, tfidf_enabled=True)
            model_ev_scores = model.get_auto_topic_name(top_words, i, stringify=False, tfidf_enabled=tfidf_enabled, return_raw_scores=True)
            max.append(model_ev_scores.most_common(1)[0][1])
            topic_vectors.append([model_ev_scores[i] for i in topic_indices])
        if scale:
            return StandardScaler().fit_transform(np.array(topic_vectors)), topic_indices
        else:
            return np.array(topic_vectors), topic_indices

    def evaluate_eurovoc_labels(self):
        # accrue all relevant vector representations of DTM topics
        m1_tfidf_vec, m1_tfidf_ind = self._get_eurovoc_topic_vectors(self.m1, scale=False)
        m1_simple_vec, m1_simple_ind = self._get_eurovoc_topic_vectors(self.m1, tfidf_enabled=False, scale=False)
        m1_simple_baseline, m1_intell_baseline = self.m1.generate_baselines()
        m2_tfidf_vec, m2_tfidf_ind = self._get_eurovoc_topic_vectors(self.m2)
        m2_simple_vec, m2_simple_ind = self._get_eurovoc_topic_vectors(self.m2, tfidf_enabled=False)
        m2_simple_baseline, m2_intell_baseline = self.m2.generate_baselines()
        assert m1_tfidf_ind == m2_tfidf_ind == m1_simple_ind == m2_simple_ind
        simple_baseline_sim = self._get_similarity(m1_simple_baseline, m2_simple_baseline, return_plottable=False)
        intell_baseline_sim = self._get_similarity(m1_intell_baseline, m2_intell_baseline, return_plottable=False)
        eurovoc_tfidf_sim = self._get_similarity(m1_tfidf_vec, m2_tfidf_vec, return_plottable=False)
        eurovoc_simple_sim = self._get_similarity(m1_simple_vec, m2_simple_vec, return_plottable=False)
        gold_standard = self._get_similarity(return_plottable=False)
        self._compare_with_gold_standard(simple_baseline_sim, gold_standard, "Simple Baseline")
        self._compare_with_gold_standard(intell_baseline_sim, gold_standard, "Intelligent Baseline")
        self._compare_with_gold_standard(eurovoc_simple_sim, gold_standard, "Eurovoc Simple")
        self._compare_with_gold_standard(eurovoc_tfidf_sim, gold_standard, "Eurovoc Tfidf")
        breakpoint()
    
    def get_similar_topics(self, threshold=0.5, gt=True, **kwargs):
        res = self._get_similarity(return_plottable=False)
        self.m1.get_top_words(with_prob=False, **kwargs)
        self.m2.get_top_words(with_prob=False, **kwargs)
        for m1_topic_ind in range(len(res)):
            for m2_topic_ind in range(len(res[m1_topic_ind])):
                if gt and res[m1_topic_ind][m2_topic_ind] > threshold:
                    print(f"{self.m1_alias} topic {m1_topic_ind} and {self.m2_alias} topic {m2_topic_ind} are similar (sim={res[m1_topic_ind][m2_topic_ind]}).")
                    print(f"{self.m1_alias} topic {m1_topic_ind} top words:")
                    print(self.m1.top_word_arr[m1_topic_ind])
                    print(f"{self.m2_alias} topic {m2_topic_ind} top words:")
                    print(self.m2.top_word_arr[m2_topic_ind])
                    print("==========")
                elif not gt and res[m1_topic_ind][m2_topic_ind] < threshold:
                    print(f"{self.m1_alias} topic {m1_topic_ind} and {self.m2_alias} topic {m2_topic_ind} are NOT similar (sim={res[m1_topic_ind][m2_topic_ind]}).")
                    print(f"{self.m1_alias} topic {m1_topic_ind} top words:")
                    print(self.m1.top_word_arr[m1_topic_ind])
                    print(f"{self.m2_alias} topic {m2_topic_ind} top words:")
                    print(self.m2.top_word_arr[m2_topic_ind])
                    print("==========")

    def _compare_with_gold_standard(self, X, gold, name):
        total_sim = self._get_similarity(X, gold, return_plottable=False)
        comparison_vals = []
        for i in range(len(total_sim)):
            comparison_vals.append(total_sim[i][i])
        average_similarity_across_methods = np.array(comparison_vals).mean()
        print("==========")
        print(f"{name} has mean similarity to gold standard of: {average_similarity_across_methods}")
    
    def get_heatmap(self, **kwargs):
        save_path = kwargs.pop("save_path", None)
        res = self._get_similarity(return_plottable=True, **kwargs)
        v.heatmap(res, save_path)
        return res
            
def test():
    m1 = {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "greyroads-year.dat",
        "seq_dat_file_name": "greyroads-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False
    }
    m2 = {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "greyroads-year.dat",
        "seq_dat_file_name": "greyroads-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False
    }
    ama = AcrossModelAnalysis(m1, m2)
    ama.m1.ntopics = 1
    ama.m2.ntopics = 1
    ama.m1.vocab = ["cat", "hat", "door"]
    ama.m2.vocab = ["cat", "rat", "mouse", "jerry"]
    ama._create_common_vocabulary()
    ama._update_model_distributions_to_common_vocab()
        

if __name__ == "__main__":
    m1 = {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "greyroads-year.dat",
        "seq_dat_file_name": "greyroads-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False
    }
    m2 = {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "eiajournal-year.dat",
        "seq_dat_file_name": "eiajournal-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False
    }
    # m2 = {
    #     "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all_bigram"),
    #     "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
    #     "doc_year_map_file_name": "greyroads-year.dat",
    #     "seq_dat_file_name": "greyroads-seq.dat",
    #     "vocab_file_name": "vocab.txt",
    #     "eurovoc_whitelist": False
    # }
    ama = AcrossModelAnalysis(m1, m2, m1_alias="IEO", m2_alias="Journals")
    # ama.get_similar_topics(gt=False, threshold=-0.2, n=20)
    # ama.get_heatmap(save_path="journals_v_aeo_heatmap.png", m1_title="Journals Topics", m2_title="AEO Topics")
    # res = ama.run_clustering()
    # ama.compare_topic_labels()
    ama.evaluate_eurovoc_labels()
    # ama.m1.get_topic_names()
    # ama.m2.get_topic_names()
    # breakpoint()
    # ama.evaluate_eurovoc_labels()

    # m1_top_words = []
    # m2_top_words = []
    # for i in range(30):
    #     word_dist_arr_ot = ama.m1.get_topic_word_distributions_ot(i)
    #     top_words = ama.m1.get_words_for_topic(word_dist_arr_ot, n=6, with_prob=False)
    #     m1_top_words.append(top_words)
    # for i in range(30):
    #     word_dist_arr_ot = ama.m2.get_topic_word_distributions_ot(i)
    #     top_words = ama.m2.get_words_for_topic(word_dist_arr_ot, n=6, with_prob=False)
    #     m2_top_words.append(top_words)
    # for m1_topic, m2_topic in res:
    #     print("==========")
    #     print(f"m1 topic: {m1_top_words[m1_topic]}")
    #     print(f"m2 topic: {m2_top_words[m2_topic]}")