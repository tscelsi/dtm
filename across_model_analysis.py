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
    def __init__(self, m1, m2):
        """
        pass two model parameter dictionaries into this class in order to create analysis classes
        for each and begin the process of analysing between the two topics. 
        """
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

    def _run_cluster_optimisation(self, X):
        best_cluster_n = None
        best_silhouette_avg = None
        for i in range(5, self.m1.ntopics):
            clusterer = KMeans(n_clusters=i, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", i, "The average silhouette_score is :", silhouette_avg)
            if not best_silhouette_avg:
                best_silhouette_avg = silhouette_avg
                best_cluster_n = i
            elif silhouette_avg > best_silhouette_avg:
                best_silhouette_avg = silhouette_avg
                best_cluster_n = i
        return best_cluster_n

    def run_clustering(self):
        """
        Here we run kmeans on the now same shape topic vocab distributions. This function only can run once 
        _update_model_distributions_to_common_vocab function has run and the common topic distributions have been created.

        It returns a list of cluster labels for the concatenated m1 and m2 topic distributions. i.e. if m1 has 10 topics and m2 has 30 topics, 
        then it will return a list of length 40, where the first ten labels represent the clusters of the m1 topics and the last 30 the m2 topics.
        """ 
        X = np.concatenate((self.m1_common, self.m2_common), axis=0)
        n_clusters = self._run_cluster_optimisation(X)
        clusterer = KMeans(n_clusters=n_clusters)
        labels = clusterer.fit_predict(X)
        return labels[:self.m1.ntopics], labels[self.m1.ntopics:self.m2.ntopics + self.m1.ntopics]
    
    def _get_eurovoc_topic_vectors(self, model):
        model._init_eurovoc(EUROVOC_PATH)
        model.create_eurovoc_topic_term_map()
        topic_indices = sorted(model.eurovoc_topics)
        topic_vectors = []
        for i in range(model.ntopics):
            word_dist_arr_ot = model.get_topic_word_distributions_ot(i)
            top_words = model.get_words_for_topic(word_dist_arr_ot, n=30, with_prob=True)
            model.get_topic_tfidf_scores(top_words, tfidf_enabled=True)
            model_ev_scores = model.get_auto_topic_name(top_words, i, stringify=False, tfidf_enabled=False, return_raw_scores=True)
            topic_vectors.append([model_ev_scores[i] for i in topic_indices])
        return StandardScaler().fit_transform(np.array(topic_vectors)), topic_indices

    def evaluate_eurovoc_labels(self):
        m1_vec, m1_ind = self._get_eurovoc_topic_vectors(self.m1)
        m2_vec, m2_ind = self._get_eurovoc_topic_vectors(self.m2)
        assert m1_ind == m2_ind
        ev_sim = self._get_similarity(m1_vec, m2_vec, return_plottable=False)
        pw_k_sim = self._get_similarity(return_plottable=False)
        total_sim = self._get_similarity(ev_sim, pw_k_sim, return_plottable=False)
        comparison_vals = []
        for i in range(len(total_sim)):
            comparison_vals.append(total_sim[i][i])
        average_similarity_across_methods = np.array(comparison_vals).mean()
        breakpoint()
        
            



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
    # m2 = {
    #     "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all_bigram"),
    #     "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
    #     "doc_year_map_file_name": "eiajournal-year.dat",
    #     "seq_dat_file_name": "eiajournal-seq.dat",
    #     "vocab_file_name": "vocab.txt",
    #     "eurovoc_whitelist": False
    # }
    m2 = {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "greyroads-year.dat",
        "seq_dat_file_name": "greyroads-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False
    }
    ama = AcrossModelAnalysis(m1, m2)
    # res = ama.run_clustering()
    res = ama._get_similarity(return_plottable=True, m1_title="Greyroads IEO", m2_title="Greyroads AEO")
    # v.heatmap(res, save_path="singledim.png")
    ama.evaluate_eurovoc_labels()
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