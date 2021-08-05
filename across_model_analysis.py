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
import sys
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

    EUROVOC_TOPICS = [
        '6606 energy policy',
        '4026 accounting',
        '6411 technology and technical regulations',
        '7236 political geography',
        '2016 trade',
        '6616 oil industry',
        '6621 electrical and nuclear industries',
        '5211 natural environment',
        '6811 chemistry',
        '2451 prices',
        '2026 consumption',
        '6416 research and intellectual property',
        '6831 building and public works',
        '6611 coal and mining industries',
        '6406 production',
        '7231 economic geography',
        '4811 organisation of transport',
        '6821 mechanical engineering',
        '3611 humanities',
        '0436 executive power and public service',
        '5216 deterioration of the environment',
        '2006 trade policy',
        '2446 taxation',
        '6626 soft energy',
        '6836 wood industry',
        '2421 free movement of capital',
        '4816 land transport',
        '0406 political framework',
        '2036 distributive trades',
        '3221 documentation',
        '1626 national accounts',
        '6816 iron, steel and other metal industries',
        '4021 management',
        '7621 world organisations',
        '3226 communications',
        '1611 economic conditions',
        '1221 justice',
        '1206 sources and branches of the law',
        '1621 economic structure',
        '5621 cultivation of agricultural land',
        '3606 natural and applied sciences',
        '2846 construction and town planning',
        '5206 environmental policy',
        '1631 economic analysis',
        '2426 financing and investment',
        '2416 financial institutions and credit',
        '1606 economic policy',
        '6826 electronics and electrical engineering',
        '2031 marketing'
    ]

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
        k_m1_common = [0.7, 0.3, 0.0001, 0, 0, 0]
        k_m2_common = [0.2, 0.001, 0, 0, 0.4, 0.4]


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
            m1_title = self.m1_alias if self.m1_alias else m1_title
            m2_title = self.m2_alias if self.m2_alias else m2_title
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
        topic_indices = sorted(self.EUROVOC_TOPICS)
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
        m1_simple_baseline, m1_thresh_baseline, m1_top_ten_baseline, m1_thresh_baseline_zeroed, m1_top_ten_baseline_zeroed = self.m1.generate_baselines(indices=self.EUROVOC_TOPICS)
        m2_tfidf_vec, m2_tfidf_ind = self._get_eurovoc_topic_vectors(self.m2)
        m2_simple_vec, m2_simple_ind = self._get_eurovoc_topic_vectors(self.m2, tfidf_enabled=False)
        m2_simple_baseline, m2_thresh_baseline, m2_top_ten_baseline, m2_thresh_baseline_zeroed, m2_top_ten_baseline_zeroed = self.m2.generate_baselines(indices=self.EUROVOC_TOPICS)
        assert m1_tfidf_ind == m2_tfidf_ind == m1_simple_ind == m2_simple_ind
        simple_baseline_sim = self._get_similarity(m1_simple_baseline, m2_simple_baseline, return_plottable=False)
        thresh_baseline_sim = self._get_similarity(m1_thresh_baseline, m2_thresh_baseline, return_plottable=False)
        top_ten_baseline_sim = self._get_similarity(m1_top_ten_baseline, m2_top_ten_baseline, return_plottable=False)
        thresh_zeroed_baseline_sim = self._get_similarity(m1_thresh_baseline_zeroed, m2_thresh_baseline_zeroed, return_plottable=False)
        top_ten_zeroed_baseline_sim = self._get_similarity(m1_top_ten_baseline_zeroed, m2_top_ten_baseline_zeroed, return_plottable=False)
        eurovoc_tfidf_sim = self._get_similarity(m1_tfidf_vec, m2_tfidf_vec, return_plottable=False)
        eurovoc_simple_sim = self._get_similarity(m1_simple_vec, m2_simple_vec, return_plottable=False)
        gold_standard = self._get_similarity(return_plottable=False)
        simple_base_res = self._compare_with_gold_standard(simple_baseline_sim, gold_standard, "Simple Baseline")
        thresh_base_res = self._compare_with_gold_standard(thresh_baseline_sim, gold_standard, "Intelligent Baseline")
        top_ten_base_res = self._compare_with_gold_standard(top_ten_baseline_sim, gold_standard, "Top Ten Baseline")
        thresh_zeroed_base_res = self._compare_with_gold_standard(thresh_zeroed_baseline_sim, gold_standard, "Intelligent Baseline Zeroed")
        top_ten_zeroed_base_res = self._compare_with_gold_standard(top_ten_zeroed_baseline_sim, gold_standard, "Top Ten Baseline Zeroed")
        simple_ev_res = self._compare_with_gold_standard(eurovoc_simple_sim, gold_standard, "Eurovoc Simple")
        tfidf_ev_res = self._compare_with_gold_standard(eurovoc_tfidf_sim, gold_standard, "Eurovoc Tfidf")
        return simple_base_res, thresh_base_res, top_ten_base_res, thresh_zeroed_base_res, top_ten_zeroed_base_res, simple_ev_res, tfidf_ev_res
    
    def get_similar_topics(self, threshold=0.5, gt=True, fp=sys.stdout, **kwargs):
        fp.write("#"*10 + "\n")
        fp.write(f"## FINDING SIMILAR TOPICS BETWEEN {self.m1_alias} AND {self.m2_alias} ##\n")
        fp.write("#"*10 + "\n")
        res = self._get_similarity(return_plottable=False)
        self.m1.get_top_words(with_prob=False, **kwargs)
        self.m2.get_top_words(with_prob=False, **kwargs)
        for m1_topic_ind in range(len(res)):
            for m2_topic_ind in range(len(res[m1_topic_ind])):
                if gt and res[m1_topic_ind][m2_topic_ind] > threshold:
                    fp.write(f"{self.m1_alias} topic {m1_topic_ind} and {self.m2_alias} topic {m2_topic_ind} are similar (sim={res[m1_topic_ind][m2_topic_ind]}).\n")
                    fp.write(f"{self.m1_alias} topic {m1_topic_ind} top words:\n")
                    fp.write(str(self.m1.top_word_arr[m1_topic_ind])+"\n")
                    fp.write(f"{self.m2_alias} topic {m2_topic_ind} top words:\n")
                    fp.write(str(self.m2.top_word_arr[m2_topic_ind])+"\n")
                    fp.write("==========\n")
                # elif not gt and res[m1_topic_ind][m2_topic_ind] < threshold:
                #     print(f"{self.m1_alias} topic {m1_topic_ind} and {self.m2_alias} topic {m2_topic_ind} are NOT similar (sim={res[m1_topic_ind][m2_topic_ind]}).")
                #     print(f"{self.m1_alias} topic {m1_topic_ind} top words:")
                #     print(self.m1.top_word_arr[m1_topic_ind])
                #     print(f"{self.m2_alias} topic {m2_topic_ind} top words:")
                #     print(self.m2.top_word_arr[m2_topic_ind])
                #     print("==========")

    def get_unique_topics(self, threshold=0.25, fp=sys.stdout, **kwargs):
        """This function return the topics from both model 1 that are not similar to any topics within model 2 and vice-versa. Thus allowing
        us to see topics that are unique between publications/datasets. i.e. if a topic appears in model 1 and not in model 2, then that topic is unique
        to the discussions found in model 1's dataset.
        """
        fp.write("#"*10 + "\n")
        fp.write(f"## FINDING UNIQUE TOPICS IN {self.m1_alias} AND {self.m2_alias} ##\n")
        fp.write("#"*10+ "\n")
        def print_unique_topic(ind, max_sim, model_name):
            if model_name == "m1":
                model = self.m1
                alias = self.m1_alias
            else:
                model = self.m2
                alias = self.m2_alias
            fp.write(f"Topic {ind} in {alias} is unique (max_sim={max_sim}).\n")
            fp.write("---\n")
            fp.write(f"Topic label: {model.topic_names[ind]}\n")
            fp.write("---\n")
            fp.write(f"Topic word list: {model.top_word_arr[ind]}\n")
            fp.write("==========\n")
        m1_unique_topics = []
        m2_unique_topics = []
        res = self._get_similarity(return_plottable=False)
        res_T = res.T
        self.m1.get_topic_names(**kwargs)
        self.m2.get_topic_names(**kwargs)
        for m1_topic_ind in range(len(res)):
            max_sim = res[m1_topic_ind].max()
            if max_sim <= threshold:
                m1_unique_topics.append(m1_topic_ind)
                print_unique_topic(m1_topic_ind, max_sim, "m1")
        for m2_topic_ind in range(len(res_T)):
            max_sim = res_T[m2_topic_ind].max()
            if max_sim <= threshold:
                m2_unique_topics.append(m2_topic_ind)
                print_unique_topic(m2_topic_ind, max_sim, "m2")

    def _compare_with_gold_standard(self, X, gold, name):
        total_sim = self._get_similarity(X, gold, return_plottable=False)
        total_sim_T = self._get_similarity(X.T, gold.T, return_plottable=False)
        comparison_vals = []
        # assuming that both models have same number of topics...
        for i in range(len(total_sim)):
            comparison_vals.append(total_sim[i][i])
            comparison_vals.append(total_sim_T[i][i])

        average_similarity_across_methods = np.array(comparison_vals).mean()
        # print("==========")
        # print(f"{name} has mean similarity to gold standard of: {average_similarity_across_methods}")
        return average_similarity_across_methods
    
    def get_heatmap(self, **kwargs):
        save_path = kwargs.pop("save_path", None)
        res = self._get_similarity(return_plottable=True, **kwargs)
        v.heatmap(res, save_path)
        return res

def compare_all_models():
    models = [{
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "greyroads-year.dat",
        "seq_dat_file_name": "greyroads-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False,
        "alias": "IEO"
    }, {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "greyroads-year.dat",
        "seq_dat_file_name": "greyroads-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False,
        "alias": "AEO"
    }, {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "eiajournal-year.dat",
        "seq_dat_file_name": "eiajournal-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False,
        "alias": "Journals - All"
    }, {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_biofuels_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "eiajournal-year.dat",
        "seq_dat_file_name": "eiajournal-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False,
        "alias": "Journals - Biofuels"
    }, {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_solar_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "eiajournal-year.dat",
        "seq_dat_file_name": "eiajournal-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False,
        "alias": "Journals - Solar"
    }, {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_coal_bigram"),
        "model_out_dir": "model_run_topics30_alpha0.01_topic_var0.05", 
        "doc_year_map_file_name": "eiajournal-year.dat",
        "seq_dat_file_name": "eiajournal-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": False,
        "alias": "Journals - Coal"
    }]
    m1_list = []
    m2_list = []
    simple_baseline = []
    thresh_baseline = []
    thresh_baseline_zeroed = []
    top_ten_baseline = []
    top_ten_baseline_zeroed = []
    simple_ev = []
    tfidf_ev = []
    # simple_base_res, thresh_base_res, top_ten_base_res, thresh_zeroed_base_res, top_ten_zeroed_base_res, simple_ev_res, tfidf_ev_res
    for i in range(len(models)):
        for j in range(len(models)):
            if j <= i:
                continue
            m1 = models[i]
            m2 = models[j]
            m1_name = m1['alias']
            m2_name = m2['alias']
            print(f"{m1_name} vs {m2_name}")
            m1_list.append(m1_name)
            m2_list.append(m2_name)
            ama = AcrossModelAnalysis(m1, m2, m1_alias=m1_name, m2_alias=m2_name)
            res = get_model_summary(ama)
            
            simple_baseline.append(res[0])
            thresh_baseline.append(res[1])
            top_ten_baseline.append(res[2])
            thresh_baseline_zeroed.append(res[3])
            top_ten_baseline_zeroed.append(res[4])
            simple_ev.append(res[5])
            tfidf_ev.append(res[6])
    df = pd.DataFrame({
        "Model 1": m1_list, 
        "Model 2": m2_list, 
        "Simple Base": simple_baseline, 
        "Thresh. Base": thresh_baseline, 
        "Top Ten Base": top_ten_baseline, 
        "Thresh. Base Zeroed": thresh_baseline_zeroed, 
        "Top Ten Base Zeroed": top_ten_baseline_zeroed, 
        "Simple EV": simple_ev, 
        "Tfidf EV": tfidf_ev})
    df.to_csv("ev_comparison_matrix.csv")
    breakpoint()

def get_model_summary(ama):
    res = ama.evaluate_eurovoc_labels()
    with open(f"{ama.m1_alias}_{ama.m2_alias}_similar_topics.txt", "w+") as fp:
        ama.get_similar_topics(fp=fp)
    with open(f"{ama.m1_alias}_{ama.m2_alias}_unique_topics.txt", "w+") as fp:
        ama.get_unique_topics(fp=fp)
    # ama.get_heatmap(save_path=f"{'_'.join(ama.m1_alias.split(' '))}_v_{'_'.join(ama.m2_alias.split(' '))}")
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
    # ama = AcrossModelAnalysis(m1, m2, m1_alias="IEO", m2_alias="Journals")
    # ama.m1._init_eurovoc(EUROVOC_PATH)
    # ama.m1.create_eurovoc_topic_term_map()
    # ama.m1.generate_baselines()
    # breakpoint()
    # ama.get_unique_topics(threshold=0.1)
    # res = ama._get_similarity()
    # print(res[0][29])
    # print(res[29][0])
    # breakpoint()
    compare_all_models()

    # ama.get_similar_topics(gt=False, threshold=-0.2, n=20)
    # ama.get_heatmap(save_path="journals_v_aeo_heatmap.png", m1_title="Journals Topics", m2_title="AEO Topics")
    # res = ama.run_clustering()
    # ama.compare_topic_labels()
    # ama.evaluate_eurovoc_labels()
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