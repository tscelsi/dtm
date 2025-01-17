from analysis import DTMAnalysis
from hansard_dtm_creator import HansardDTMCreator
from greyroads_dtm_creator import GreyroadsDTMCreator
from journals_dtm_creator import JournalDTMCreator
from dtm_creator import DTMCreator
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import os
import sys
import numpy as np
import traceback

class CoherenceAnalysis(DTMAnalysis):
    def __init__(self, data_path, _type, text_col_name, date_col_name, bigram, limit, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_root = kwargs.get('model_root')
        if _type == "hansard":
            self.dc = HansardDTMCreator(model_root, data_path, text_col_name, date_col_name, bigram=bigram, limit=limit)
        elif _type == "greyroads":
            self.dc = GreyroadsDTMCreator(model_root, data_path, text_col_name, date_col_name, bigram=bigram, limit=limit)
        elif _type == "journals":
            self.dc = JournalDTMCreator(model_root, data_path, text_col_name, date_col_name, bigram=bigram, limit=limit)
        elif _type == "reverse":
            self.dc = None
        else:
            print("need to specify one of 'hansard|greyroads|journals' as type.")
            sys.exit(1)
        
    def init_coherence(self, mult_path=None, vocab_path=None, **kwargs):
        if mult_path and vocab_path:
            self.paras_processed = DTMCreator.get_paras_from_mult_dat(mult_path, vocab_path)
        elif self.dc:
            print("preprocessing paragraphs...")
            self.dc.preprocess_paras(write_vocab=False, **kwargs)
        else:
            print("init incorrect.")
            sys.exit(1)
    
    def get_coherence(self, coherence='c_uci'):
        # topics = self.top_words[self.top_words['year'] == str(year)]['top_words'].tolist()
        # dates_arr = np.array(self.dc.years_final)
        # year_indexes = np.where(dates_arr == year)
        paras_processed = []
        topics = []
        proportions = np.array(self._get_topic_proportions_per_year(logged=True).tolist())
        for i in range(self.ntopics):
            word_dist_arr_ot = self.get_topic_word_distributions_ot(i)
            top_words = self.get_words_for_topic(word_dist_arr_ot, n=20, timestep_proportions=np.array(proportions[:,i]), with_prob=False)
            topics.append(top_words)
        iterator = self.dc.paras_processed if self.dc else self.paras_processed
        for para in iterator:
            merged_para = []
            for sent in para:
                merged_para.extend(sent)
            paras_processed.append(merged_para)

        # paras_arr = np.array(paras_processed)[year_indexes[0]].tolist()
        dictionary = Dictionary(documents=paras_processed)
        # corpus = [dictionary.doc2bow(doc) for doc in paras_processed]
        try:
            cm = CoherenceModel(topics=topics, dictionary=dictionary, texts=paras_processed, coherence=coherence)
        except Exception as e:
            traceback.print_exc()
        coherence = cm.get_coherence()
        return coherence



if __name__ == "__main__":
    NDOCS = 12271 # number of lines in -mult.dat file.
    NTOPICS = 10
    data_path = os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "energy_policy_applied_energy__solar_journals.csv")
    coh = CoherenceAnalysis(
        data_path,
        NDOCS, 
        NTOPICS,
        model_root=os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy__solar_seeded"),
        doc_year_map_file_name="eiajournal-year.dat",
        seq_dat_file_name="eiajournal-seq.dat",
        vocab_file_name="vocab.txt",
        model_out_dir="model_run")
    # coh.create_top_words_df()
    coh.init_coherence()
    coh.get_coherence(2000)