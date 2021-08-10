#! env/bin/python3
""" 
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'journals' dataset as mentioned in the
accompanying paper.
"""

from dtm_creator import DTMCreator
import os
import pandas as pd

class CivilityDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return [int(d.split("-")[0]) for d in self.df[date_col_name].tolist()]
    
    @classmethod
    def split_docs(self, docs, n=300):
        new_docs = []
        for doc,year in docs:
            toks = doc.split(" ")
            if len(toks) > n:
                ind = 0
                remainder = len(toks)
                while ind<len(toks):
                    new_docs.append((" ".join(toks[ind:ind+n]), year))
                    ind += n
                    remainder -= n
            else:
                new_docs.append((" ".join(toks), year))
        return new_docs
        


def create_model_inputs(model_root, csv_path, bigram=True, limit=None, min_freq=150, ds_upper_limit=1000):
    jdtmc = CivilityDTMCreator(model_root, csv_path, text_col_name='main_text', date_col_name='date', bigram=bigram, limit=limit)
    jdtmc.create(min_freq, ds_upper_limit)


if __name__ == "__main__":
    # replace
    # create_model_inputs(
    #     os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "Civility_energy_policy_applied_energy_all_years_abstract_all_bigram_2"), 
    #     os.path.join(os.environ['ROADMAP_SCRAPER'], "Civilitys", "Civilitys_energy_policy_applied_energy_all_years_abstract.csv"), 
    #     bigram=True, 
    #     limit=None)
    # docs = [("Hey this is tom i am a whore",2020),("hi i'm zac and yu're miri lets make porn",2020),("my dear mr watson",2020),("understandable, she didn't like playing with kites",2020)]
    # docs = CivilityDTMCreator.split_docs(docs, n=4)
    df = pd.read_csv(os.path.join(os.environ['ROADMAP_SCRAPER'], "civility_filtered_gt_600.csv"))
    docs = []
    dates = [int(d.split("-")[0]) for d in df['date'].tolist()]
    for doc, year in zip(df['main_text'].tolist(),dates):
        docs.append((doc,year))
    docs = CivilityDTMCreator.split_docs(docs, n=500)
    print(f"turned {len(df['main_text'])} into {len(docs)}...")
    civility = CivilityDTMCreator("civility_2011_2020_gt_600_ngram_min_freq_20", docs, text_col_name='main_text', date_col_name='date', bigram=True, limit=90000, spacy_batch_size=128) 
    # civility = CivilityDTMCreator("civility_2011_2020_gt_600_ngram", os.path.join(os.environ['ROADMAP_SCRAPER'], "civility_filtered_gt_600.csv"), text_col_name='main_text', date_col_name='date', bigram=True, limit=None, spacy_batch_size=64)
    civility.preprocess_paras(min_freq=20, ngrams=True, write_vocab=True)