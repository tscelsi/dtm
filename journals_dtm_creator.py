#! env/bin/python3
""" 
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'journals' dataset as mentioned in the
accompanying paper.
"""

from dtm_creator import DTMCreator
import os

class JournalDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return [int(d.split("-")[0]) for d in self.df[date_col_name].tolist()]


def create_model_inputs(model_root, csv_path, bigram=True, limit=None, min_freq=150, ds_upper_limit=1000):
    jdtmc = JournalDTMCreator(model_root, csv_path, text_col_name='section_txt', date_col_name='date', bigram=bigram, limit=limit)
    jdtmc.create(min_freq, ds_upper_limit)


if __name__ == "__main__":
    # replace
    # create_model_inputs(
    #     os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all_bigram_2"), 
    #     os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"), 
    #     bigram=True, 
    #     limit=None)
    jdtmc = JournalDTMCreator("journal_energy_policy_applied_energy_all_years_abstract_all_ngram", os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"), text_col_name='section_txt', date_col_name='date', bigram=True, limit=None)
    jdtmc.preprocess_paras(ngrams=True, write_vocab=True)