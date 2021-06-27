#! env/bin/python3
"""
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'greyroads' dataset as mentioned in the
accompanying paper.
"""

from .dtm_creator import DTMCreator
import os

class GreyroadsDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return [int(x) for x in self.df[date_col_name].tolist()]


def create_model_inputs(model_root, csv_path, bigram=True, limit=None, min_freq=150, ds_upper_limit=1000):
    jdtmc = GreyroadsDTMCreator(model_root, csv_path, text_col_name='para_text', date_col_name='year', bigram=bigram, limit=limit)
    jdtmc.create(min_freq, ds_upper_limit)

if __name__ == "__main__":
    # replace
    create_model_inputs(
        "greyroads_aeo_all_bigram_downsampled_100", 
        os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all.csv"), 
        bigram=True, 
        limit=None,
        ds_upper_limit=100
    )