#! env/bin/python3
"""
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'hansard' dataset.
"""

from dtm_creator import DTMCreator
import os

class HansardDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return [int(d.split("-")[2]) for d in self.df[date_col_name].tolist()]

def create_model_inputs(model_root, csv_path, text_col_name='section_txt', date_col_name='date', bigram=True, limit=None):
    jdtmc = HansardDTMCreator(model_root, csv_path, text_col_name=text_col_name, date_col_name=date_col_name, bigram=bigram, limit=limit)
    jdtmc.create()

if __name__ == "__main__":
    create_model_inputs(
        "dataset_20000subset_coal", 
        os.path.join(os.environ['HANSARD'], "coal_data", "04_model_inputs", "coal_full_downloaded.csv"), 
        text_col_name="main_text", 
        date_col_name="date", 
        bigram=False, 
        limit=20000
    )