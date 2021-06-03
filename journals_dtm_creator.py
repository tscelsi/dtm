#! env/bin/python3
""" 
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'journals' dataset as mentioned in the
accompanying paper.
"""

from dtm_creator import DTMCreator


class JournalDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return [int(d.split("-")[0]) for d in self.df[date_col_name].tolist()]


def create_model_inputs(model_root, csv_path, bigram=True, limit=None):
    jdtmc = JournalDTMCreator(model_root, csv_path, text_col_name='section_txt', date_col_name='date', bigram=bigram, limit=limit)
    jdtmc.create()


if __name__ == "__main__":
    # replace
    create_model_inputs(*args, **kwargs)