#! env/bin/python3
"""
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'greyroads' dataset as mentioned in the
accompanying paper.
"""

from dtm_creator import DTMCreator


class GreyroadsDTMCreator(DTMCreator):
    def _extract_date(self, date_col_name):
        return [int(x) for x in self.df[date_col_name].tolist()]


def create_model_inputs(model_root, csv_path, bigram=True, limit=None):
    jdtmc = JournalDTMCreator(model_root, csv_path, text_col_name='para_text', date_col_name='year', bigram=bigram, limit=limit)
    jdtmc.create()

if __name__ == "__main__":
    # replace
    create_model_inputs(*args, **kwargs)