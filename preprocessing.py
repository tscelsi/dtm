from spacy.lang.en.stop_words import STOP_WORDS
import re
from collections import Counter
import os
import spacy


class Preprocessing:
    preproc_ngram_path = os.path.join(os.environ['ROADMAP_SCRAPER'], "NGRAMS_PREPROC.txt")

    def __init__(self, docs, term_blacklist=[], verbose=True):
        self.docs = docs
        self.term_blacklist = term_blacklist
        self.verbose = verbose
    
    def _print_progress(self, n, total, increment=0.25):
        if n == int(total)*0.25:
            print("25%...")
        elif n == int(total)*0.5:
            print("50%...")
        elif n == int(total)*0.75:
            print("75%...")

    def __add_bigrams(self, ngrams=False):
        """A spaCy approach to bigramming a corpus. DEPRECATED in favour of a faster and less involved approach.
        
        This function retokenises spaCy documents to include bigrams (and ngrams if ngrams enabled) instead of just
        single word tokens. It does this by referencing a corpus of pre-computed ngrams and finding matches of these ngrams
        in the corpus using the spaCy PhraseMatcher class. Take a paragraph:

        "The cat realised that climate change was important."

        In the above sentence, we may have the phrase 'climate change' in our ngram reference corpus. spaCy has initially tokenised the sentence as:

        ['the', 'cat', 'realised', 'that', 'climate', 'change', 'was', 'important', '.']

        We want to have spaCy pick up 'climate change' as a single bigram. Thus we use the PhraseMatcher to find it in the original sentence, and retokenise
        it. Such that the above tokenisation becomes:

        ['the', 'cat', 'realised', 'that', 'climate change', 'was', 'important', '.']

        We change the 'LEMMA' property of each token to represent the underscore connected ngram. For the 'climate change' token:

        tok.lemma_ = 'climate_change'

        Args:
            ngrams (bool, optional): If set to true, instead of using just bigrams from bigram_path, ngrams will be matched
                as well based on the ngrams found at ngram_path. Defaults to False.
        """
        self.ngram_match_counts = Counter()
        # read in ngram file
        if ngrams:
            bigrams = [x.strip("\n") for x in open(ngram_path, "r").readlines()]
        else:
            bigrams = [x.strip("\n") for x in open(bigram_path, "r").readlines()]
        # init matcher with all the ngrams
        bigram_matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        for bigram in self.nlp.pipe(bigrams, n_process=11, batch_size=self.spacy_batch_size):
            bigram_matcher.add(bigram.text,[bigram])
        new_paras = []
        count = 0
        for para in self.rdocs:
            if count % 1000 == 0:
                print(f"para {count}")
            count += 1
            matches = bigram_matcher(para)
            match_mapping = {para[start:end].lemma_:_id for _id, start, end in matches}
            # we remove overlapping matches taking the longest match as the source of truth
            # i.e. if we match 'climate change' and 'climate', we take 'climate change'.
            filtered_spans = filter_spans([para[start:end] for _,start,end in matches])
            if filtered_spans:
                with para.retokenize() as r:
                    for span in filtered_spans:
                        _id = match_mapping[span.lemma_]
                        match_replace_with = self.nlp.vocab.strings[_id].replace(" ", "_")
                        r.merge(span, attrs={"LEMMA": match_replace_with})
                        self.ngram_match_counts.update([self.nlp.vocab.strings[_id]])
            new_paras.append(para)
        self.rdocs = new_paras

    def __add_bigrams2(self, n=None):
        """
        Deprecated, experimental bigramming approach. Not used but stored for historical purposes.
        """
        if n == None:
            # default is bi/trigram has to appear on in half of docs on avg.
            n = len(self.paras_processed) / 20
        bigram_measures = BigramAssocMeasures()
        trigram_measures = TrigramAssocMeasures()
        docs = []
        for para in self.paras_processed:
            for sent in para:
                docs.append(sent)
        bigram_finder = BigramCollocationFinder.from_documents(docs)
        trigram_finder = TrigramCollocationFinder.from_documents(docs)
        bigram_finder.apply_freq_filter(n)
        trigram_finder.apply_freq_filter(n)
        trigrams = trigram_finder.score_ngrams(trigram_measures.pmi)
        bigrams = bigram_finder.score_ngrams(bigram_measures.pmi)

        breakpoint()

    def _substitute_ngrams(self, doc, ngrams):
        ngrammed_doc = []
        for sent in doc:
            ngrammed_sent = " ".join(sent)
            for ngram_patt in ngrams:
                ngrammed_sent = ngrammed_sent.replace(ngram_patt, f' {ngram_patt.strip().replace(" ", "_")} ')
            # edge cases are start and end of sent or sent is whole ngram
            for ngram_patt in ngrams:
                ngram = ngram_patt.strip()
                if ngrammed_sent.startswith(ngram):
                    ngrammed_sent = re.sub(r'^%s ' % ngram, r'%s ' % ngram.replace(" ", "_"), ngrammed_sent)
                if ngrammed_sent.endswith(ngram):
                    ngrammed_sent = re.sub(r' %s$' % ngram, r' %s' % ngram.replace(" ", "_"), ngrammed_sent)
                if ngrammed_sent == ngram:
                    ngrammed_sent = ngrammed_sent.replace(" ", "_")
                    break
            split_sent = ngrammed_sent.split(" ")
            ngrammed_doc.append(split_sent)
        return ngrammed_doc

    def _add_ngrams(self):
        """
        ['here', 'is', 'new', 'south', 'wales']
        match with ngram 'new south wales'
        """
        ngram_strings = [" %s " % x.strip('\n') for x in sorted([y for y in open(self.preproc_ngram_path, "r").readlines()], key=lambda x: len(x), reverse=True)]
        ngrammed_docs = []
        for i,doc in enumerate(self.paras_processed):
            if self.verbose:
                print(f"doc {i}")
                self._print_progress(i, len(self.paras_processed))
            ngrammed_docs.append(self._substitute_ngrams(doc, ngram_strings))
        self.paras_processed = ngrammed_docs
        if self.verbose:
            print("done ngramming!")

    def preprocess(
            self,
            ngrams=True
        ):
        """This function takes the spaCy documents found in this classes docs attribute and preprocesses them.
        The preprocessing pipeline tokenises each document and removes:
        1. punctuation
        2. spaces
        3. numbers
        4. urls
        5. stop words and single character words.

        It then lemmatises and lowercases each token and joins multi-word tokens together with an _.
        It then adds ngrams from a ngram list by joining matched ngrams in the corpus with an _.

        Args:
            ngrams (bool, optional): Whether to add ngrams or to keep the corpus as unigram. Defaults to True.
        """
        self.paras_processed = []
        for doc in self.docs:
            sents = []
            for s in doc.sents:
                words = []
                for w in s:
                    # PREPROCESS: lemmatize
                    # PREPROCESS: remove * puncuation
                    #                    * words that are / contain numbers
                    #                    * URLs
                    #                    * stopwords
                    #                    * words of length==1
                    if not w.is_punct \
                        and not w.is_space \
                        and not w.like_num \
                        and not any(i.isdigit() for i in w.lemma_) \
                        and not w.like_url \
                        and not w.text.lower() in STOP_WORDS \
                        and len(w.lemma_) > 1:
                        words.append(w.lemma_.lower().replace(" ", "_"))
                sents.append(words)
            self.paras_processed.append(sents)
        if ngrams:
            if self.verbose:
                print("adding ngrams...")
            self.matches = self._add_ngrams()
            # self._add_bigrams(ngrams=ngrams)
        new_paras = []
        if self.verbose:
            print("filtering terms...")
        for i,doc in enumerate(self.paras_processed):
            if self.verbose:
                self._print_progress(i, len(self.paras_processed))
            filtered_doc = []
            for sent in doc:
                filtered_doc.append([w for w in filter(lambda x: x not in self.term_blacklist and x != '', sent)])
            new_paras.append(filtered_doc)
        self.paras_processed = new_paras
        return self.paras_processed

class NgramPreprocessing(Preprocessing):
    """Class for generating preprocessing ngram list as seen in NGRAMS_PREPROC.txt

    Args:
        Preprocessing (Class): Parent class which holds the preprocessing logic.
    """
    
    ngram_unprocessed_path = os.path.join(os.environ['ROADMAP_SCRAPER'], "NGRAMS_LG.txt")

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.docs = self.nlp.pipe([x.strip('\n') for x in open(self.ngram_unprocessed_path).readlines()], n_process=11, batch_size=128)
    
    def preprocess(self):
        res = super().preprocess(ngrams=False)
        corpus = []
        for doc in res:
            joined_doc = []
            for sent in doc:
                joined_doc.extend(sent)
            corpus.append(joined_doc)
        with open(os.path.join(os.environ['ROADMAP_SCRAPER'], "NGRAMS_PREPROC.txt"), "w+") as fp:
            for phrase in corpus:
                fp.write(f"{' '.join(phrase)}\n")





if __name__ == "__main__":
    ngrampp = NgramPreprocessing()
    ngrampp.preprocess()