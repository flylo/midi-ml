import os
from collections import defaultdict
from itertools import product
from typing import List
import mido
import numpy as np
from scipy.sparse import dok_matrix, vstack


def window_gen(sequence, n):
    """
    Generator that iterates over a window of size n in a given sequence
    :param sequence: the sequence we wish to iterate over
    :param n: the window size of the iteration
    :return:
    """
    low = 0
    high = n
    for element in sequence:
        window = sequence[low:high]
        if len(window) < n:
            break
        low += 1
        high += 1
        yield tuple(window)


class MidiFeatureCorpus(object):
    """
    Class to search through a directory for midi files and add them to a sparse_matrix of note tuples
    """

    def __init__(self, path: str, note_window_size: int = 2):
        self.path = path
        self.note_window_size_ = note_window_size
        self.files_ = self._depth_first_midi_search(self.path)
        self.note_sequence_set = self.initialize_note_sequence_set(note_window_size)
        self.sparse_matrix = dok_matrix((len(self.files_),
                                         len(self.note_sequence_set)),
                                        dtype=np.float32)

    def _depth_first_midi_search(self, path: str) -> List[str]:
        """
        Perform a recursive depth-first search through the files
        :param path:
        :return:
        """
        files_out = []
        paths = os.listdir(path)
        for p in paths:
            full_subpath = path + "/" + p
            try:
                os.listdir(full_subpath)
                dfs_results = self._depth_first_midi_search(full_subpath)
                for file in dfs_results:
                    files_out.append(file)
            except NotADirectoryError:
                if full_subpath.endswith(".mid"):
                    files_out.append(full_subpath)
        return files_out

    @staticmethod
    def initialize_note_sequence_set(window_size: int):
        notes = [str(i) for i in range(128)]
        notes_copies = [notes for i in range(window_size)]
        note_sequences = []
        for combo in product(*notes_copies):
            note_sequences.append(combo[0] + "|" + combo[1])
        return note_sequences

    @staticmethod
    def get_n_note_sequence(midi: mido.MidiFile,
                            note_window_size: int = 2):
        notes = [str(m.note) for m in midi if m.type == "note_on"]
        n_note_sequences = []
        for note_seq in window_gen(notes, note_window_size):
            n_note_sequences.append("|".join([note for note in note_seq]))
        return n_note_sequences

    @staticmethod
    def sequence_encoder(seq: List[str]):
        d = defaultdict(float)
        for entry in seq:
            d[entry] += 1.
        return d

    def _parse_file_as_sequence(self, file_name):
        parsed_file = mido.MidiFile(file_name)
        return self.get_n_note_sequence(parsed_file, self.note_window_size_)

    def parse_corpus(self):
        for i, file in enumerate(self.files_):
            try:
                sequence = self._parse_file_as_sequence(file)
                encoded_sequence = self.sequence_encoder(sequence)
                for (seq, count) in encoded_sequence.items():
                    j = self.note_sequence_set.index(seq)
                    self.sparse_matrix[i, j] = count
            # we accept generic errors here to avoid any of the possible corruptions in our midi files
            except:
                continue


class LabeledCorpusSet(object):
    """

    """

    def __init__(self, path: str, note_window_size: int = 2):
        self.path_ = path
        self.note_window_size_ = note_window_size
        self.corpus_name_list_ = os.listdir(self.path_)
        self.corpus_labels = []
        self.corpus_list_ = []
        matrix_shape = (0, len(MidiFeatureCorpus.initialize_note_sequence_set(note_window_size)))
        self.sparse_matrix = dok_matrix(matrix_shape)
        self.parsed_ = False

    def parse_corpus_set(self):
        """
        Iterates through the files in the corpus. Will ignore directory structure within
        a corpus (e.g. if cantatas and sonatas are in different files)
        """
        matrix_set = []
        for corpus_name in self.corpus_name_list_:
            file_path = self.path_ + corpus_name
            print("reading from {}".format(file_path))
            corpus = MidiFeatureCorpus(file_path, self.note_window_size_)
            corpus.parse_corpus()
            self.corpus_list_.append(corpus)
            for label in range(corpus.sparse_matrix.shape[0]):
                self.corpus_labels.append(corpus_name)
            matrix_set.append(corpus.sparse_matrix)
        self.sparse_matrix = vstack(matrix_set)
        self.parsed_ = True


def main(input: str, output: str, bucket: str) -> LabeledCorpusSet:
    # download_from_gcs(bucket_name=bucket,
    #                   prefix=input,
    #                   local_fs_loc=os.environ["LOCAL_DATA_LOC"])
    labeled_corpus = LabeledCorpusSet(os.environ["LOCAL_DATA_LOC"])
    return labeled_corpus.parse_corpus_set()
