import collections
import math
import json
from task1 import data_pre_processing


# Upload task1 vocabulary
def load_vocabulary(vocab_file):
    vocabulary = set()
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            vocabulary.add(line.strip())
    return vocabulary

# Calculate number of documents in collection
def count_unique_documents(file_path):
    document_ids = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                document_ids.add(parts[1])
                N = len(document_ids)
    return N

# Build inverted index
def build_inverted_index(file_path, vocabulary):
    inverted_index = collections.defaultdict(dict)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            pid, passage = parts[1], parts[3].lower()

            words = data_pre_processing(passage)
            words = [word for word in words if word in vocabulary]
            word_freq = collections.Counter(words)

            for word, tf in word_freq.items():
                inverted_index[word][pid] = tf

    return inverted_index

# Compute IDF values
def compute_idf(inverted_index, N):
    idf_dict = {}  # {word: idf}
    for word, doc_dict in inverted_index.items():
        n_t = len(doc_dict)
        if n_t == 0:
            print("Log（0）is unacceptable")
        else:
            idf_dict[word] = math.log10(N / n_t)
    return idf_dict

# Save inverted index and IDF to a JSON file
def save_as_json(inverted_index, idf_dict, output_file="inverted_index.json"):
    data = {
        "inverted_index": inverted_index,
        "idf": idf_dict
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)  # save as json


def task2_function():
    vocabulary_file = "task1_vocabulary.txt"
    file_path = "candidate-passages-top1000.tsv"
    output_file = "inverted_index.json"

    # Upload task1 vocabulary
    vocabulary = load_vocabulary(vocabulary_file)

    # Build inverted index and compute idf
    N = count_unique_documents(file_path)
    inverted_index = build_inverted_index(file_path, vocabulary)
    idf_dict = compute_idf(inverted_index, N)

    # Save results to file as json
    save_as_json(inverted_index, idf_dict, output_file)


if __name__ == "__main__":
    task2_function()

