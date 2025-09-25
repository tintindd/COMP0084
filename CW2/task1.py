from collections import defaultdict, Counter
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import math
import csv
# from nltk.stem import PorterStemmer
# from tqdm import tqdm

nltk.download('stopwords')

# Data pre-processing
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        return text

def data_pre_processing(text):

    tokens = re.findall(r'\b\w+\b', text.lower())

    stop_words = set(stopwords.words('english'))
    words_remove = [word for word in tokens if word not in stop_words]

    return words_remove


def load_tsv_data(file_path):
    unique_passages = {}
    query_passage_rel = defaultdict(dict)
    queries = {}

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            qid, pid, query, passage, relevancy = row[0], row[1], row[2], row[3], float(row[4])

            if qid not in queries:
                queries[qid] = data_pre_processing(query)

            if pid not in unique_passages:
                unique_passages[pid] = data_pre_processing(passage)

            query_passage_rel[qid][pid] = relevancy

    return unique_passages, queries, query_passage_rel


# Build inverted index
def build_inverted_index(passages):
    inverted_index = defaultdict(dict)

    for pid, terms in passages.items():
        term_freq = Counter(terms)
        for term, freq in term_freq.items():
            inverted_index[term][pid] = freq

    return inverted_index


# Calculate dl
def calculate_dl(passages):
    return {pid: len(terms) for pid, terms in passages.items()}


# Calculate avgdl
def calculate_avgdl(dl):
    return sum(dl.values()) / len(dl)


# Calculate BM25 idf
def calculate_bm25_idf(N, term, inverted_index):
    if term not in inverted_index:
        return 0
    df = len(inverted_index[term])
    bm25_idf = math.log((N - df + 0.5) / (df + 0.5))
    return bm25_idf


# Calculate BM25 scores
def calculate_bm25(query_terms, passage_rels, inverted_index, N, dl, avgdl, k1=1.2, k2=100, b=0.75):
    bm25_scores = {}

    for pid in passage_rels.keys():
        bm25_score = 0
        K = k1 * ((1 - b) + b * (dl[pid] / avgdl))
        qf_dict = Counter(query_terms)

        for term in query_terms:
            idf = calculate_bm25_idf(N, term, inverted_index)
            tf = inverted_index.get(term, {}).get(pid, 0)
            tf_weight = ((k1 + 1) * tf) / (K + tf)

            qf = qf_dict[term]
            qf_weight = ((k2 + 1) * qf) / (k2 + qf)

            bm25_score += idf * tf_weight * qf_weight

        bm25_scores[pid] = bm25_score

    return bm25_scores


# Calculate AP
def compute_ap(bm25_scores, query_passage_rel):
    total_ap = 0

    for qid, pid_scores in bm25_scores.items():
        sorted_pids = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)

        retrieved_count, relevant_count, precision_sum = 0, 0, 0
        for pid, _ in sorted_pids:
            retrieved_count += 1
            rel = query_passage_rel[qid].get(pid, 0) == 1
            # if rel >0
            if rel == 1:
                relevant_count += 1
                precision_sum += relevant_count / retrieved_count

        if relevant_count > 0:
            total_ap += precision_sum / relevant_count

    return total_ap / len(bm25_scores)


# Calculate NDCG
def compute_ndcg(bm25_scores, query_passage_rel):
    total_ndcg = 0

    for qid, pid_scores in bm25_scores.items():
        sorted_pids = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)

        dcg, idcg = 0, 0
        relevances = sorted(query_passage_rel[qid].values(), reverse=True)  # 最优排序的 relevance

        for i, (pid, _) in enumerate(sorted_pids):
            rel = query_passage_rel[qid].get(pid, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 2)

        for i, rel in enumerate(relevances):
            idcg += (2 ** rel - 1) / np.log2(i + 2)

        total_ndcg += (dcg / idcg) if idcg != 0 else 0
        # total_ndcg += (dcg / idcg) if idcg > 0 else 0

    return total_ndcg / len(bm25_scores)


# Main
if __name__ == "__main__":
    file_path = "validation_data.tsv"
    unique_passages, queries, query_passage_rel = load_tsv_data(file_path)

    inverted_index = build_inverted_index(unique_passages)

    dl = calculate_dl(unique_passages)
    avgdl = calculate_avgdl(dl)
    N = len(unique_passages)
    # print(N)
    # print(len(inverted_index))
    bm25_scores = {qid: calculate_bm25(queries[qid], query_passage_rel[qid], inverted_index, N, dl, avgdl) for qid
                   in queries}

    bm25_ap = compute_ap(bm25_scores, query_passage_rel)
    bm25_ndcg = compute_ndcg(bm25_scores, query_passage_rel)

    print("BM25_mAP:", bm25_ap)
    print("BM25_mNDCG:", bm25_ndcg)