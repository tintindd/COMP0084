import json
from collections import defaultdict
from collections import Counter
import numpy as np
import math
from task1 import data_pre_processing

# Loading...
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["inverted_index"], data["idf"]

# Calculate idf vector (queries)
def get_query_doc_vector(idf_dict, query_processed, pid, inverted_index, query_dict):
    query_vector = np.zeros(len(query_processed))
    doc_vector = np.zeros(len(query_processed))

    for i, term in enumerate(query_processed):
        query_vector[i] = query_dict.get(term, 0) * idf_dict.get(term, 0)
        doc_vector[i] = inverted_index.get(term, {}).get(pid, 0) * idf_dict.get(term, 0)

    return query_vector, doc_vector

# Calculate cosine similarity
def calculate_cosine_similarity(file_path, inverted_index, idf_dict):
    cosine_similarity_dict = defaultdict(dict)

    # idf_vector = get_idf_vector(idf_dict)
    # sq_idf_vector = np.linalg.norm(idf_vector)
    # tfdict = {word: i for i, word in enumerate(sorted(inverted_index.keys()))}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) >= 3:
                qid = parts[0]
                pid = parts[1]
                query = parts[2]
                passage = parts[3]
                # tf_vector = get_tf_vector(query, pid, inverted_index, tfdict)

                query_dict = Counter(data_pre_processing(query.lower()))
                query_and_doc = list(set(data_pre_processing(query) + data_pre_processing(passage.lower())))
                query_vector, doc_vector = get_query_doc_vector(idf_dict, query_and_doc, pid, inverted_index, query_dict)
                sq_query_vector = np.linalg.norm(query_vector)
                sq_doc_vector = np.linalg.norm(doc_vector)
                product_query_doc = np.dot(query_vector, doc_vector)

                if sq_query_vector != 0:
                    cosine_similarity = product_query_doc / (sq_query_vector * sq_doc_vector)
                else:
                    cosine_similarity = 0

                cosine_similarity_dict[qid][pid] = cosine_similarity
    return cosine_similarity_dict

# Get matched results
def rank_qid(file_path, qid_dict):
    matched_results = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid_q = parts[0]

                if qid_q in qid_dict:
                    all_pids = sorted(qid_dict[qid_q].items(), key=lambda x: x[1], reverse=True)
                    matched_results[qid_q] = all_pids[:100]
                else:
                    matched_results[qid_q] = []
    return matched_results

# Save results
def save_final_results(matched_results, output_file):
    results = []

    for qid, pid_scores in matched_results.items():
        for pid, score in pid_scores:
            results.append(f"{qid},{pid},{score}")  # 确保格式化 `score`

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

# Calculate dl to calculate BM25
def calculate_dl(file_path):
    dl = {}
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                qid = parts[0]
                pid = parts[1]
                passage = parts[3]
                dl[pid] = len(data_pre_processing(passage.lower()))
    return dl

# Calculate average dl for BM25
def calculate_avgdl(dl):
    N = len(dl)
    avgdl = sum(dl.values()) / N
    return N, avgdl

# Calculate bm25 idf
def calculate_bm25_idf(N, term, inverted_index):
    if term not in inverted_index:
        return 0
    df = len(inverted_index[term])
    bm25_idf = math.log((N - df + 0.5) / (df + 0.5))
    return bm25_idf

def calculate_bm25(file_path, inverted_index, N, dl, avgdl, k1, k2, b):
    bm25_scores = defaultdict(dict)

    with (open(file_path, "r", encoding="utf-8") as f):
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid = parts[0]
                pid = parts[1]
                query = parts[2]

                query_terms = data_pre_processing(query.lower())
                qf_dict = Counter(query_terms)
                bm25_score = 0
                K = k1 * ((1 - b) + b * (dl[pid] / avgdl))

                for term in query_terms:
                    idf = calculate_bm25_idf(N, term, inverted_index)
                    tf = inverted_index.get(term, {}).get(pid, 0)
                    tf_weight = ((k1 + 1) * tf) / (K + tf)

                    qf = qf_dict[term]
                    qf_weight = ((k2 + 1) * qf) / (k2 + qf)

                    bm25_score += idf * tf_weight * qf_weight
                bm25_scores[qid][pid] = bm25_score

    return bm25_scores


def task3_d5():

    cosine_similarity_dict = calculate_cosine_similarity(file_path, inverted_index, idf_dict)

    matched_results = rank_qid(queries_file, cosine_similarity_dict)

    save_final_results(matched_results, output_file_tfidf)


def task3_d6():
    dl = calculate_dl(file_path)

    N, avgdl = calculate_avgdl(dl)

    bm25 = calculate_bm25(file_path, inverted_index, N, dl, avgdl, k1=1.2, k2=100, b=0.75)

    matched_results_bm25 = rank_qid(queries_file, bm25)

    save_final_results(matched_results_bm25, output_file_bm25)


if __name__ == "__main__":
    inverted_index_file = "inverted_index.json"
    file_path = "candidate-passages-top1000.tsv"
    queries_file = "test-queries.tsv"
    output_file_tfidf = "tfidf.csv"
    output_file_bm25 = "bm25.csv"

    # Upload inverted_index and idf_dict
    inverted_index, idf_dict = load_json(inverted_index_file)

    task3_d5()
    task3_d6()


