from collections import defaultdict
import pandas as pd
import numpy as np
from task1 import data_pre_processing
import task3

def calculate_vocab_size(inverted_index):
    vocab_size = len(inverted_index)
    return vocab_size

def calculate_vocab_size_total(inverted_index):
    total_num = 0
    cqi_dict = {}
    for term in inverted_index:
        cqi_dict[term] = sum(inverted_index.get(term, {}).values())
        total_num += sum(inverted_index.get(term, {}).values())
    return total_num, cqi_dict

# def calculate_avg_doc_length_and_unique_terms(file_path):
#     doc_lengths = []
#     unique_term_counts = []
#
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split("\t")
#             if len(parts) >= 4:
#                 passage = parts[3].lower()
#                 terms = data_pre_processing(passage)
#
#                 doc_lengths.append(len(terms))
#                 unique_term_counts.append(len(set(terms)))
#
#     avg_doc_length = np.mean(doc_lengths) if doc_lengths else 0
#     avg_unique_terms = np.mean(unique_term_counts) if unique_term_counts else 0
#
#     return avg_doc_length, avg_unique_terms


def calculate_laplace_smooth(file_path, inverted_index, vocab_size, dl):
    laplace_scores = defaultdict(dict)

    with (open(file_path, "r", encoding="utf-8") as f):
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid = parts[0]
                pid = parts[1]
                query = parts[2]

                query_terms = data_pre_processing(query.lower())
                laplace_score = 0

                for term in query_terms:
                    tf = inverted_index.get(term, {}).get(pid, 0)
                    doc_length = dl[pid]
                    prob = (tf + 1) / (doc_length + vocab_size)
                    laplace_score += np.log(prob)

                laplace_scores[qid][pid] = laplace_score

    return laplace_scores

def calculate_lidstone_smooth(file_path, inverted_index, vocab_size, dl, epsilon):
    lidstone_scores = defaultdict(dict)

    with (open(file_path, "r", encoding="utf-8") as f):
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid = parts[0]
                pid = parts[1]
                query = parts[2]

                query_terms = data_pre_processing(query.lower())
                lidstone_score = 0

                for term in query_terms:
                    tf = inverted_index.get(term, {}).get(pid, 0)
                    doc_length = dl[pid]
                    prob = (tf + epsilon) / (doc_length + epsilon * vocab_size)
                    lidstone_score += np.log(prob)

                lidstone_scores[qid][pid] = lidstone_score

    return lidstone_scores

def calculate_dirichlet_smooth(file_path, inverted_index, dl, micro):
    dirichlet_scores = defaultdict(dict)
    total_num, cqi_dict = calculate_vocab_size_total(inverted_index)

    with (open(file_path, "r", encoding="utf-8") as f):
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid = parts[0]
                pid = parts[1]
                query = parts[2]

                query_terms = data_pre_processing(query.lower())
                dirichlet_score = 0

                for term in query_terms:
                    tf = inverted_index.get(term, {}).get(pid, 0)
                    doc_length = dl[pid]
                    cqi = cqi_dict.get(term, 0)
                    prob = (doc_length / (doc_length + micro)) * (tf / doc_length) + (micro / (micro + doc_length)) * (cqi / total_num)
                    if prob != 0:
                        dirichlet_score += np.log(prob)
                    else:
                        dirichlet_score = 0

                dirichlet_scores[qid][pid] = dirichlet_score

    return dirichlet_scores

def task4_laplace():

    laplace = calculate_laplace_smooth(file_path, inverted_index, vocab_size, dl)

    matched_results_laplace = task3.rank_qid(queries_file, laplace)

    task3.save_final_results(matched_results_laplace, output_file_laplace)


def task4_lidstone():
    epsilon = 0.1

    lidstone = calculate_lidstone_smooth(file_path, inverted_index, vocab_size, dl, epsilon)

    matched_results_lidstone = task3.rank_qid(queries_file, lidstone)

    task3.save_final_results(matched_results_lidstone, output_file_lidstone)

def task4_dirichlet():
    micro = 50

    dirichlet = calculate_dirichlet_smooth(file_path, inverted_index, dl, micro)

    matched_results_dirichlet = task3.rank_qid(queries_file, dirichlet)

    task3.save_final_results(matched_results_dirichlet, output_file_dirichlet)

if __name__ == "__main__":
    inverted_index_file = "inverted_index.json"
    file_path = "candidate-passages-top1000.tsv"
    queries_file = "test-queries.tsv"
    output_file_laplace = ("laplace.csv")
    output_file_lidstone = ("lidstone.csv")
    output_file_dirichlet = ("dirichlet.csv")

    inverted_index, idf_dict = task3.load_json(inverted_index_file)

    dl = task3.calculate_dl(file_path)

    vocab_size = calculate_vocab_size(inverted_index)



    # avg_doc_length, avg_unique_terms = calculate_avg_doc_length_and_unique_terms(file_path)
    # print(avg_doc_length)
    # print(avg_unique_terms)


    task4_laplace()
    task4_lidstone()
    task4_dirichlet()
