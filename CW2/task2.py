import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import random
from collections import defaultdict
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from task1 import data_pre_processing

def parse_labeled_data(file_path):
    query_tokens = {}
    passage_tokens = {}
    relevancy_matrix = defaultdict(dict)
    all_pairs = []
    relevant_map = defaultdict(dict)
    irrelevant_map = defaultdict(dict)

    rel_samples = []
    irrel_samples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)

        for qid, pid, query, passage, label in reader:
            label = float(label)

            if qid not in query_tokens:
                query_tokens[qid] = data_pre_processing(query)

            if pid not in passage_tokens:
                passage_tokens[pid] = data_pre_processing(passage)

            all_pairs.append((qid, pid))
            relevancy_matrix[qid][pid] = label

            if label > 0:
                relevant_map[qid][pid] = label
                rel_samples.append((qid, pid))
            else:
                irrelevant_map[qid][pid] = label
                irrel_samples.append((qid, pid))

    return (query_tokens, passage_tokens,
            relevant_map, irrelevant_map,
            rel_samples, irrel_samples,
            relevancy_matrix, all_pairs)


def parse_test_data(file_path):
    query_tokens = {}
    passage_tokens = {}
    qid_pid_pairs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)

        for qid, pid, query, passage in reader:
            if qid not in query_tokens:
                query_tokens[qid] = data_pre_processing(query)

            if pid not in passage_tokens:
                passage_tokens[pid] = data_pre_processing(passage)

            qid_pid_pairs.append((qid, pid))

    return query_tokens, passage_tokens, qid_pid_pairs


def downsample_irrelevant_pairs(irrelevant_pairs_dict, max_negatives=10, seed=None):
    if seed is not None:
        random.seed(seed)

    filtered_irrelevant_dict = defaultdict(dict)
    filtered_pair_list = []

    for qid in irrelevant_pairs_dict:
        pid_list = list(irrelevant_pairs_dict[qid].keys())
        selected_pids = (
            pid_list if len(pid_list) <= max_negatives
            else random.sample(pid_list, max_negatives)
        )

        for pid in selected_pids:
            filtered_irrelevant_dict[qid][pid] = irrelevant_pairs_dict[qid][pid]
            filtered_pair_list.append((qid, pid))

    return filtered_irrelevant_dict, filtered_pair_list



def save_text_for_word2vec(pairs, query_dict, passage_dict, query_path, passage_path):
    with open(query_path, 'w', encoding='utf-8') as q_out, open(passage_path, 'w', encoding='utf-8') as p_out:
        for index, (qid, pid) in enumerate(pairs):
        # for qid, pid in pairs:
            q_words = query_dict.get(qid, [])
            p_words = passage_dict.get(pid, [])
            q_out.write(' '.join(q_words) + '\n')
            p_out.write(' '.join(p_words) + '\n')


def train_word2vec_from_txt(file_path):
    sentences = LineSentence(file_path)
    model_word2vec = Word2Vec(sentences, sg=1, vector_size=100, window=5, min_count=1, negative=5, hs=0, workers=4)
    return model_word2vec


def get_w2v_features(data_list, qid_query_dict, pid_passage_dict, model_query, model_passage, rel_dict=None):
    query_vectors = defaultdict()
    passage_vectors = defaultdict()

    X = []
    y = [] if rel_dict else None

    for qid, pid in data_list:
        # Mean of query
        if qid in qid_query_dict:
            q_words = qid_query_dict[qid]
            q_embeds = [model_query.wv[w] for w in q_words if w in model_query.wv]
            if q_embeds:
                query_vectors[qid] = np.mean(q_embeds, axis=0)

        # Mean of passage
        if pid in pid_passage_dict:
            p_words = pid_passage_dict[pid]
            p_embeds = [model_passage.wv[w] for w in p_words if w in model_passage.wv]
            if p_embeds:
                passage_vectors[pid] = np.mean(p_embeds, axis=0)

        if qid in query_vectors and pid in passage_vectors:
            feature_vec = np.hstack((query_vectors[qid], passage_vectors[pid]))
            X.append(feature_vec)

            if rel_dict:
                label = rel_dict.get(qid, {}).get(pid, 0)
                y.append(label)

    X = np.array(X)
    y = np.array(y) if y is not None else None

    return query_vectors, passage_vectors, X, y

# Save to txt
def save_feature_file(data_list, X_data, y_data=None, filename='output.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, (qid_pid, x) in enumerate(zip(data_list, X_data)):
            qid, pid = qid_pid
            line = f"{qid}\t{pid}\t" + '\t'.join(map(str, x))
            if y_data is not None:
                line += '\t' + str(y_data[idx])
            f.write(line + '\n')
    print(f"Data written to {filename}")

# ================================================ Logistic Regression ===========================================
def load_feature_file(filename, has_label=True):
    data_load = []
    X = []
    y = [] if has_label else None

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = line.strip().split('\t')
            qid = data[0]
            pid = data[1]
            features = list(map(float, data[2:-1])) if has_label else list(map(float, data[2:]))

            data_load.append((qid, pid))
            X.append(features)

            if has_label:
                y.append(float(data[-1]))

    X = np.array(X)
    y = np.array(y) if has_label else None

    return data_load, X, y


def generate_prediction_scores(data_list, X_data, model, y_data=None):
    scores_dict = defaultdict(dict)
    label_dict = defaultdict(dict) if y_data is not None else None

    for i, (qid, pid) in enumerate(data_list):
        score = model.predict(X_data[i])
        scores_dict[qid][pid] = score

        if y_data is not None:
            label_dict[qid][pid] = y_data[i]

    return scores_dict, label_dict

class LogisticRegressionModel:
    def __init__(self, feature_dim, learning_rate=0.01, max_iter=100, tol=1e-4):
        self.weights = np.zeros(feature_dim)
        self.bias = 0
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias  # bias
        return self.sigmoid(z)

    def compute_loss(self, X, y):
        preds = self.predict(X)
        loss = -np.mean(y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9))
        return loss

    def compute_gradient(self, X, y):
        preds = self.predict(X)
        errors = preds - y
        dw = np.dot(X.T, errors) / len(y)
        db = np.sum(errors) / len(y)
        return dw, db

    def train(self, X, y):
        print(f"Training Logistic Regression (lr={self.lr}, max_iter={self.max_iter})")
        self.loss_history.append(self.compute_loss(X, y))

        for i in range(self.max_iter):
            dw, db = self.compute_gradient(X, y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            print(f"Epoch {i + 1}/{self.max_iter} - Loss: {loss:.4f}")

            if np.linalg.norm(dw) < self.tol:
                print("Converged.")
                break

        return self.loss_history

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)



def plot_learning_curves(X_train, y_train, learning_rates, max_iter=50):
    plt.figure(figsize=(8, 6))
    plt.title("Training Loss across Different Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    for lr in learning_rates:
        model = LogisticRegressionModel(
            feature_dim=X_train.shape[1],
            learning_rate=lr,
            max_iter=max_iter,
        )
        loss_history = model.train(X_train, y_train)
        plt.plot(range(1, len(loss_history) + 1), loss_history, label=f"lr={lr}")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure1.pdf", bbox_inches='tight', format='pdf')
    # plt.savefig("figure2.pdf", bbox_inches='tight', format='pdf')


def evaluate_ranking(scores_dict, label_dict, k=10):

    def average_precision(relevant_list):
        hits = 0
        sum_precisions = 0.0
        for i, rel in enumerate(relevant_list, start=1):
            if rel == 1:
                hits += 1
                sum_precisions += hits / i
        return sum_precisions / hits if hits > 0 else 0.0

    def dcg(relevant_list, k):
        return sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevant_list[:k]))

    def ndcg(relevant_list, k):
        ideal = sorted(relevant_list, reverse=True)
        ideal_dcg = dcg(ideal, k)
        actual_dcg = dcg(relevant_list, k)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    ap_list = []
    ndcg_list = []

    for qid in scores_dict:
        pid_scores = scores_dict[qid]
        pid_labels = label_dict.get(qid, {})

        ranked = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)

        relevant_list = [pid_labels.get(pid, 0) for pid, _ in ranked]

        ap_list.append(average_precision(relevant_list))
        ndcg_list.append(ndcg(relevant_list, k))

    mean_ap = np.mean(ap_list)
    mean_ndcg = np.mean(ndcg_list)

    return mean_ap, mean_ndcg

def create_file(score_map, query_file_path, output_path, method_name):
    lines = []
    with open(query_file_path, 'r', encoding='utf-8') as qf:
        reader = csv.reader(qf, delimiter='\t')

        for entry in reader:
            if len(entry) < 2:
                continue

            query_id = entry[0]

            if query_id in score_map:
                passage_scores = score_map[query_id]

                top_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:100]

                for rank_idx, (passage_id, score) in enumerate(top_passages, start=1):
                    line = f"{query_id} A2 {passage_id} {rank_idx} {score} {method_name}"
                    lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("\n".join(lines))


# ------------------- Main -------------------
if __name__ == "__main__":
    train_file = "train_data.tsv"
    val_file = "validation_data.tsv"
    test_file = "candidate_passages_top1000.tsv"

    output_file_train = "train_new.txt"
    output_file_val = "val_new.txt"
    output_file_test = "test_new.txt"
    output_task2 = "LR.txt"

    queries_train = "queries_train.txt"
    passage_train = "passage_train.txt"
    queries_val = "queries_val.txt"
    passage_val = "passage_val.txt"
    queries_test = "queries_test.txt"
    passage_test = "passage_test.txt"

    test_queries = "test-queries.tsv"


    # ============================================= Train set ===========================================================
    # Subsampling
    query_tokens, passage_tokens, relevant_map, irrelevant_map, rel_samples, irrel_samples, relevancy_matrix, all_pairs = parse_labeled_data(train_file)
    filtered_irrelevant_dict, filtered_pair_list = downsample_irrelevant_pairs(irrelevant_map, seed = 114514)

    train_samples_new = rel_samples + filtered_pair_list
    print(len(train_samples_new)) # 50548
    save_text_for_word2vec(train_samples_new, query_tokens, passage_tokens, queries_train, passage_train)

    train_queries_vector = train_word2vec_from_txt(queries_train)
    train_passage_vector = train_word2vec_from_txt(passage_train)

    qv_train, pv_train, X_train, y_train = get_w2v_features(train_samples_new, query_tokens, passage_tokens, train_queries_vector, train_passage_vector, rel_dict= relevancy_matrix)

    save_feature_file(train_samples_new, X_train, y_train, output_file_train)

    # ============================================= Validation set ===========================================================
    query_tokens_val, passage_tokens_val, relevant_map_val, irrelevant_map_val, rel_samples_val, irrel_samples_val, relevancy_matrix_val, all_pairs_val = parse_labeled_data(val_file)

    save_text_for_word2vec(all_pairs_val, query_tokens_val, passage_tokens_val, queries_val, passage_val)

    val_queries_vector = train_word2vec_from_txt(queries_val)
    val_passage_vector = train_word2vec_from_txt(passage_val)

    qv_val, pv_val, X_val, y_val = get_w2v_features(all_pairs_val, query_tokens_val, passage_tokens_val, val_queries_vector, val_passage_vector, rel_dict= relevancy_matrix_val)

    save_feature_file(all_pairs_val, X_val, y_val, output_file_val)

    # ============================================= Test set ===========================================================
    query_tokens_test, passage_tokens_test, qid_pid_pairs_test = parse_test_data(test_file)

    save_text_for_word2vec(qid_pid_pairs_test, query_tokens_test, passage_tokens_test, queries_test,passage_test)

    test_queries_vector = train_word2vec_from_txt(queries_test)
    test_passage_vector = train_word2vec_from_txt(passage_test)

    qv_test, pv_test, X_test, _ = get_w2v_features(qid_pid_pairs_test, query_tokens_test, passage_tokens_test, test_queries_vector, test_passage_vector)

    save_feature_file(qid_pid_pairs_test, X_test, filename = output_file_test)

    # ============================================= Logistic Regression ===========================================================
    train_data_load, X_train, y_train = load_feature_file(output_file_train, has_label=True)

    model = LogisticRegressionModel(feature_dim=X_train.shape[1], learning_rate=0.01, max_iter=500)
    losses = model.train(X_train, y_train)
    model.save_model("LRmodel.pkl")

    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    plot_learning_curves(X_train, y_train, learning_rates, max_iter=500)
    # plot_learning_curves(X_train, y_train, learning_rates, max_iter=5000)
    # plot_learning_curves(X_train, y_train, learning_rates, max_iter=50000)


    LR_model = LogisticRegressionModel.load_model("LRmodel.pkl")
    val_data_load, X_validation, y_validation = load_feature_file(output_file_val, has_label=True)
    scores_dict, label_dict = generate_prediction_scores(val_data_load, X_validation, LR_model, y_data= y_validation)

    mean_ap, mean_ndcg = evaluate_ranking(scores_dict, label_dict, k = None)
    print(f"Mean AP: {mean_ap:.4f}") # 0.0137
    print(f"Mean NDCG: {mean_ndcg:.4f}") # 0.1329

    test_data_load, X_test, _ = load_feature_file(output_file_test, has_label= False)

    scores_dict_test, _ = generate_prediction_scores(test_data_load, X_test, LR_model)
    create_file(scores_dict_test, test_queries, output_task2, method_name="LR")

























