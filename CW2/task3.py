from task2 import *
import xgboost as xgb
from itertools import product

def load_vectors(file_path, has_label=True):
    pair_list = []
    query_vecs = []
    passage_vecs = []
    labels = [] if has_label else None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            qid, pid = parts[0], parts[1]

            values = list(map(float, parts[2:]))
            if has_label:
                features, label = values[:-1], values[-1]
            else:
                features = values

            midpoint = len(features) // 2
            q_vec = features[:midpoint]
            p_vec = features[midpoint:]

            if len(q_vec) != len(p_vec):
                raise ValueError(f"Error: Uneven query/passage vector at QID={qid}, PID={pid}")

            pair_list.append((qid, pid))
            query_vecs.append(q_vec)
            passage_vecs.append(p_vec)
            if has_label:
                labels.append(label)

    return (
        pair_list,
        np.array(query_vecs),
        np.array(passage_vecs),
        np.array(labels) if has_label else None
    )


def prepare_lambdamart_data_cosine(pair_list, query_vecs, passage_vecs, labels=None):
    from collections import defaultdict

    def compute_cosine(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return np.dot(a, b) / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    features = []
    reordered_labels = [] if labels is not None else None
    group_sizes = [] if labels is not None else None
    reordered_pairs = []

    temp_data = defaultdict(list)

    for idx, (qid, pid) in enumerate(pair_list):
        q_vec = query_vecs[idx]
        p_vec = passage_vecs[idx]
        sim = compute_cosine(q_vec, p_vec)

        combined = np.concatenate([q_vec, p_vec, [sim]])

        if labels is not None:
            label = labels[idx]
            temp_data[qid].append((pid, combined, label))
        else:
            features.append(combined)
            reordered_pairs.append((qid, pid))

    if labels is not None:
        for qid, pid_entries in temp_data.items():
            group_sizes.append(len(pid_entries))
            pid_entries.sort(key=lambda x: x[2], reverse=True)
            for pid, feat, lbl in pid_entries:
                features.append(feat)
                reordered_labels.append(lbl)
                reordered_pairs.append((qid, pid))
        return group_sizes, np.array(features), np.array(reordered_labels), reordered_pairs
    else:
        return np.array(features)


def train_LGBmodel(hyperparams, X_train, y_train, q_group):
    results = []

    param_combos = product(
        hyperparams.get('lr', [0.1]),
        hyperparams.get('depth', [6]),
        hyperparams.get('subsample', [1.0]),
        hyperparams.get('colsample', [1.0]),
        hyperparams.get('trees', [100])
    )

    for lr, depth, subsample, colsample, trees in param_combos:
        ranker = xgb.XGBRanker(
            objective="rank:ndcg",
            booster="gbtree",
            learning_rate=lr,
            max_depth=depth,
            subsample=subsample,
            colsample_bytree=colsample,
            n_estimators=trees
        )

        print(f"Training: lr={lr}, depth={depth}, subsample={subsample}, colsample={colsample}, trees={trees}")
        ranker.fit(X_train, y_train, group=q_group)

        results.append(ranker)

    return results


def creat_prediction_dict(pair_list, X_features, model, true_labels=None):
    prediction_dict = defaultdict(dict)
    label_dict = defaultdict(dict) if true_labels is not None else None

    y_scores = model.predict(X_features)

    for i, (qid, pid) in enumerate(pair_list):
        prediction_dict[qid][pid] = y_scores[i]
        if true_labels is not None:
            label_dict[qid][pid] = true_labels[i]

    return prediction_dict, label_dict


def evaluate_and_select_best_model(model_list, X_val, y_val, val_pairs):
    best_model_idx = -1
    best_map = -1
    best_ndcg = -1

    for i, model in enumerate(model_list):
        pred_dict, label_dict = creat_prediction_dict(
            pair_list=val_pairs,
            X_features=X_val,
            model=model,
            true_labels=y_val
        )
        map_score, mean_ndcg = evaluate_ranking(pred_dict, label_dict, k = None)

        print(f"\nModel {i}: MAP = {map_score:.4f}, mNDCG = {mean_ndcg:.4f}")

        if map_score > best_map or (map_score == best_map and mean_ndcg > best_ndcg):
            best_model_idx = i
            best_map = map_score
            best_ndcg = mean_ndcg

    return best_model_idx, best_map, best_ndcg

if __name__ == "__main__":
    # Train
    train_new = "train_new.txt"
    train_pairs, q_vecs_train, p_vecs_train, labels_train = load_vectors(train_new, has_label=True)
    group_train, X_train, y_train, reordered_pairs_train = prepare_lambdamart_data_cosine(train_pairs, q_vecs_train, p_vecs_train, labels_train)

    hyperparams = {
        'lr': [0.01, 0.1],
        'depth': [5, 10],
        'subsample': [0.8, 1],
        'colsample': [0.8, 1],
        'trees': [100, 200]
    }
    training_model = train_LGBmodel(hyperparams, X_train, y_train, group_train)

    # Validation
    val_new = "val_new.txt"

    val_pairs, q_vecs_val, p_vecs_val, labels_val = load_vectors(val_new, has_label=True)
    group_val, X_val, y_val, reordered_pairs_val = prepare_lambdamart_data_cosine(val_pairs, q_vecs_val, p_vecs_val, labels_val)
    best_model_list,  best_map, best_ndcg = evaluate_and_select_best_model(training_model, X_val, y_val, reordered_pairs_val)

    best_model =  training_model[best_model_list]
    print(f"MAP: {best_map:.4f}, mNDCG: {best_ndcg:.4f}") # 0.0151 0.1348

    # Test
    test_new = "test_new.txt"
    test_pairs, q_vecs_val, p_vecs_val, _ = load_vectors(test_new, has_label= False)
    X_test = prepare_lambdamart_data_cosine(test_pairs, q_vecs_val, p_vecs_val, labels= None)

    test_score_dict, _ = creat_prediction_dict(test_pairs, X_test, best_model)

    test_queries = "test-queries.tsv"
    output_task3 = "LM.txt"

    create_file(test_score_dict, test_queries, output_task3, method_name="LM")
    print("Finished writing LM.txt")













