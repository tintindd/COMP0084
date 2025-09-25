import torch
from task3 import *

from collections import defaultdict
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import torch.nn as nn
from tqdm import tqdm


class DSSM(nn.Module):
    def __init__(self, vec_dim, hidden_dim=128):
        super(DSSM, self).__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(vec_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        self.passage_proj = nn.Sequential(
            nn.Linear(vec_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        self.output_layer = nn.Linear(2, 2)

    def forward(self, x):
        x = x.squeeze(1)  # [B, D]
        total_dim = x.shape[1]
        vec_dim = (total_dim - 1) // 2  # exclude cosine
        q_vec = x[:, :vec_dim]
        p_vec = x[:, vec_dim:2 * vec_dim]
        cosine = x[:, -1].unsqueeze(1)

        q_proj = self.query_proj(q_vec)
        p_proj = self.passage_proj(p_vec)
        sim = F.cosine_similarity(q_proj, p_proj).unsqueeze(1)

        concat = torch.cat([sim, cosine], dim=1)

        return self.output_layer(concat)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def train_dssm_model(X_train, y_train, vec_dim, epochs=20, lr=0.001, batch_size=64):
    model = DSSM(vec_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    for epoch in tqdm(range(epochs), desc="Training DSSM", ncols=80):
        model.train()
        total_loss = 0
        for i in range(0, len(X_tensor), batch_size):
            x_batch = X_tensor[i:i+batch_size].unsqueeze(1)
            y_batch = y_tensor[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model


def generate_nn_score_dict(pairs, inputs, net_model, ground_truth=None, batch=64):
    predict_dict = defaultdict(dict)
    truth_dict = defaultdict(dict) if ground_truth is not None else None

    net_model.eval()
    total = len(pairs)
    idx = 0

    while idx < total:
        batch_end = min(idx + batch, total)
        qid_pid_batch = pairs[idx:batch_end]
        input_batch = inputs[idx:batch_end]

        with torch.no_grad():
            out = net_model(input_batch.unsqueeze(1))
            probs = F.softmax(out, dim=-1)[:, 1]

        for k in range(len(qid_pid_batch)):
            qid, pid = qid_pid_batch[k]
            predict_dict[qid][pid] = probs[k].item()

            if ground_truth is not None:
                truth_dict[qid][pid] = ground_truth[idx + k].item()

        idx = batch_end

    return predict_dict, truth_dict

if __name__ == "__main__":
    # Training
    train_new = "train_new.txt"

    train_pairs, q_vecs_train, p_vecs_train, labels_train = load_vectors(train_new, has_label=True)
    group_train, X_train, y_train, reordered_pairs_train = prepare_lambdamart_data_cosine(train_pairs, q_vecs_train, p_vecs_train, labels_train)

    vec_dim = (X_train.shape[1] - 1) // 2

    dssm_model = train_dssm_model(X_train, y_train, vec_dim=vec_dim, epochs=50, lr=0.001)

    dssm_model.save_model("DSSMmodel.pkl")
    print("Save to DSSMmodel.pkl")

    # Validation
    val_new = "val_new.txt"
    val_pairs, q_vecs_val, p_vecs_val, y_val = load_vectors(val_new, has_label=True)
    group_val, X_val, y_val, reordered_val = prepare_lambdamart_data_cosine(val_pairs, q_vecs_val, p_vecs_val, y_val)

    dssm_model = DSSM.load_model("DSSMmodel.pkl").to(device)
    print("Loading success")

    val_X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    pred_dict, label_dict = generate_nn_score_dict(reordered_val, val_X_tensor, dssm_model, ground_truth=y_val)


    map_score, ndcg_score = evaluate_ranking(pred_dict, label_dict, k = None)
    print(f"DSSM Validation - MAP: {map_score:.4f}, mNDCG: {ndcg_score:.4f}") # 0.0108 0.1297

    # Test
    test_file = "test_new.txt"
    test_pairs, qv_test, pv_test, _ = load_vectors(test_file, has_label=False)
    X_test = prepare_lambdamart_data_cosine(test_pairs, qv_test, pv_test)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_pred_dict, _ = generate_nn_score_dict(test_pairs, X_test_tensor, dssm_model)

    create_file(test_pred_dict, "test-queries.tsv", "NN.txt", method_name="NN")


