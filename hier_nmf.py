# coding: utf-8
import numpy as np
from numpy.linalg import norm
from numpy.linalg import svd
from numpy.linalg import matrix_rank


# hier8_neat.m
def hier8_neat(X, k, **params):
    params.setdefault('trial_allowance', 3)
    params.setdefault('unbalanced', 0.1)
    params.setdefault('vec_norm', 2.0)
    params.setdefault('normW', True)
    params.setdefault('anls_alg', anls_entry_rank2_precompute)
    params.setdefault('tol', 1e-4)
    params.setdefault('maxiter', 500)

    m, n = X.shape
    clusters = [None] * (2 * (k - 1))
    Ws = [None] * (2 * (k - 1))
    W_buffer = [None] * (2 * (k - 1))
    H_buffer = [None] * (2 * (k - 1))
    priorities = np.zeros(2 * k - 1, dtype=np.float32)
    is_leaf = -np.ones(2 * (k - 1), dtype=np.float32)
    tree = np.zeros((2, 2 * (k - 1)), dtype=np.float32)
    splits = -np.ones(k - 1, dtype=np.float32)

    term_subset = np.where(np.sum(X, axis=1) != 0)[0]
    W = np.random.random((len(term_subset), 2))
    H = np.random.random((2, n))
    if len(term_subset) == m:
        W, H = nmfsh_comb_rank2(X, W, H, **params)
    else:
        W_tmp, H = nmfsh_comb_rank2(X[term_subset, :], W, H, **params)
        W = np.zeros((m, 2), dtype=np.float32)
        W[term_subset, :] = W_tmp
        del W_tmp

    result_used = 0
    for i in range(k - 1):
        if i == 0:
            split_node = 0
            new_nodes = [0, 1]
            min_priority = 1e40
            split_subset = np.arange(n)
        else:
            leaves = np.where(is_leaf == 1)[0]
            temp_priority = priorities[leaves]
            min_priority = np.min(temp_priority[temp_priority > 0])
            split_node = np.argmax(temp_priority)
            if temp_priority[split_node] < 0:
                print(f'Cannot generate all {k} leaf clusters')
                
                Ws = [W for W in Ws if W is not None]
                return tree, splits, is_leaf, clusters, Ws, priorities

            split_node = leaves[split_node]
            is_leaf[split_node] = 0
            W = W_buffer[split_node]
            H = H_buffer[split_node]
            split_subset = clusters[split_node]
            new_nodes = [result_used, result_used + 1]
            tree[:, split_node] = new_nodes

        result_used += 2
        cluster_subset = np.argmax(H, axis=0)
        clusters[new_nodes[0]] = split_subset[np.where(cluster_subset == 0)[0]]
        clusters[new_nodes[1]] = split_subset[np.where(cluster_subset == 1)[0]]
        Ws[new_nodes[0]] = W[:, 0]
        Ws[new_nodes[1]] = W[:, 1]
        splits[i] = split_node
        is_leaf[new_nodes] = 1

        subset = clusters[new_nodes[0]]
        subset, W_buffer_one, H_buffer_one, priority_one = trial_split(min_priority, X, subset, W[:, 0], **params)
        clusters[new_nodes[0]] = subset
        W_buffer[new_nodes[0]] = W_buffer_one
        H_buffer[new_nodes[0]] = H_buffer_one
        priorities[new_nodes[0]] = priority_one

        subset = clusters[new_nodes[1]]
        subset, W_buffer_one, H_buffer_one, priority_one = trial_split(min_priority, X, subset, W[:, 1], **params)
        clusters[new_nodes[1]] = subset
        W_buffer[new_nodes[1]] = W_buffer_one
        H_buffer[new_nodes[1]] = H_buffer_one
        priorities[new_nodes[1]] = priority_one

    return tree, splits, is_leaf, clusters, Ws, priorities


def trial_split(min_priority, X, subset, W_parent, **params):
    m = X.shape[0]

    trial = 0
    subset_backup = subset
    while trial < params['trial_allowance']:
        cluster_subset, W_buffer_one, H_buffer_one, priority_one = actual_split(X, subset, W_parent, **params)
        if priority_one < 0:
            break

        unique_cluster_subset = np.unique(cluster_subset)
        if len(unique_cluster_subset) != 2:
            print('Invalid number of unique sub-clusters!')

        length_cluster1 = len(np.where(cluster_subset == unique_cluster_subset[0])[0])
        length_cluster2 = len(np.where(cluster_subset == unique_cluster_subset[1])[0])
        if min(length_cluster1, length_cluster2) < params['unbalanced'] * len(cluster_subset):
            idx_small = np.argmin(np.array([length_cluster1, length_cluster2]))
            subset_small = np.where(cluster_subset == unique_cluster_subset[idx_small])[0]
            subset_small = subset[subset_small]
            _, _, _, priority_one_small = actual_split(X, subset_small, W_buffer_one[:, idx_small], **params)
            if priority_one_small < min_priority:
                trial += 1
                if trial < params['trial_allowance']:
                    print(f"Drop {len(subset_small)} documents...")
                    subset = np.setdiff1d(subset, subset_small)
            else:
                break
        else:
            break

    if trial == params['trial_allowance']:
        print(f"Recycle {len(subset_backup) - len(subset)} documents...")
        subset = subset_backup
        W_buffer_one = np.zeros((m, 2), dtype=np.float32)
        H_buffer_one = np.zeros((2, len(subset)), dtype=np.float32)
        priority_one = -2

    return subset, W_buffer_one, H_buffer_one, priority_one


def actual_split(X, subset, W_parent, **params):
    m = X.shape[0]
    if len(subset) <= 3:
        cluster_subset = np.ones(len(subset), dtype=np.float32)
        W_buffer_one = np.zeros((m, 2), dtype=np.float32)
        H_buffer_one = np.zeros((2, len(subset)), dtype=np.float32)
        priority_one = -1
    else:
        term_subset = np.where(np.sum(X[:, subset], axis=1) != 0)[0]
        X_subset = X[term_subset, :][:, subset]
        W = np.random.random((len(term_subset), 2))
        H = np.random.random((2, len(subset)))
        W, H = nmfsh_comb_rank2(X_subset, W, H, **params)
        cluster_subset = np.argmax(H, axis=0)
        W_buffer_one = np.zeros((m, 2), dtype=np.float32)
        W_buffer_one[term_subset, :] = W
        H_buffer_one = H
        if len(np.unique(cluster_subset)) > 1:
            priority_one = compute_priority(W_parent, W_buffer_one)
        else:
            priority_one = -1

    return cluster_subset, W_buffer_one, H_buffer_one, priority_one


def compute_priority(W_parent, W_child):
    n = len(W_parent)
    idx_parent = np.argsort(W_parent)[::-1]
    sorted_parent = W_parent[idx_parent]
    idx_child1 = np.argsort(W_child[:, 0])[::-1]
    idx_child2 = np.argsort(W_child[:, 1])[::-1]

    n_part = len(np.where(W_parent != 0)[0])
    if n_part <= 1:
        priority = -3
    else:
        weight = np.log(np.arange(n, 0, -1))
        first_zero = np.where(sorted_parent == 0)[0]
        if len(first_zero) > 0:
            weight[first_zero[0]:] = 1

        weight_part = np.zeros(n, dtype=np.float32)
        weight_part[: n_part] = np.log(np.arange(n_part, 0, -1))
        idx1 = np.argsort(idx_child1)
        idx2 = np.argsort(idx_child2)
        max_pos = np.maximum(idx1, idx2)
        discount = np.log(n - max_pos[idx_parent] + 1)
        discount[discount == 0] = np.log(2)
        weight /= discount
        weight_part /= discount
        
        ndcg1 = NDCG_part(idx_parent, idx_child1, weight, weight_part)
        ndcg2 = NDCG_part(idx_parent, idx_child2, weight, weight_part)
        priority = ndcg1 * ndcg2

    return priority


def NDCG_part(ground, test, weight, weight_part):
    seq_idx = np.argsort(ground)
    weight_part = weight_part[seq_idx]

    n = len(test)
    uncum_score = weight_part[test]
    uncum_score[2:] /= np.log2(np.arange(2, n))
    cum_score = np.sum(uncum_score)

    ideal_score = np.sort(weight)[::-1]
    ideal_score[2:] /= np.log2(np.arange(2, n))
    cum_ideal_score = np.sum(ideal_score)

    score = cum_score / cum_ideal_score
    return score


# nmfsh_comb_rank2.m
def nmfsh_comb_rank2(A, Winit, Hinit, **params):
    eps = 1e-6
    m, n = A.shape
    W, H = Winit, Hinit
    
    if W.shape[1] != 2:
        print("Error: Wrong size of W!")
    if H.shape[0] != 2:
        print("Error: Wrong size of H!")

    vec_norm = params.get('vec_norm', 2.0)
    normW = params.get('normW', True)
    tol = params.get('tol', 1e-4)
    maxiter = params.get('maxiter', 500)

    left = H.dot(H.T)
    right = A.dot(H.T)
    for iter_ in range(maxiter):
        if matrix_rank(left) < 2:
            print('The matrix H is singular')
            W = np.zeros((m, 2), dtype=np.float32)
            H = np.zeros((2, n), dtype=np.float32)
            U, S, V = svd(A, full_matrices=False)
            U, V = U[:, 0], V[0, :]
            if sum(U) < 0:
                U, V = -U, -V
            
            W[:, 0] = U
            H[0, :] = V
            return W, H

        W = anls_entry_rank2_precompute(left, right, W)
        norms_W = norm(W, axis=0)
        if np.min(norms_W) < eps:
            print('Error: Some column of W is essentially zero')

        W /= norms_W
        left = W.T.dot(W)
        right = A.T.dot(W)
        if matrix_rank(left) < 2:
            print('The matrix W is singular')

            W = np.zeros((m, 2), dtype=np.float32)
            H = np.zeros((2, n), dtype=np.float32)
            U, S, V = svd(A, full_matrices=False)
            U, V = U[:, 0], V[0, :]
            if sum(U) < 0:
                U, V = -U, -V
            
            W[:, 0] = U
            H[0, :] = V
            return W, H

        H = anls_entry_rank2_precompute(left, right, H.T).T
        gradH = left.dot(H) - right.T
        left = H.dot(H.T)
        right = A.dot(H.T)
        gradW = W.dot(left) - right

        if iter_ == 0:
            gradW_square = np.sum(np.power(gradW[np.logical_or(gradW <= 0, W > 0)], 2))
            gradH_square = np.sum(np.power(gradH[np.logical_or(gradH <= 0, H > 0)], 2))
            initgrad = np.sqrt(gradW_square + gradH_square)
            continue
        else:
            gradW_square = np.sum(np.power(gradW[np.logical_or(gradW <= 0, W > 0)], 2))
            gradH_square = np.sum(np.power(gradH[np.logical_or(gradH <= 0, H > 0)], 2))
            projnorm = np.sqrt(gradW_square + gradH_square)

        if projnorm < tol * initgrad:
            break

    # grad = projnorm / initgrad

    if vec_norm != 0:
        if normW:
            norms = np.power(np.sum(np.power(W, vec_norm), axis=0), 1/vec_norm)
            W /= norms
            H *= norms[:, None]
        else:
            norms = np.power(np.sum(np.power(H, vec_norm), axis=1), 1/vec_norm)
            W *= norms[None, :]
            H /= norms

    return W, H


# anls_entry_rank2_precompute.m
def anls_entry_rank2_precompute(left, right, H):
    eps = 1e-6
    n = right.shape[0]

    solve_either = np.zeros((n, 2), dtype=np.float32)
    solve_either[:, 0] = right[:, 0] / left[0, 0]
    solve_either[:, 1] = right[:, 0] / left[1, 1]
    cosine_either = solve_either * np.sqrt(np.array([left[0, 0], left[1, 1]]))
    choose_first = cosine_either[:, 0] >= cosine_either[:, 1]
    solve_either[choose_first, 1] = 0
    solve_either[np.logical_not(choose_first), 0] = 0

    if np.abs(left[0, 0]) < eps and abs(left[0, 1]) < eps:
        print('Error: The 2x2 matrix is close to singular or the input data matrix has tiny values')
    else:
        if np.abs(left[0, 0] >= np.abs(left[0, 1])):
            t = left[1, 0] / left[0, 0]
            a2 = left[0, 0] + t * left[1, 0]
            b2 = left[0, 1] + t * left[1, 1]
            d2 = left[1, 1] - t * left[0, 1]
            if np.abs(d2/a2) < eps:
                print('Error: The 2x2 matrix is close to singular')

            e2 = right[:, 0] + t * right[: ,1]
            f2 = right[:, 1] - t * right[:, 0]
        else:
            ct = left[0, 0] / left[1, 0]
            a2 = left[1, 0] + ct * left[0, 0]
            b2 = left[1, 1] + ct * left[0, 1]
            d2 = -left[0, 1] + ct * left[1, 1]
            if np.abs(d2/a2) < eps:
                print('Error: The 2x2 matrix is close to singular')
            
            e2 = right[:, 1] + ct * right[:, 0]
            f2 = -right[:, 0] + ct * right[:, 1]

        H[:, 1] = f2 / d2
        H[:, 0] = (e2 - b2 * H[:, 1]) / a2

    use_either = np.logical_not(np.all(H > 0, axis=1))
    H[use_either, :] = solve_either[use_either, :]

    return H


if __name__ == '__main__':
    pass
