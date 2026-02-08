"""Forward Regression with Orthogonal Least Squares (FROLS) algorithm."""

from typing import Dict, List, Optional, Union
import numpy as np


def frols(
    regressors: np.ndarray,
    target: np.ndarray,
    selection_criteria: Union[float, int],
    colnames: Optional[List[str]] = None,
    epsilon: float = 1e-12,
) -> Dict:
    """
    Forward Orthogonal Least Squares for model term selection.
    
    Args:
        regressors: Candidate regressor matrix (N, M)
        target: Target vector (N,)
        selection_criteria: ERR threshold (float in [0,1]) or number of terms (int)
        colnames: Column names for regressors
        epsilon: Numerical stability threshold
        
    Returns:
        Dictionary with keys:
            - theta: Parameter estimates
            - selected_colnames: Names of selected terms
            - selected_indices: Indices of selected terms
            - err_values: ERR values for each selected term
            - P_selected: Selected regressor matrix
    """
    target = target.reshape(-1, 1) if target.ndim == 1 else target
    n_samples, n_candidates = regressors.shape

    empty_result = {
        "theta": np.array([]),
        "selected_colnames": [],
        "selected_indices": [],
        "err_values": np.array([]),
        "P_selected": np.empty((n_samples, 0)),
        "A": np.empty((0, 0)),
    }

    if n_samples == 0 or n_candidates == 0:
        return empty_result

    # Parse selection criteria
    is_err_mode = isinstance(selection_criteria, float) and 0 <= selection_criteria <= 1
    if isinstance(selection_criteria, int):
        max_terms = selection_criteria
    elif not is_err_mode:
        raise ValueError("selection_criteria: float in [0,1] or positive int")
    else:
        max_terms = n_candidates

    # Initialize
    sig_yy = max((target.T @ target).item(), epsilon)
    selected_indices: List[int] = []
    err_list: List[float] = []
    g_list: List[float] = []
    Q = np.empty((n_samples, 0))
    A = np.empty((0, 0))

    for step in range(n_candidates):
        errs = np.full(n_candidates, -np.inf)
        gs = np.zeros(n_candidates)
        qs = np.zeros((n_samples, n_candidates))

        for m in range(n_candidates):
            if m in selected_indices:
                continue

            p_m = regressors[:, m:m+1]
            
            if step == 0:
                q_m = p_m.copy()
            else:
                q_m = p_m.copy()
                for r in range(len(selected_indices)):
                    q_r = Q[:, r:r+1]
                    q_r_norm = (q_r.T @ q_r).item()
                    if q_r_norm >= epsilon:
                        alpha = (p_m.T @ q_r).item() / q_r_norm
                        q_m -= alpha * q_r

            qs[:, m] = q_m.flatten()
            q_norm = (q_m.T @ q_m).item()
            
            if q_norm >= epsilon:
                gs[m] = (target.T @ q_m).item() / q_norm
                errs[m] = (gs[m] ** 2 * q_norm) / sig_yy

        if np.all(np.isneginf(errs)):
            break

        # Select best term
        best_idx = int(np.argmax(errs))
        selected_indices.append(best_idx)
        err_list.append(errs[best_idx])
        g_list.append(gs[best_idx])

        # Update Q
        q_new = qs[:, best_idx:best_idx+1]
        Q = np.hstack((Q, q_new)) if Q.shape[1] > 0 else q_new

        # Update A matrix
        p_orig = regressors[:, best_idx:best_idx+1]
        if len(selected_indices) == 1:
            A = np.array([[1.0]])
        else:
            a_col = np.zeros((len(selected_indices) - 1, 1))
            for r in range(len(selected_indices) - 1):
                q_r = Q[:, r:r+1]
                q_r_norm = (q_r.T @ q_r).item()
                if q_r_norm >= epsilon:
                    a_col[r, 0] = (p_orig.T @ q_r).item() / q_r_norm
            A = np.block([
                [A, a_col],
                [np.zeros((1, len(selected_indices) - 1)), np.array([[1.0]])]
            ])

        # Check stopping criteria
        if is_err_mode:
            if (1.0 - sum(err_list)) <= selection_criteria:
                break
        else:
            if len(selected_indices) >= max_terms:
                break

    if len(selected_indices) == 0:
        return empty_result

    # Solve for parameters
    g = np.array(g_list).reshape(-1, 1)
    theta = np.linalg.solve(A, g).flatten()

    return {
        "theta": theta,
        "selected_colnames": [colnames[i] for i in selected_indices] if colnames else [],
        "selected_indices": selected_indices,
        "err_values": np.array(err_list),
        "P_selected": regressors[:, selected_indices],
        "A": A,
    }
