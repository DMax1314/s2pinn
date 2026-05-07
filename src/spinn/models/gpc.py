import numpy as np
import scipy.special
from typing import Tuple, List, Literal, Optional

def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """Probabilists' Hermite polynomial H_n(x)"""
    return scipy.special.hermite(n)(x)

def legendre_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """Legendre polynomial P_n(x)"""
    return scipy.special.legendre(n)(x)

def eval_gpc_basis(basis: Literal["hermite", "legendre"], degree: int, x: np.ndarray) -> np.ndarray:
    """Evaluate gPC basis functions up to degree at points x (shape: [num_points])"""
    if basis == "hermite":
        return np.stack([hermite_polynomial(n, x) for n in range(degree + 1)], axis=-1)
    elif basis == "legendre":
        return np.stack([legendre_polynomial(n, x) for n in range(degree + 1)], axis=-1)
    else:
        raise ValueError(f"Unknown basis: {basis}")

def gauss_hermite_quadrature(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite quadrature nodes and weights for degree"""
    nodes, weights = np.polynomial.hermite.hermgauss(degree)
    weights /= np.sqrt(np.pi)  # Normalize for probabilists' Hermite
    return nodes, weights

def gauss_legendre_quadrature(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature nodes and weights for degree"""
    nodes, weights = np.polynomial.legendre.leggauss(degree)
    return nodes, weights

def tensor_quadrature(R: int, p: int, basis: Literal["hermite", "legendre"]) -> Tuple[np.ndarray, np.ndarray]:
    """Full tensor product quadrature for R random dims, order p"""
    if basis == "hermite":
        nodes_1d, weights_1d = gauss_hermite_quadrature(p)
    else:
        nodes_1d, weights_1d = gauss_legendre_quadrature(p)
    grids = np.meshgrid(*([nodes_1d] * R), indexing="ij")
    nodes = np.stack([g.flatten() for g in grids], axis=-1)
    weights = np.prod(np.meshgrid(*([weights_1d] * R), indexing="ij"), axis=0).flatten()
    return nodes, weights

def smolyak_indices(R: int, p: int) -> List[List[int]]:
    """Generate Smolyak multi-index set for R dims, order p (level p)"""
    from itertools import product
    indices = []
    for idx in product(range(1, p + 1), repeat=R):
        if sum(idx) <= R + p - 1:
            indices.append(list(idx))
    return indices

def smolyak_quadrature(R: int, p: int, basis: Literal["hermite", "legendre"]) -> Tuple[np.ndarray, np.ndarray]:
    """Smolyak sparse grid quadrature for R random dims, order p"""
    indices = smolyak_indices(R, p)
    node_list = []
    weight_list = []
    for idx in indices:
        nodes_1d = []
        weights_1d = []
        for l in idx:
            if basis == "hermite":
                n, w = gauss_hermite_quadrature(l)
            else:
                n, w = gauss_legendre_quadrature(l)
            nodes_1d.append(n)
            weights_1d.append(w)
        grids = np.meshgrid(*nodes_1d, indexing="ij")
        nodes = np.stack([g.flatten() for g in grids], axis=-1)
        weights = np.prod(np.meshgrid(*weights_1d, indexing="ij"), axis=0).flatten()
        node_list.append(nodes)
        weight_list.append(weights)
    # Concatenate and average duplicates
    all_nodes = np.concatenate(node_list, axis=0)
    all_weights = np.concatenate(weight_list, axis=0)
    # Remove duplicates (within tolerance)
    uniq_nodes, idx = np.unique(np.round(all_nodes, 8), axis=0, return_index=True)
    uniq_weights = np.zeros(len(uniq_nodes))
    for i, node in enumerate(uniq_nodes):
        mask = np.all(np.isclose(all_nodes, node, atol=1e-8), axis=1)
        uniq_weights[i] = np.sum(all_weights[mask])
    return uniq_nodes, uniq_weights

def get_quadrature(R: int, p: int, basis: Literal["hermite", "legendre"], quadrature: Literal["tensor", "smolyak"]) -> Tuple[np.ndarray, np.ndarray]:
    """Unified quadrature interface"""
    if quadrature == "tensor":
        return tensor_quadrature(R, p, basis)
    elif quadrature == "smolyak":
        return smolyak_quadrature(R, p, basis)
    else:
        raise ValueError(f"Unknown quadrature: {quadrature}")

def get_basis_type(dist: Literal["gaussian", "uniform"]) -> str:
    """Map distribution to basis type"""
    if dist == "gaussian":
        return "hermite"
    elif dist == "uniform":
        return "legendre"
    else:
        raise ValueError(f"Unknown distribution: {dist}")

def eval_gpc_multi(R: int, p: int, basis: Literal["hermite", "legendre"], z: np.ndarray) -> np.ndarray:
    """
    Evaluate multi-dimensional gPC basis at points z (shape: [num_points, R])
    Returns shape: [num_points, num_basis]
    """
    from itertools import product
    num_points = z.shape[0]
    # Multi-index for total degree <= p
    multi_idx = [idx for idx in product(range(p + 1), repeat=R) if sum(idx) <= p]
    num_basis = len(multi_idx)
    out = np.zeros((num_points, num_basis))
    for i, idx in enumerate(multi_idx):
        val = np.ones(num_points)
        for r in range(R):
            if basis == "hermite":
                val *= hermite_polynomial(idx[r], z[:, r])
            else:
                val *= legendre_polynomial(idx[r], z[:, r])
        out[:, i] = val
    return out

def get_multi_indices(R: int, p: int) -> List[List[int]]:
    """Return multi-indices for total degree <= p"""
    from itertools import product
    return [list(idx) for idx in product(range(p + 1), repeat=R) if sum(idx) <= p]
