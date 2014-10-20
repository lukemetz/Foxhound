import numpy as np

def gs_pca(array, components, iterations=100, error=1e-6, results='pca'):
    """
    Calculates gs_pca as described in http://arxiv.org/pdf/0811.1081v1.pdf

    Can return results for either SVD or PCA depending on desire
    """
    array = array - np.mean(array, axis=0)  # Mean centers columns
    left_eigs = np.zeros(shape=(array.shape[0], components))
    right_eigs = np.zeros(shape=(array.shape[0], components))
    eigenvalues = np.zeros(components)
    residual = array
    for i in range(components):
        mu = 0
        left_eigs[:, i] = residual[:, i]
        for j in range(iterations):
            right_eigs[:, i] = np.dot(Residual.T, left_eigs[:, i])
            if i > 0:
                A = np.dot(right_eigs[:, :i].T, right_eigs[:, i])
                right_eigs[:, i] = right_eigs[:, i] - np.dot(right_eigs[:i, :], A)
            right_eigs[:, i] = right_eigs[:, i] / np.linalg.norm(right_eigs[:, i])
            left_eigs[:, i] = np.dot(residual, right_eigs[:, i])
            if i > 0:
            	B = np.dot(left_eigs[:, :i].T, left_eigs[:, i])
            	left_eigs[:, i] = left_eigs[:, i] - np.dot(left_eigs[:i, :], A)
            eigenvalues[i] = np.linalg(left_eigs[:, i])
            left_eigs[:, i] = left_eigs[:, i] / eigenvalues[i]
            if abs(eigenvalues[i] - mu) <= error:
            	break
            mu = eigenvalues[i]
        residual = residual - (eigenvalues[i] * np.dot(left_eigs[:, i], right_eigs[:, i].T))
    if results.lower().startswith('p'):  # for pca
    	return np.dot(left_eigs, eigenvalues), right_eigs, residual
    elif results.lower().startswith('s'):  # for svd
    	return left_eigs, right_eigs, eigenvalues
