#!/usr/bin/env python
#coding: utf-8
#this file is used to compute the low rank factors of the correlation matrix R using randomized svd
#Usage: python compute_lowrank_factors.py
#R comes from a Matern GP kernel with fixed parameters (nu = 1.5, correlation length 0.45 degrees, about 50 km);
#R is approximated as U_R diag(s_R) U_R', r = 15 the target low rank chosen such that r << n

import torch
import gpytorch
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

#load template file to get coordinates
template_file = '../../data_fusion/input/WaPOR_ETa_2003.01.01.tif'
with rioxarray.open_rasterio(template_file) as rds:
    x_coords = rds.x.values.astype(np.float32)
    y_coords = rds.y.values.astype(np.float32)
    template_data = rds.squeeze().values.astype(np.float32)

LON, LAT = np.meshgrid(x_coords, y_coords) #create spatial coordinate grid
spatial_coords = np.stack([LON.flatten(), LAT.flatten()], axis=1)
valid_mask = ~np.isnan(template_data.flatten())

print(f"grid size: {len(x_coords)} x {len(y_coords)}")
print(f" valid pixels: {np.sum(valid_mask)}")
valid_coords = torch.tensor(spatial_coords[valid_mask], dtype=torch.float32).to(device)
n_pixels = len(valid_coords)

lengthscale = 0.45
outputscale = 1.0
nu = 1.5

#Create kernel
base_kernel = gpytorch.kernels.MaternKernel(nu=nu)
covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
base_kernel.lengthscale = torch.tensor(lengthscale)
covar_module.outputscale = torch.tensor(outputscale)

base_kernel = base_kernel.to(device)
covar_module = covar_module.to(device)
valid_coords = valid_coords.to(device)


#R appriximation using randomized svd

def rsvd(valid_coords, covar_module, k, n_oversamples=10,
                              n_iter=3, batch_size=2000, device='mps'):
    torch.manual_seed(42)
    n = len(valid_coords)
    l = k + n_oversamples

    coords_tensor = valid_coords.to(device)

    def matvec(V):
        V_dev = V.to(device)
        result = torch.zeros((n, V.shape[1]), dtype=torch.float32, device=device)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                end_i = min(i + batch_size, n)
                batch_i = coords_tensor[i:end_i]

                for j in range(0, n, batch_size):
                    end_j = min(j + batch_size, n)
                    batch_j = coords_tensor[j:end_j]

                    K_block = covar_module(batch_i, batch_j).evaluate()
                    result[i:end_i] += K_block @ V_dev[j:end_j]

        return result.cpu()

    Omega = torch.randn(n, l, dtype=torch.float32)
    print("computing Y = R @ Omega")
    Y = matvec(Omega)

    for it in range(n_iter):
        print(f"implementing power iteration {it+1}/{n_iter}:")
        Q, _ = torch.linalg.qr(Y)
        Y = matvec(matvec(Q))
        if str(device) == 'mps':
            torch.mps.empty_cache()

    Q, _ = torch.linalg.qr(Y)

    print("computing B = Q.T @ R @ Q")
    RQ = matvec(Q)
    B = Q.T @ RQ
    U_hat, s, _ = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    return U[:, :k].numpy(), s[:k].numpy()


#compute low rank factors

k = 15
U, s = rsvd(
    valid_coords, covar_module, k=k,
    n_iter=3, batch_size=2000, device=device)

print(f"singular values: {s}")
print(f"condition number: {s[0]/s[-1]:.2e}")
print(f"cumulative energy: {np.cumsum(s**2)/np.sum(s**2)}")

total_variance = np.sum(s**2)
cumulative = np.cumsum(s**2) / total_variance

print(f"variance captured by top 10: {cumulative[9]*100:.1f}%")
print(f"variance captured by top 15: {cumulative[14]*100:.1f}%")
print(f"variance captured by all {k}: {cumulative[-1]*100:.1f}%")

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.semilogy(s, 'o')
plt.xlabel('index')
plt.ylabel('singular value')
plt.title('singular value decay')

plt.subplot(1, 2, 2)
plt.plot(cumulative * 100)
plt.axhline(95, color='r', linestyle='--', label='95%')
plt.xlabel('number of components')
plt.ylabel('cumulative variance (%)')
plt.legend()
plt.tight_layout()
plt.savefig('../output/singular_values.png', dpi=250)

#save the low rank factors
np.savez('../output/lowrank_factors.npz', U=U, s=s, valid_mask=valid_mask)


#check orthonormality

UtU = U.T @ U
ortho_error = np.linalg.norm(UtU - np.eye(k), 'fro')
print(f"orthonormality check ||U.T @ U - I||_F: {ortho_error:.6e}")


#helper functions

def R_matvec(v):
    "R @ v using low-rank approximation"
    return U @ (s * (U.T @ v))
def R_inv_matvec(v):
    "R^-1 @ v using low-rank approximation"
    return U @ ((U.T @ v) / s)
def true_R_matvec(v, batch_size=2000):
    "R @ v using true kernel (for validation)"
    n = len(valid_coords)
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    result = torch.zeros(n, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                K = covar_module(valid_coords[i:end_i], valid_coords[j:end_j]).evaluate()
                result[i:end_i] += K @ v_t[j:end_j]

    return result.cpu().numpy()


#validate the approximation and its inverse

#using randomly generated pixels
#to validate R^-1
np.random.seed(42)
v = np.random.randn(n_pixels)
v_proj = U @ (U.T @ v)

R_inv_v = R_inv_matvec(v_proj)
R_true_R_inv_v = true_R_matvec(R_inv_v)

rel_error = np.linalg.norm(R_true_R_inv_v - v_proj) / np.linalg.norm(v_proj)
print(f"||R_true @ R_inv @ v_proj - v_proj|| / ||v_proj||: {rel_error:.6e}")

assert rel_error < 1e-3, "R_inv validation failed!"


#compare original and low rank approximation

subset = 10000

#true R
coords_subset = valid_coords[:subset]

with torch.no_grad():
    R_true_subset = covar_module(coords_subset, coords_subset).evaluate().cpu().numpy()

#low-rank R
U_subset = U[:subset, :]
R_approx_subset = (U_subset * s) @ U_subset.T
#diff
R_diff = R_true_subset - R_approx_subset
rel_error = np.linalg.norm(R_diff, 'fro') / np.linalg.norm(R_true_subset, 'fro')
print(f"relative Frobenius error: {rel_error:.4f}")


fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(R_true_subset, cmap='viridis')
axes[0].set_title('Original R (True Kernel)')
plt.colorbar(im0, ax=axes[0], label='Correlation')

im1 = axes[1].imshow(R_approx_subset, cmap='viridis')
axes[1].set_title(f'Low-rank Approximation (k={k})')
plt.colorbar(im1, ax=axes[1], label='Correlation')

im2 = axes[2].imshow(R_diff, cmap='viridis')
axes[2].set_title('Difference (True - Approx)')
plt.colorbar(im2, ax=axes[2], label='Error')

plt.tight_layout()
plt.savefig('../output/R_comparison.png', dpi=250)
