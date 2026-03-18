import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations


def plot_image(x_vector):
    img_reshaped = x_vector.reshape(3, 32, 32)

    img_transposed = img_reshaped.transpose(1, 2, 0)

    img_min = img_transposed.min()
    img_max = img_transposed.max()
    img_normalized = (img_transposed - img_min) / (img_max - img_min)

    plt.imshow(img_normalized)
    plt.axis("off")
    plt.show()


def linear_kernel(X_1, X_2):
    gram = X_1 @ X_2.transpose()
    return gram


def rbf_kernel(X_1, X_2, gamma):
    n1 = X_1.shape[0]
    n2 = X_2.shape[0]
    d = X_1.shape[1]
    L = (X_1**2).sum(axis=1)  # (n1,)
    C = (X_2**2).sum(axis=1)  # (n2,)
    L = np.hstack(n2 * [L.reshape(n1, 1)])  # (n1,n2)
    C = np.vstack(n1 * [C])  # (n1,n2)
    gram = np.exp(-gamma * (L + C - 2 * linear_kernel(X_1, X_2)))
    return gram


def polynomial_kernel(X_1, X_2, c=0.0, degree=3):
    return (X_1 @ X_2.T + c) ** degree


class OvOKernelRidge:
    def __init__(self, kernel_func, lambda_reg=0.1):
        self.kernel_func = kernel_func
        self.lambda_reg = lambda_reg
        self.classes = np.arange(10)
        self.models = {}

    def train(self, Y, X):
        for class_A, class_B in combinations(self.classes, 2):
            mask = np.array([Y == class_A, Y == class_B]).any(axis=0)
            X_loc = X[mask]
            Y_loc = Y[mask]
            n = Y_loc.shape[0]
            A_mask = Y_loc == class_A
            B_mask = Y_loc == class_B
            Y_loc[A_mask] = 1
            Y_loc[B_mask] = -1
            K = self.kernel_func(X_loc, X_loc)
            alpha = np.linalg.solve(K + n * self.lambda_reg * np.eye(n), Y_loc)
            self.models[(class_A, class_B)] = {"alpha": alpha, "X": X_loc}

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))

        for (class_A, class_B), model_data in self.models.items():
            alpha = model_data["alpha"]
            X = model_data["X"]
            G = self.kernel_func(X_test, X)
            pred = G @ alpha
            votes[pred > 0, class_A] += 1
            votes[pred < 0, class_B] += 1
        Yte = votes.argmax(axis=-1)
        return Yte


# Turn data into HOG data
def compute_gradients_and_orientations(X):
    N = X.shape[0]
    X_reshaped = X.reshape(N, 3, 32, 32)
    X_gray = (
        0.299 * X_reshaped[:, 0, :, :]
        + 0.587 * X_reshaped[:, 1, :, :]
        + 0.114 * X_reshaped[:, 2, :, :]
    )

    Gx = np.zeros(X_gray.shape)  # padding avec zeros
    Gy = np.zeros(X_gray.shape)

    Gx[:, :, 1:-1] = X_gray[:, :, 2:] - X_gray[:, :, :-2]  # dérivée spatiale discrète
    Gy[:, 1:-1, :] = X_gray[:, 2:, :] - X_gray[:, :-2, :]

    mod = np.sqrt(Gx**2 + Gy**2)
    arg = np.rad2deg(np.arctan2(Gy, Gx)) % 180

    return mod, arg


def build_hog_histograms(mod, arg):
    N = mod.shape[0]

    mod_reshaped = mod.reshape(N, 4, 8, 4, 8)
    arg_reshaped = arg.reshape(N, 4, 8, 4, 8)

    mod_cells = mod_reshaped.transpose(0, 1, 3, 2, 4)
    arg_cells = arg_reshaped.transpose(0, 1, 3, 2, 4)

    mod_flat = mod_cells.reshape(N, 4, 4, 64)
    arg_flat = arg_cells.reshape(N, 4, 4, 64)

    bin_indices = (arg_flat / 20).astype("int") % 9

    histograms = np.zeros((N, 4, 4, 9))

    for b in range(9):
        mask = bin_indices == b
        histograms[:, :, :, b] = np.sum(mod_flat * mask, axis=3)

    return histograms


def normalize_hog_blocks(histograms):
    N = histograms.shape[0]

    blocks = np.zeros((N, 3, 3, 2, 2, 9))

    for y in range(3):
        for x in range(3):
            blocks[:, y, x, :, :, :] = histograms[:, y : y + 2, x : x + 2, :]

    blocks_flat = blocks.reshape(N, 3, 3, 36)

    block_norms = np.linalg.norm(blocks_flat, axis=3, keepdims=True) + 1e-7

    blocks_normalized = blocks_flat / block_norms

    X_hog = blocks_normalized.reshape(N, -1)

    return X_hog


best_lambda = 0.0001
best_gamma = 5.0
# nouveau prétraitement (sans cross-val ici)
Xtr_raw = np.array(pd.read_csv("Xtr.csv", header=None, sep=",", usecols=range(3072)))
Xte_raw = np.array(pd.read_csv("Xte.csv", header=None, sep=",", usecols=range(3072)))
Ytr_raw = np.array(pd.read_csv("Ytr.csv", sep=",", usecols=[1])).squeeze()

# data augmentation

Xtr_flipped = Xtr_raw.reshape(-1, 3, 32, 32)[:, :, :, ::-1].reshape(-1, 3072)
X_train_full_aug = np.vstack((Xtr_raw, Xtr_flipped))
Y_train_full_aug = np.concatenate((Ytr_raw, Ytr_raw))

mag_tr, ori_tr = compute_gradients_and_orientations(X_train_full_aug)
hist_tr = build_hog_histograms(mag_tr, ori_tr)
X_train_hog_unnorm = normalize_hog_blocks(hist_tr)

# ssur le Test
mag_te, ori_te = compute_gradients_and_orientations(Xte_raw)
hist_te = build_hog_histograms(mag_te, ori_te)
X_test_hog_unnorm = normalize_hog_blocks(hist_te)
# normalisation
norm_tr = np.linalg.norm(X_train_hog_unnorm, axis=1, keepdims=True) + 1e-7
X_train_final = X_train_hog_unnorm / norm_tr

norm_te = np.linalg.norm(X_test_hog_unnorm, axis=1, keepdims=True) + 1e-7
X_test_final = X_test_hog_unnorm / norm_te

# entrainement
final_kernel = lambda x1, x2: rbf_kernel(x1, x2, gamma=best_gamma)

model = OvOKernelRidge(kernel_func=final_kernel, lambda_reg=best_lambda)
model.train(Y_train_full_aug, X_train_final)
Yte_pred = model.predict(X_test_final)

dataframe = pd.DataFrame({"Prediction": Yte_pred})
dataframe.index += 1
dataframe.to_csv("Yte_pred.csv", index_label="Id")
print("Fichier généré sous le nom : Yte_pred.csv")
