from sbkm.sbkm import SBKM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pickle
import time

#
# LOAD PARAMS
#
def load_parameters():
    parameters = {'intel': \
                      ('./data/intel.csv',
                       (0.2, 0.2), # grid resolution for occupied samples and free samples, respectively
                       (-20, 20, -25, 10),  # area to be mapped [x1_min, x1_max, x2_min, x2_max]
                       1, # skip
                       6.71,  # gamma: kernel parameter
                       25, # k_nearest: picking K nearest relevance vectors
                       20 # max_iter: maximum number of iterations
                       ),
                  }
    return parameters['intel']
fn_train, res, cell_max_min, skip, gamma, k_nearest, max_iter = load_parameters()

# read data
g = pd.read_csv(fn_train, delimiter=',').values
# 90% for training
X_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 0:3])
Y_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 3][:, np.newaxis]).ravel()  # * 2 - 1
# 10% for testing
X_test = np.float_(g[::10, 0:3])
Y_test = np.float_(g[::10, 3][:, np.newaxis]).ravel()  # * 2 - 1
print(len(g), len(Y_test), len(Y_train))
print(sum(Y_train), sum(Y_test))

#
# Plot the dataset accumulated from all time steps.
#
''''''
plt.figure()
plt.scatter(X_train[:, 1], X_train[:, 2], c=Y_train, s=2)
plt.title('Training data')
plt.colorbar()
plt.show()


#
# Set up our SBKM model
#
sbkm_map = SBKM(n_iter = max_iter, gamma = gamma)

#
# query locations for plotting map of the environment
#
q_resolution = 0.25
xx, yy = np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
                     np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
X_query = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

#
# Occupied (positive) and free (negative) data points can be sampled on a grid with different resolutions.
#
pres = res[0]
nres = res[1]
t1 = time.time()
max_t = 100
print("Total number of scans = ", max_t)
for ith_scan in range(0, max_t, skip):

    # extract data points of the ith scan
    ith_scan_indx = X_train[:, 0] == ith_scan
    print('{}th scan:\n  N={}'.format(ith_scan, np.sum(ith_scan_indx)))
    y = Y_train[ith_scan_indx]
    X = X_train[ith_scan_indx, 1:]

    # Update RVM model
    sbkm_map.fit(X, y, k_nearest = k_nearest, pres=pres, nres=nres)

    # query the model
    rv_grid = sbkm_map.predict_proba(X_query)
    Y_query = rv_grid[:,1]
    svrv = np.array(sbkm_map.all_rv_X)
    rvy = np.array(sbkm_map.all_rv_y)
    w = sbkm_map.Mn

    # plot data points, relevance vectors and our probabilistic map.
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    ones_ = np.where(y > 0.5)
    zeros_ = np.where(y < 0.5)
    plt.scatter(X[ones_, 0], X[ones_, 1], c='r', cmap='jet', s=5, edgecolors='')
    plt.scatter(X[zeros_, 0], X[zeros_, 1], c='b', cmap='jet', s=5, edgecolors='')
    plt.title('Data points at t={}'.format(ith_scan))
    plt.xlim([cell_max_min[0], cell_max_min[1]]);
    plt.ylim([cell_max_min[2], cell_max_min[3]])
    plt.subplot(132)
    ones_ = np.where(y == 1)
    pos_rv = plt.scatter(svrv[rvy > 0.5, 0], svrv[rvy > 0.5, 1], c='r', cmap='jet', s=5, edgecolors='')
    neg_rv = plt.scatter(svrv[rvy < 0.5, 0], svrv[rvy < 0.5, 1], c='b', cmap='jet', s=5, edgecolors='')
    plt.title('relevance vectors at t={}'.format(ith_scan))
    plt.xlim([cell_max_min[0], cell_max_min[1]]);
    plt.ylim([cell_max_min[2], cell_max_min[3]])
    plt.subplot(133)
    plt.title('Our map at t={}'.format(ith_scan))
    plt.scatter(X_query[:, 0], X_query[:, 1], c=Y_query, cmap='jet', s=10, marker='8', edgecolors='')
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlim([cell_max_min[0], cell_max_min[1]]);
    plt.ylim([cell_max_min[2], cell_max_min[3]])
    plt.savefig('./outputs/imgs/step' + str(ith_scan).zfill(3) + '.png', bbox_inches='tight')
    plt.close("all")


# f = open("./outputs/sbkm_class.pkl", "wb")
# pickle.dump(sbkm_map, f)
# f.close()
#
# print("Saved file")