#from sbkm.sbkm import SBKM
import sbkm.utils as utils
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import matplotlib.pyplot as plt
import pickle
from scipy.special import erf
'''
This demo shows how our collision checking works based on an "inflated boundary" developed in our paper.
We trained our map on the intel dataset and then store the class object for this demo.
'''
################################################# PARAMS ###############################################################
e = 0.2 # decision threshold param e
bar_e = (1+erf(e))/2 # decision threshold \bar{e}
N1 = 1 # Params for collision checking
N2 = 100 # Params for collision checking
epsilon = 0.1 # threshold for checking curves for collision
line_width = 2 # For plotting

################################################## LOAD THE SBKM OBJECT ################################################
f = open("./trained/sbkm_class_obj.pkl","rb")
sbkm_obj = pickle.load(f)
print("Number of relevance vectors:", len(sbkm_obj.Mn))
# Demo rebuild the posterior's mean and covariance using Laplace approximation.
sbkm_obj.global_posterior_approx()
# Demo building the rtree for collision checking.
# If the object is loaded from a file, i.e. the rtree is not saved, then we can call this function to rebuild the rtree.
sbkm_obj.build_rtree_collision_checking()
print("############################################# LOADED SBKM OBJECT ##############################################")

########################################### CALCULATE THE OCCUPANCY PROBABILITY ########################################
cell_max_min = (5, 20, -10, 0)
q_resolution = 0.1
x = np.arange(cell_max_min[0], cell_max_min[1]+1, q_resolution)
y = np.arange(cell_max_min[2], cell_max_min[3]+1, q_resolution)
xx, yy = np.meshgrid(x,y)
X_query = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
rv_grid = sbkm_obj.predict_proba(X_query)
Y_query = rv_grid[:, 1]
g3 = sbkm_obj.upperbound_g3(X_query, e = e, n1 = N1, n2 = N2)
print("Queried the map for visualization!")

######################################################## CHECK LINE SEGMENTS ###########################################

# Plot the map for visualization. Not needed for collision checking.
plt.figure(figsize=(16, 10))
plt.title('SBKM map')
plt.xlim([cell_max_min[0], cell_max_min[1]])
plt.ylim([cell_max_min[2], cell_max_min[3]])
plt.contourf(x, y, np.reshape(Y_query, (len(y), len(x))), 20, cmap='Greys')
cs1 = plt.contour(x, y, np.reshape(Y_query, (len(y), len(x))), levels=[bar_e], cmap="Greys_r", linestyles="solid", linewidths=line_width)
cs2 = plt.contour(x, y, np.reshape(g3, (len(y), len(x))), levels=[0.0], cmap="Greys_r", linestyles="dashed", linewidths=line_width)
plt.xlim([cell_max_min[0], cell_max_min[1]])
plt.ylim([cell_max_min[2], cell_max_min[3]])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
h1, _ = cs1.legend_elements()
h2, _ = cs2.legend_elements()

# Check line segments
print("############################################# CHECKING LINE SEGMENTS ##########################################")
A = np.array([14.6, -6.5])
B = np.array([15.6, -6.5])
v = np.linalg.norm(B-A)
line_count = 10
for i in range(line_count):
    Bi = A + v*np.array([np.cos(2*i*np.pi/line_count), np.sin(2*i*np.pi/line_count)])
    arr1 = plt.arrow(A[0], A[1], Bi[0]-A[0], Bi[1]-A[1], head_width=0.1,
                     head_length=0.1, fc="xkcd:green", ec="xkcd:green", width=0.05, label="colliding segment")
for i in range(line_count):
    Bi = A + v*np.array([np.cos(2*i*np.pi/line_count), np.sin(2*i*np.pi/line_count)])
    collision_free, t_uA, t_uB = sbkm_obj.check_line_segment(A, Bi, e = e, n1 = N1, n2 = N2, k_nearest=50)
    print("line number ", i , "free" if collision_free else "colliding", t_uA, t_uB)
    if not collision_free and 1 > t_uA > 0:
        #intersect = min(intersect,1.0)
        Bx = A + t_uA*v*np.array([np.cos(2*i*np.pi/line_count), np.sin(2*i*np.pi/line_count)])
        inter = plt.scatter(Bx[0], Bx[1], color = "xkcd:red", marker='x', s = 200, linewidths=10)

# Plot
plt.legend([h1[0], h2[0], arr1, inter], \
           ['true boundary', 'inflated boundary', 'line segments', 'intersection'], loc = 2, fontsize=15)
plt.show()


#################################### CHECK SECOND ORDER POLYNORMIAL CURVES ####################################################

# Plot the map for visualization. Not needed for collision checking.
plt.figure(figsize=(16, 10))
plt.title('SBKM map')
plt.xlim([cell_max_min[0], cell_max_min[1]])
plt.ylim([cell_max_min[2], cell_max_min[3]])
plt.contourf(x, y, np.reshape(Y_query, (len(y), len(x))), 20, cmap='Greys')
cs1 = plt.contour(x, y, np.reshape(Y_query, (len(y), len(x))), levels=[bar_e], cmap="Greys_r", linestyles="solid", linewidths=line_width)
cs2 = plt.contour(x, y, np.reshape(g3, (len(y), len(x))), levels=[0.0], cmap="Greys_r", linestyles="dashed", linewidths=line_width)
plt.xlim([cell_max_min[0], cell_max_min[1]])
plt.ylim([cell_max_min[2], cell_max_min[3]])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
h1, _ = cs1.legend_elements()
h2, _ = cs2.legend_elements()

# Checking a colliding curve
print("################################################# CHECKING CURVES #############################################")
t_f = 0.45
t = np.linspace(0,t_f, 1000)
c1 = [15, -6.5, -1.0]
c2 = [-7.0, 5.5, -20.0]
curve_x_colliding = c1[0] + c1[1]*t + c1[2]*t**2
curve_y_colliding = c2[0] + c2[1]*t + c2[2]*t**2
print("Checking a curve!")
curve_colliding = plt.plot(curve_x_colliding, curve_y_colliding, 'xkcd:red', markersize=1, label="colliding", linewidth=3)
curr_x = c1[0]
curr_y = c2[0]
radius = sbkm_obj.get_radius(np.array([curr_x, curr_y]), e = e, n1 = N1, n2 = N2, k_nearest=50)
next_t = 0.0
curr_t = 0.0
ax = plt.axes()
while radius > 0.1 and curr_t < t_f:
    circle = plt.Circle([curr_x, curr_y], radius, color='xkcd:red', fill=False, label="free ball", linewidth=3)
    inter = plt.scatter(curr_x, curr_y, color="xkcd:blue", marker='x', s=200, linewidths=20)
    ax.add_artist(circle)
    next_t = utils.solve_next_t(c1[2], c1[1], c1[0], c2[2], c2[1], c2[0], curr_t, radius)
    curr_x = c1[0] + c1[1] * next_t + c1[2] * next_t ** 2
    curr_y = c2[0] + c2[1] * next_t + c2[2] * next_t ** 2
    radius = sbkm_obj.get_radius(np.array([curr_x, curr_y]), e = e, n1 = N1, n2 = N2, k_nearest=50)
    curr_t = next_t

if curr_t < t_f:
    print("The first curve: colliding!")
else:
    print("The first curve: free!")
######################################## CHECKING A FREE POLYNOMIAL CURVE###############################################
t_f = 0.4
t = np.linspace(0,t_f, 1000)
c1 = [15, -11.0, 5.0]
c2 = [-7.0, 3.5, 8.0]
curve_x_free = c1[0] + c1[1]*t + c1[2]*t**2
curve_y_free = c2[0] + c2[1]*t + c2[2]*t**2
print("Checking another curve!")
curve_free = plt.plot(curve_x_free, curve_y_free, 'xkcd:blue', markersize=1, label="free", linewidth=3)
curr_x = c1[0]
curr_y = c2[0]
radius = sbkm_obj.get_radius(np.array([curr_x, curr_y]), e = e, n1 = N1, n2 = N2, k_nearest=50)
next_t = 0.0
curr_t = 0.0
ax = plt.axes()
while radius > 0.1 and curr_t < t_f:
    circle = plt.Circle([curr_x, curr_y], radius, color='xkcd:blue', fill=False, label="free ball", linewidth=3)
    inter = plt.scatter(curr_x, curr_y, color="xkcd:blue", marker='x', s=200, linewidths=20)
    ax.add_artist(circle)
    next_t = utils.solve_next_t(c1[2], c1[1], c1[0], c2[2], c2[1], c2[0], curr_t, radius)
    curr_x = c1[0] + c1[1] * next_t + c1[2] * next_t ** 2
    curr_y = c2[0] + c2[1] * next_t + c2[2] * next_t ** 2
    radius = sbkm_obj.get_radius(np.array([curr_x, curr_y]), e = e, n1 = N1, n2 = N2, k_nearest=50)
    curr_t = next_t
if curr_t < t_f:
    print("The second curve: colliding!")
else:
    print("The second curve: free!")

# Plot
plt.legend([h1[0], h2[0], curve_colliding[0], curve_free[0], inter], \
           ['true boundary', 'inflated boundary', 'colliding curve', 'free curve', 'intersection'], loc = 2, fontsize=15)
plt.show()