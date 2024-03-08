
# MPC simulation for a differential-wheel mobile robot using OSQP 
import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

V_nominal = 0.001
nx, nu = 3, 2

# Differential-wheel mobile robot model
def dynamics(x, u):
    # Differential-wheel mobile robot model
    V = u[0]
    Omega = u[1]
    x_dot = V*np.cos(x[2])
    y_dot = V*np.sin(x[2])
    theta_dot = Omega
    return np.array([x_dot, y_dot, theta_dot])

# Discrete time model of a differential-wheel mobile robot
def linearized_dynamics(x, u, dt):
    V = u[0]
    theta = x[2]
    A = sparse.csc_matrix([
        [1, 0, -V*np.sin(theta)*dt],
        [0, 1,  V*np.cos(theta)*dt],
        [0, 0,  1]
    ])
    B = sparse.csc_matrix([
        [np.cos(theta)*dt, 0],
        [np.sin(theta)*dt, 0],
        [0,             dt]
    ])
    return A, B

# Constraints
u0 = 0.0
umin = np.array([-0.2, -np.pi/4]) - u0
umax = np.array([ 0.2,  np.pi/4]) - u0
xmin = np.array([-3, -3, -3*np.pi])
xmax = np.array([ 3,  3,  3*np.pi])

# Objective function
Q = sparse.diags([1.0, 1.0, 0.01])
QN = Q
R = sparse.diags([0.001, 0.1])

# Initial and reference states
x0 = np.array([0.,0.,0.0])
xr = np.array([2.,2.,np.pi/4])

# Prediction horizon
N = 500

# Time step
dt = 0.1

# Matrices for linearized dynamics
Ad, Bd = linearized_dynamics(xr, np.array([V_nominal, 0.0]), dt)

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')

# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])

# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq

# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u)

# Simulate in closed loop
N_sim = 5000
dt_sim = dt
X = [x0]
ctrl = np.array([0.0, 0.0])

for i in range(N_sim):
    print('Time: ' + str(i*dt))
    print('Current State: ' + str(x0))
    print('Reference State: ' + str(xr))

    if np.linalg.norm(x0[0:2] - xr[0:2]) < 0.01:
        break
    elif i > 100 and np.linalg.norm(ctrl) < 0.001:
        break

    if i*dt_sim % dt <= 0.0001:
        # Solve MPC problem
        res = prob.solve()
        ctrl = res.x[-N*nu:-(N-1)*nu]
        print('Control: ' + str(ctrl))
    
    x0 = dynamics(x0, ctrl) * dt + x0

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)
    X.append(x0)

# Plot results
X = np.array(X)
plt.figure()
plt.plot(X[:, 0], X[:, 1], 'b-')
plt.plot(xr[0], xr[1], 'ro')
plt.plot(X[0,0], X[0,1], 'rx')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MPC simulation')
plt.show()


