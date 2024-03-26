from systems import CartPole
from cddp import CDDP
import numpy as np

if __name__ == "__main__":
    system = CartPole()
    system.set_cost(np.zeros((4, 4)), 0.01*np.identity(1))
    Q_f = np.identity(4)
    Q_f[0, 0] = 100
    Q_f[1, 1] = 100
    Q_f[2, 2] = 10
    Q_f[3, 3] = 10
    system.set_final_cost(Q_f)
    tN = 1000

    x0 = np.array([0, np.pi-np.pi/6, 0, 0])
    # simulate initial trajectory
    x_trajectories = np.zeros((4, tN))
    x_trajectories[:, 0] = x0
    for i in range(tN-1):
        x_trajectories[:, i+1] = system.rk4_step(x_trajectories[:, i], np.zeros(1))
    # system.draw_trajectories(x_trajectories)

    solver = CDDP(system,x0, horizon=1000)

    # solve for initial trajectories
    system.set_goal(np.array([0, 0, 0, 0]))
    print("total cost: ", system.calculate_final_cost(x0))
    for i in range(10):
        solver.backward_pass()
        solver.forward_pass()
    # print(solver.x_trajectories[:, -1])
    # solver.system.draw_trajectories(solver.x_trajectories)