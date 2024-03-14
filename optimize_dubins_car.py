from systems import DoubleIntegrator, Car, DubinsCar
from constraints import CircleConstraintForDoubleIntegrator, CircleConstraintForCar
from cddp import CDDP
import numpy as np

if __name__ == "__main__":
        system = DubinsCar()
        system.set_cost(np.zeros((3, 3)), 0.05*np.identity(2))
        Q_f = np.identity(3)
        Q_f[0, 0] = 50
        Q_f[1, 1] = 50
        Q_f[2, 2] = 10
        system.set_final_cost(Q_f)

        x0 = np.array([0, 0, 0.0])
        solver = CDDP(system,x0, horizon=100)

        #solve for initial trajectories
        system.set_goal(np.array([2, 2, np.pi/2]))
        # print("total cost: ", system.calculate_final_cost(x0))
        for i in range(10):
            solver.backward_pass()
            solver.forward_pass()
        print(solver.x_trajectories[:, -1])
        solver.system.draw_trajectories(solver.x_trajectories)

            