from systems import DoubleIntegrator, Car
from constraints import CircleConstraintForDoubleIntegrator, CircleConstraintForCar
from cddp import CDDP
import numpy as np


if __name__ == '__main__':
	system = Car()
	system.set_cost(np.zeros((4, 4)), 0.05*np.identity(2))
	Q_f = np.identity(4)
	Q_f[0, 0] = 50
	Q_f[1, 1] = 50
	Q_f[2, 2] = 50
	Q_f[3, 3] = 10
	system.set_final_cost(Q_f)

	solver = CDDP(system, np.zeros(4), horizon=100)
	

	#solve for initial trajectories
	system.set_goal(np.array([3, 3, np.pi/2, 0]))
	for i in range(10):
		solver.backward_pass()
		solver.forward_pass()

	# constraint = CircleConstraintForCar(np.ones(2), 0.5, system)
	# constraint2 = CircleConstraintForCar(np.array([2, 2]), 1.0, system)
		
	constraint = CircleConstraintForCar(np.array([1,1]), 0.5, system)
	constraint2 = CircleConstraintForCar(np.array([1, 2.5]), 0.5, system)
	constraint3 = CircleConstraintForCar(np.array([2.5, 2.5]), 0.5, system)
	solver.add_constraint(constraint)
	# solver.add_constraint(constraint2)
	solver.add_constraint(constraint3)
	system.set_goal(np.array([3, 3, np.pi/2, 0]))
	for i in range(2):
		solver.backward_pass()
		solver.forward_pass()
	solver.system.draw_trajectories(solver.x_trajectories)