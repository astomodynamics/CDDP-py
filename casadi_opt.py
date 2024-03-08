import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

class TrajectoryOptimizationNode():
    def __init__(self):
        # Define parameters
        self.N_ = 200 # optimizaiton horizon
        self.tf_ = 10.0 # Final time
        self.dt_ = self.tf_ / self.N_ # optimization time step
        self.n_ = 3 # state dimension
        self.m_ = 2 # control dimension

        self.integrator_ = "euler" # euler or rk4

        self.goal_state_ = ca.DM([
            4.0, 6.0, np.pi/2
        ])

        self.current_state_ = ca.DM([
            3.0, 0.0, np.pi/2
        ])

        self.initialize_bounds()
        self.initialize_costs()

        self.initialize_problem()
        self.optimize_trajectory()
        
    def initialize_bounds(self):
        self.x_min_ = -0.1
        self.x_max_ = 5.0
        self.y_min_ = -0.1
        self.y_max_ = 7.0
        self.theta_min_ = -2*np.pi 
        self.theta_max_ = 2*np.pi

        self.v_min_ = -0.5
        self.v_max_ = 0.5
        self.omega_min_ = -np.pi
        self.omega_max_ = np.pi
        # self.v_min_ = -10
        # self.v_max_ = 10
        # self.omega_min_ = -10
        # self.omega_max_ = 10
    
    def initialize_costs(self):
        self.Q_x_ = 1e-1
        self.Q_y_ = 1e-1
        self.Q_theta_ = 1e-2

        self.R_v_ = 1e-0
        self.R_omega_ = 1e-0

        self.Q_x_f_ = 10
        self.Q_y_f_ = 10
        self.Q_theta_f_ = 1.0

    def initialize_problem(self):
        # 1. Define states, controls (as CasADi SX or MX symbols)
        # State
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)

        # Control
        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)

        # Matrix containing all states over all time step +1
        X = ca.SX.sym('X', self.n_, self.N_ + 1)

        # Matrix containing all controls over all time step
        U = ca.SX.sym('U', self.m_, self.N_)

        # Column vector for storing initial and final states
        P = ca.SX.sym('P', self.n_ + self.n_)

        # State weight matrix 
        Q = ca.diagcat(self.Q_x_, self.Q_y_, self.Q_theta_)

        # Control weight matrix
        R = ca.diagcat(self.R_v_, self.R_omega_)

        # Terminal state wight matrix
        Q_f = ca.diagcat(self.Q_x_f_, self.Q_y_f_, self.Q_theta_f_)


        # 2. Formulate dynamics 
        # dynamics: 
        # f(x,u) = [V * cos(theta), V* sin(theta), omega]
        RHS = ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            omega
        )
        # Create dynamics function
        f = ca.Function('f', [states, controls], [RHS])

        # 3. Define cost function
        cost_func = 0

        # Set initial constraint
        g = X[:,0] - P[:self.n_]

        # numerical integration
        if self.integrator_ == 'euler':
            # Euler integration
            for k in range(self.N_):
                state = X[:, k]
                control = U[:, k]

                cost_func += (state - P[self.n_:]).T @ Q @ (state - P[self.n_:]) + control.T @ R @ control
                
                state_dot = f(state, control)

                state_next = X[:, k+1]
                state_next_euler = state + self.dt_ * state_dot
                g = ca.vertcat(g, state_next - state_next_euler)

            cost_func += (X[:,-1] - P[self.n_:]).T @ Q_f @ (X[:,-1] - P[self.n_:])

        elif self.integrator_ == 'rk4':
            # Runge-kutta 4th Integration
            # Runge-kutta 4th Integration
            for k in range(self.N_):
                state = X[:, k]
                control = U[:, k]

                cost_func += (state - P[self.n_:]).T @ Q @ (state - P[self.n_:]) + control.T @ R @ control
                
                # RK4 Stages
                k1 = f(state, control)
                k2 = f(state + self.dt_/2 * k1, control) 
                k3 = f(state + self.dt_/2 * k2, control)
                k4 = f(state + self.dt_ * k3, control)

                # State propagation
                state_next = X[:, k+1]
                state_next_rk4 = state + self.dt_ / 6 * (k1 + 2*k2 + 2*k3 + k4)  
                g = ca.vertcat(g, state_next - state_next_rk4)

            cost_func += (X[:,-1] - P[self.n_:]).T @ Q_f @ (X[:,-1] - P[self.n_:])

        # 4. Define constraints
        # Initial constraint is already set
            
        # Set terminal constraint
        g = ca.vertcat(g, X[:,-1] - P[self.n_:])

        # Rectangular obstacle
        x_min_obs = 2.0
        x_max_obs = 5.0
        y_min_obs = 2.8
        y_max_obs = 4.3
        c_safety = 0.1
        # Line segments 
        obs_line_1 = [x_min_obs, y_min_obs, x_min_obs, y_max_obs]  # Left side 
        obs_line_2 = [x_min_obs, y_max_obs, x_max_obs, y_max_obs]  # Top side
        obs_line_3 = [x_max_obs, y_max_obs, x_max_obs, y_min_obs]  # Right side
        obs_line_4 = [x_max_obs, y_min_obs, x_min_obs, y_min_obs]  # Bottom side

        print("size of g:", g.size())
        # Setup line segments into constraint
        # for k in range(self.N_):  
        #     state = X[:, k]  

        #     # Distance to each obstacle line (replace with suitable distance function)
        #     dist_to_line1 = self.distance_point_to_line(state, obs_line_1) 
        #     dist_to_line2 = self.distance_point_to_line(state, obs_line_2)
        #     dist_to_line3 = self.distance_point_to_line(state, obs_line_3)
        #     dist_to_line4 = self.distance_point_to_line(state, obs_line_4)

        #     # Constraints 
        #     g = ca.vertcat(g, dist_to_line1 - c_safety)  
        #     g = ca.vertcat(g, dist_to_line2 - c_safety) 
        #     g = ca.vertcat(g, dist_to_line3 - c_safety)
        #     g = ca.vertcat(g, dist_to_line4 - c_safety)
        


        # Converting states and controls into single dimension
        OPT_variables = ca.vertcat(
            X.reshape((-1, 1)), # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            U.reshape((-1, 1))
        )

        # 5. Define Non-Linear Programing (NLP) problem
        prob = {
            'f': cost_func,
            'x': OPT_variables,
            'g': g,
            'p': P
        }

        # 6. Setup options
        opts = {
            'ipopt': {
                'max_iter': 2000,
                # 'print_level': 0,
                # 'acceptable_tol': 1e-8,
                # 'acceptable_obj_change_tol': 1e-6
            },
            # 'print_time': 1
        }

        # 7. Setup nlp solver
        self.solver_ = ca.nlpsol('solver', 'ipopt', prob, opts)

        # setup variable bounds
        lbx = -ca.DM.ones((self.n_ * (self.N_ + 1) + self.m_ * self.N_)) * ca.inf
        ubx = ca.DM.ones((self.n_ * (self.N_ + 1) + self.m_ * self.N_)) * ca.inf

        lbx[0: self.n_*(self.N_+1): self.n_] = self.x_min_     # X lower bound
        lbx[1: self.n_*(self.N_+1): self.n_] = self.y_min_     # Y lower bound
        lbx[2: self.n_*(self.N_+1): self.n_] = self.theta_min_ # theta lower bound

        ubx[0: self.n_*(self.N_+1): self.n_] = self.x_max_      # X upper bound
        ubx[1: self.n_*(self.N_+1): self.n_] = self.y_max_      # Y upper bound
        ubx[2: self.n_*(self.N_+1): self.n_] = self.theta_max_  # theta upper bound

        # Bounds for controls 
        lbx[self.n_ * (self.N_ + 1) + 0: self.n_*(self.N_+1) + self.m_ * self.N_ : self.m_] = self.v_min_   # v lower bound 
        ubx[self.n_ * (self.N_ + 1) + 0: self.n_*(self.N_+1) + self.m_ * self.N_ : self.m_] = self.v_max_  # v upper bound 
        lbx[self.n_ * (self.N_ + 1) + 1: self.n_*(self.N_+1) + self.m_ * self.N_ : self.m_] = self.omega_min_  # omega lower bound
        ubx[self.n_ * (self.N_ + 1) + 1: self.n_*(self.N_+1) + self.m_ * self.N_ : self.m_] = self.omega_max_  # omega upper bound

        self.lbx_ = lbx
        self.ubx_ = ubx

        lbg = ca.vertcat(
            ca.DM.zeros((self.n_*(self.N_+1) + self.n_, 1)),
            # -ca.inf * ca.DM.ones((4 * self.N_, 1))
        )

        ubg = ca.vertcat(
            ca.DM.zeros((self.n_*(self.N_+1) + self.n_, 1)),
            # ca.DM.zeros((4 * self.N_, 1))
        )

        # Setup constraints arguments
        self.args_ = {
            'lbg': lbg,  # constraints lower bound
            'ubg': ubg,  # constraints upper bound
            'lbx': lbx, # variables lower bound
            'ubx': ubx  #variables upper bound
        }

        


    def optimize_trajectory(self):
        # 1. Get current state and goal state from subscribers
        x0 = self.current_state_
        xf = self.goal_state_

        print("x0:", x0)
        print("xf:", xf)

        # 2. Initialize NLP problem
        U0 = ca.DM.zeros((self.m_, self.N_))  # initial control
        X0 = ca.repmat(x0, 1, self.N_+1)      # initial state full
        self.args_['x0'] = ca.vertcat(
            ca.reshape(X0, self.n_ * (self.N_ + 1), 1),
            ca.reshape(U0, self.m_ * self.N_, 1)
        )

        # 3. Set initial and final states
        self.args_['p'] = ca.vertcat(x0, xf)

        # 4. Solve nlp problem
        sol = self.solver_(
            **self.args_
        )

        # Reshape the solution
        X_sol = ca.reshape(sol['x'][: self.n_ * (self.N_ + 1)], self.n_, self.N_ + 1)
        U_sol = ca.reshape(sol['x'][self.n_ * (self.N_ + 1):], self.m_, self.N_)

        self.X_sol_ = X_sol
        self.U_sol_ = U_sol
        print(U_sol[:,0])
        print(X_sol[:,0])
        print(X_sol[:,-1])

        print(X_sol)

    def distance_point_to_line(self, state, line_segment):
        """Calculates the distance between a point and a line segment using CasADi.

        Args:
            point: A CasADi SX or MX representing the point (x, y).
            line_segment: A CasADi SX or MX of shape (4,) representing the line segment
                        with coordinates (x1, y1, x2, y2).

        Returns:
            CasADi SX or MX: The distance between the point and the line segment.
        """

        x1, y1, x2, y2 = line_segment

        line_vec = ca.vertcat(x2 - x1, y2 - y1)
        point_vec = ca.vertcat(state[0] - x1, state[1] - y1)

        line_len = ca.norm_2(line_vec)
        line_unitvec = line_vec / line_len

        projection = point_vec.T @ line_unitvec * line_unitvec

        # CasADi conditional logic
        closest_point = ca.if_else(projection[0] < 0, ca.vertcat(x1, y1), 
                                ca.if_else(projection[0] > line_len, ca.vertcat(x2, y2), projection))

        distance = ca.norm_2(point_vec - closest_point)

        return distance 


    def get_solution(self):
        return (self.X_sol_, self.U_sol_)
    

traj_opt_node = TrajectoryOptimizationNode()

X_sol, U_sol = traj_opt_node.get_solution()

X = np.zeros((3,traj_opt_node.N_))
U = np.zeros((2,traj_opt_node.N_))
print(X_sol.size())
for i in range(traj_opt_node.N_-1):
    X[0,i] = float(X_sol[0,i])
    X[1,i] = float(X_sol[1,i])
    X[2,i] = float(X_sol[2,i])
    U[0,i] = float(U_sol[0,i])
    U[1,i] = float(U_sol[1,i])

# Plot the trajectory
plt.figure(figsize=(8,6)) # Adjust figure size if needed
plt.plot(X[0, :-1], X[1, :-1], label='Optimized Trajectory')

# Plot the initial state
plt.scatter(X[0,0], X[1,0], label='Start', color='g')

# Plot the goal state
plt.scatter(X[0,-1], X[1,-1], label='Goal', color='r')

# Plot the obstacle
x_min_obs = 2.0
x_max_obs = 5.0
y_min_obs = 2.8
y_max_obs = 4.3
plt.plot([x_min_obs, x_min_obs], [y_min_obs, y_max_obs], color='black')
plt.plot([x_min_obs, x_max_obs], [y_max_obs, y_max_obs], color='black')
plt.plot([x_max_obs, x_max_obs], [y_max_obs, y_min_obs], color='black')
plt.plot([x_max_obs, x_min_obs], [y_min_obs, y_min_obs], color='black')


# Optional: Customize plot
plt.xlabel('X') 
plt.ylabel('Y')
plt.title('Optimized Trajectory')
plt.grid(True)
plt.legend()

# Plot the trajectory
plt.figure(figsize=(8,6)) # Adjust figure size if needed
plt.plot(U[0, :-2], label='v')
plt.plot(U[1,:-2], label='omega')

# Optional: Customize plot
plt.grid(True)
plt.legend()

# Show the plot
plt.show() 