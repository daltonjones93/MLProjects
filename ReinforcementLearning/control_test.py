import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

class LinearSystem:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.sys = ctrl.ss(A, B, C, D)  # Create a state-space system

    def simulate(self, initial_state, time):
        return ctrl.initial_response(self.sys, T=time, X0=initial_state)

class LQGController:
    def __init__(self, A, B, C, Q, R, N):
        self.K, _, _ = ctrl.lqr(A, B, Q, R)  # Compute LQR controller gain
        self.L, _, _ = ctrl.lqr(A.T, C.T, Q, R)  # Compute Kalman filter gain
        self.L = self.L.T
        self.N = N

    def control_input(self, state_estimate, reference):
        return -self.K @ state_estimate + self.N @ reference

    def update_state_estimate(self, state_estimate, measurement):
        print(self.L.shape, self.L)
        print(measurement.shape)
        return (self.L @ measurement).flatten()

# Define system matrices
A = np.array([[0, 1], [-1, -1]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Create Linear System and LQG Controller
system = LinearSystem(A, B, C, D)
Q = np.eye(2)  # State cost matrix
R = 1  # Control cost
N = np.zeros((2, 1))  # Feedforward term
lqg_controller = LQGController(A, B, C, Q, R, N)

# Simulation parameters
initial_state = np.array([0, 0])
time = np.linspace(0, 5, 500)

# Simulate the system without control
response_no_control = system.simulate(initial_state, time)

# Simulate the system with LQG control
state_history = []
control_history = []
state_estimate = np.zeros((2, 1))

for t in time:
    reference = np.array([1])  # Setpoint
    control_input = lqg_controller.control_input(state_estimate, reference)
    state_estimate = lqg_controller.update_state_estimate(state_estimate, np.array([reference]))  # Assuming we have direct measurement of the output
    state_history.append(state_estimate.flatten())
    control_history.append(control_input.flatten())

# Plot results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time, response_no_control[1], label='Without Control')
plt.plot(time, np.array(state_history)[:, 1], label='With LQG Control')
plt.title('System Response')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, control_history, label='Control Input')
plt.title('Control Input')
plt.xlabel('Time')
plt.ylabel('Control Input')
plt.legend()

plt.tight_layout()
plt.show()
