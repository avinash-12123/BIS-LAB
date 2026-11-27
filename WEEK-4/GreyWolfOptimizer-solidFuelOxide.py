import numpy as np
def objective(x):
    T, Uf, Ua, J = x

    # Simplified SOFC voltage model
    E = 1.1 - 0.0002 * (T - 800)
    activation_loss = 0.05 * np.log(1 + J)
    ohmic_loss = 0.15 * J
    concentration_loss = 0.03 * J**2

    V = E - activation_loss - ohmic_loss - concentration_loss
    P = J * V * Uf * Ua

    return -P  

wolves = 20
iterations = 100
dimensions = 4

lb = np.array([700, 0.5, 0.5, 0.1])
ub = np.array([1000, 0.9, 0.9, 2.0])

positions = np.random.uniform(lb, ub, (wolves, dimensions))

alpha_pos = np.zeros(dimensions)
beta_pos = np.zeros(dimensions)
delta_pos = np.zeros(dimensions)

alpha_score = float("inf")
beta_score = float("inf")
delta_score = float("inf")

for t in range(iterations):

    for i in range(wolves):

        positions[i] = np.clip(positions[i], lb, ub)

        fitness = objective(positions[i])

        if fitness < alpha_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()

            beta_score = alpha_score
            beta_pos = alpha_pos.copy()

            alpha_score = fitness
            alpha_pos = positions[i].copy()

        elif fitness < beta_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()

            beta_score = fitness
            beta_pos = positions[i].copy()

        elif fitness < delta_score:
            delta_score = fitness
            delta_pos = positions[i].copy()

    a = 2 - t * (2 / iterations)

    for i in range(wolves):

        for j in range(dimensions):

            r1, r2 = np.random.random(), np.random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = abs(C1 * alpha_pos[j] - positions[i][j])
            X1 = alpha_pos[j] - A1 * D_alpha

            r1, r2 = np.random.random(), np.random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = abs(C2 * beta_pos[j] - positions[i][j])
            X2 = beta_pos[j] - A2 * D_beta

            r1, r2 = np.random.random(), np.random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = abs(C3 * delta_pos[j] - positions[i][j])
            X3 = delta_pos[j] - A3 * D_delta

            positions[i][j] = (X1 + X2 + X3) / 3

print("Optimal SOFC Parameters:")
print("Temperature:", alpha_pos[0])
print("Fuel Utilization:", alpha_pos[1])
print("Air Utilization:", alpha_pos[2])
print("Current Density:", alpha_pos[3])
print("Maximum Power Density:", -alpha_score)
