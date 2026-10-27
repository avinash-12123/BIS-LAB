import numpy as np
import random

class SchoolBusACO:
    def __init__(self, num_stops, distance_matrix, demands, bus_capacity, 
                 num_ants=20, max_iter=100, alpha=1.0, beta=2.0, decay=0.1, Q=100):
        self.num_stops = num_stops  # Node 0 is the School
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.bus_capacity = bus_capacity
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha  
        self.beta = beta    
        self.decay = decay  
        self.Q = Q          
        
        self.pheromone = np.ones((num_stops, num_stops)) * 0.1

    def run(self):
        best_routes = None
        best_distance = float('inf')

        for iteration in range(self.max_iter):
            all_ant_routes = []
            all_ant_distances = []

            for ant in range(self.num_ants):
                routes, total_dist = self.construct_solution()
                all_ant_routes.append(routes)
                all_ant_distances.append(total_dist)

                if total_dist < best_distance:
                    best_distance = total_dist
                    best_routes = routes

            
            self.pheromone *= (1.0 - self.decay)
            for routes, total_dist in zip(all_ant_routes, all_ant_distances):
                for route in routes:
                    for i in range(len(route) - 1):
                        u, v = route[i], route[i+1]
                        self.pheromone[u][v] += self.Q / total_dist

        return best_routes, best_distance

    def construct_solution(self):
        unvisited = set(range(1, self.num_stops))  # School (0) is excluded from targets
        routes = []
        total_distance = 0

        while unvisited:
            route = [0]  # Every bus starts at the school/depot
            current_capacity = self.bus_capacity
            current_node = 0

            while unvisited:
                
                candidates = [node for node in unvisited if self.demands[node] <= current_capacity]
                if not candidates:
                    break  # Bus is full, needs to return to school

                
                probabilities = []
                for node in candidates:
                    phero = self.pheromone[current_node][node] ** self.alpha
                    # Avoid division by zero if distance is 0
                    dist_val = self.distance_matrix[current_node][node]
                    heur = (1.0 / dist_val if dist_val > 0 else 1e-5) ** self.beta
                    probabilities.append(phero * heur)

                prob_sum = sum(probabilities)
                if prob_sum == 0:
                    next_node = random.choice(candidates)
                else:
                    probabilities = [p / prob_sum for p in probabilities]
                    next_node = random.choices(candidates, weights=probabilities, k=1)[0]

                route.append(next_node)
                current_capacity -= self.demands[next_node]
                unvisited.remove(next_node)
                total_distance += self.distance_matrix[current_node][next_node]
                current_node = next_node

            route.append(0)  # Complete loop back to school
            total_distance += self.distance_matrix[current_node][0]
            routes.append(route)

        return routes, total_distance

if __name__ == "__main__":
    num_nodes = 6
    capacity = 15  # Bus capacity limit

    stop_demands = [0, 5, 8, 4, 6, 7] 

    np.random.seed(42)
    coordinates = np.random.rand(num_nodes, 2) * 100
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])

    aco = SchoolBusACO(
        num_stops=num_nodes, 
        distance_matrix=dist_matrix, 
        demands=stop_demands, 
        bus_capacity=capacity,
        max_iter=150,
        num_ants=25
    )
    
    optimal_routes, min_distance = aco.run()

    print(f"--- Optimization Complete ---")
    print(f"Minimum Total Fleeting Distance: {min_distance:.2f} km\n")
    print("Generated Dispatch Routes:")
    for index, route in enumerate(optimal_routes, 1):
        route_demand = sum(stop_demands[node] for node in route)
        print(f"  Bus {index}: {' -> '.join(map(str, route))} (Students Transported: {route_demand}/{capacity})")
