import random
import math

ASSETS = ["Stock A", "Stock B", "Stock C", "Bond", "Gold"]
NUM_ASSETS = len(ASSETS)
EXPECTED_RETURNS = [0.12, 0.10, 0.08, 0.06, 0.07]
RISK_FREE_RATE = 0.04
COVARIANCE = [
    [0.0625, 0.0100, 0.0080, 0.0020, 0.0050],
    [0.0100, 0.0400, 0.0090, 0.0020, 0.0040],
    [0.0080, 0.0090, 0.0225, 0.0015, 0.0030],
    [0.0020, 0.0020, 0.0015, 0.0064, 0.0020],
    [0.0050, 0.0040, 0.0030, 0.0020, 0.0144]
]

def create_individual():
    weights = [random.uniform(0.05, 0.40) for _ in range(NUM_ASSETS)]
    total = sum(weights)
    return [w / total for w in weights]

def calculate_return(portfolio):
    return sum(portfolio[i] * EXPECTED_RETURNS[i] for i in range(NUM_ASSETS))

def calculate_risk(portfolio):
    variance = 0
    for i in range(NUM_ASSETS):
        for j in range(NUM_ASSETS):
            variance += portfolio[i] * portfolio[j] * COVARIANCE[i][j]
    return math.sqrt(variance)

def calculate_sharpe_ratio(portfolio):
    portfolio_return = calculate_return(portfolio)
    portfolio_risk = calculate_risk(portfolio)
    return (portfolio_return - RISK_FREE_RATE) / portfolio_risk

def crossover(parent1, parent2):
    point = random.randint(1, NUM_ASSETS - 1)
    child = parent1[:point] + parent2[point:]
    total = sum(child)
    return [w / total for w in child]

def mutation(portfolio, p_m):
    if random.random() < p_m:
        i = random.randint(0, NUM_ASSETS - 1)
        change = random.uniform(-0.10, 0.10)
        portfolio[i] = max(0, portfolio[i] + change)
        total = sum(portfolio)
        if total > 0:
            portfolio = [w / total for w in portfolio]
    return portfolio

def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    return max(selected, key=calculate_sharpe_ratio)

def run_portfolio_ga(pop_size=30, p_c=0.85, p_m=0.15, generations=150):
    population = [create_individual() for _ in range(pop_size)]
    best_portfolio = max(population, key=calculate_sharpe_ratio)

    for gen in range(generations):
        new_population = []
        new_population.append(best_portfolio.copy())

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < p_c:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = mutation(child, p_m)
            new_population.append(child)

        population = new_population
        current_best = max(population, key=calculate_sharpe_ratio)

        if calculate_sharpe_ratio(current_best) > calculate_sharpe_ratio(best_portfolio):
            best_portfolio = current_best.copy()

    return best_portfolio, calculate_return(best_portfolio), calculate_risk(best_portfolio), calculate_sharpe_ratio(best_portfolio)

best_portfolio, expected_return, risk, sharpe = run_portfolio_ga(pop_size=30, p_c=0.85, p_m=0.15, generations=150)

print("Best Portfolio Found:")
for i in range(NUM_ASSETS):
    print(ASSETS[i], ":", round(best_portfolio[i] * 100, 2), "%")

print("Expected Annual Return:", round(expected_return * 100, 2), "%")
print("Portfolio Risk:", round(risk * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe, 2))
