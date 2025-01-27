import math
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Dynamic font settings
plt.rcParams['font.sans-serif'] = ['SimHei']  # To display Chinese characters correctly
plt.rcParams['axes.unicode_minus'] = False  # To display negative signs correctly

# Define objective functions
def evaluate(individual):
    x1, x2 = individual  # x1: Tourist count, x2: Tax
    # Maximize tourism revenue
    f1 = x1 * (x2 + 2000)
    # Minimize environmental pressure (assuming environmental pressure is proportional to tourist count)
    G = 0.4  # Environmental vulnerability
    f2 = (G * pow(x1, 1.05)) - math.log(x2)  # Environmental pressure related to tourist count
    # f2 = 0.1 * x1 + 0.01 * x1**2  # Environmental pressure function
    # Maximize resident satisfaction (assuming satisfaction is positively related to tourist count and negatively related to tax)
    f3 = 100 * 0.65 - 2.217 * x1 / (1 + x1) + 0.387 * x2
    f3 = np.clip(f3, 0, 100)
    # f3 = 0.5 * x1 - 0.2 * x2  # Resident satisfaction function
    # Minimize tourism hidden costs (assuming costs are proportional to tourist count and tax)
    f4 = 0.3 * x1 + 0.4 * x2  # Hidden costs function
    return f1, f2, f3, f4

# Define problem
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0, -1.0))  # Positive weights for maximization, negative for minimization
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_x1", random.uniform, 0, 100)  # Tourist count range
toolbox.register("attr_x2", random.uniform, 0, 50)   # Tax range
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_x1, toolbox.attr_x2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[0, 0], up=[100, 50], eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[0, 0], up=[100, 50], eta=20.0, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# Run NSGA-II algorithm
def main():
    pop_size = 100
    n_gen = 50
    cx_prob = 0.9
    mut_prob = 0.1

    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cx_prob, mut_prob, n_gen,
                                       stats=stats, halloffame=hof, verbose=True)

    return pop, logbook, hof

# Run optimization
pop, logbook, hof = main()

# Extract Pareto optimal solutions
pareto_front = np.array([ind.fitness.values for ind in hof])
pareto_solutions = np.array([ind for ind in hof])

# Convert Pareto optimal solutions to DataFrame
df = pd.DataFrame(pareto_front, columns=["f1", "f2", "f3", "f4"])

# Visualization
plt.figure(figsize=(18, 6))

# (1) Pareto Front Plot (2 objective functions)
plt.subplot(1, 2, 1)
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', label="Pareto Front")
plt.xlabel("f1: Tourism Revenue")
plt.ylabel("f2: Environmental Pressure")
plt.title("Pareto Optimal Solutions - Objective Space")
plt.legend()

# (2) Decision Space Plot
plt.subplot(1, 2, 2)
plt.scatter(pareto_solutions[:, 0], pareto_solutions[:, 1], c='blue', label="Pareto Solutions")
plt.xlabel("x1: Tourist Count")
plt.ylabel("x2: Tax")
plt.title("Pareto Optimal Solutions in Decision Space")
plt.legend()

plt.tight_layout()
plt.show()

# (3) Plotting Parallel Coordinates with Plotly
fig = px.parallel_coordinates(df, color="f1",  # Using f1 for color mapping
                              labels={"f1": "Tourism Revenue", "f2": "Environmental Pressure", "f3": "Resident Satisfaction", "f4": "Tourism Hidden Costs"},
                              title="Pareto Optimal Solutions - Parallel Coordinates Plot")
fig.show()
