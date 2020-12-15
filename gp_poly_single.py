import operator
import std_functions
import random
from random import randint
from deap import algorithms
import math
from deap import base, creator, gp, tools
from deap.gp import PrimitiveTree
import datetime


def if_then_else(input, output1, output2):
    return output1 if input else output2


def larger_than(input1, input2):
    return input1 > input2


def smaller_than(input1, input2):
    return True if input1 < input2 else False


def larger_equal_than(input1, input2):
    return True if input1 >= input2 else False


def smaller_equal_than(input1, input2):
    return True if input1 <= input2 else False


def equal_to(input1, input2):
    return True if input1 == input2 else False


def equal(input1, input2):
    return input2


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def power(input1, input2):
    return input1 ** input2


pset = gp.PrimitiveSetTyped("main", [float], float)



pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
#pset.addPrimitive(power, [float, float], float)
#pset.addPrimitive(equal, [float, float], float)

pset.addTerminal(0.0, float)
pset.addTerminal(1.0, float)
pset.addTerminal(2.0, float)
pset.addTerminal(3.0, float)
pset.addTerminal(4.0, float)



pset.renameArguments(ARG0="x")


POPULATION = 1000

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    function = gp.compile(individual, pset)
    tot_fitness = []
    for i in range(10):
        x = randint(-100, 100)
        result = function(x)
        std_result = std_functions.poly_single(x)
        fitness = abs(result - std_result)
        tot_fitness.append(fitness)
    return math.fsum(tot_fitness),


toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

pop = toolbox.population(n=POPULATION)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 400,
                               halloffame=hof, verbose=True)

my_result = hof[0]
print(my_result)
print(evaluate(my_result))


# result evaluation

function = gp.compile(my_result, pset)

any_diff = 0
out_results = []
std_results = []
runtime1 = 0
runtime2 = 0
for i in range(10000):
    x = randint(-1000, 1000)
    # test case
    starttime = datetime.datetime.now()
    result = function(x)
    endtime = datetime.datetime.now()
    runtime1 += (endtime - starttime).microseconds
    # std case
    starttime = datetime.datetime.now()
    std_result = std_functions.poly_single(x)
    endtime = datetime.datetime.now()
    runtime2 += (endtime - starttime).microseconds
    out_results.append(result)
    std_results.append(std_result)


for i in range(len(out_results)):
    if out_results[i] != std_results[i]:
        #print(out_results[i])
        #print(std_results[i])
        #print('='*8)
        any_diff = 1

if any_diff == 0:
    print("evaluation passed")
else:
    print("evaluation not passed")

print("New function cost {} microseconds\nStd func cost {} microseconds".format(runtime1, runtime2))


nodes, edges, labels = gp.graph(my_result)

import matplotlib.pyplot as plt
import networkx as nx

g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")

nx.draw_networkx_nodes(g, pos)
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos, labels)
plt.show()
