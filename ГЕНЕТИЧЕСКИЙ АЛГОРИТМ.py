from deap import base, algorithms
from deap import creator
from deap import tools

import random
import matplotlib.pyplot as plt
import numpy as np

import gym

"""Создание окружения для машинки"""
# env = gym.make('MountainCar-v0', render_mode="rgb_array")

""" Основные константы
    нужны для генетического алгоритма """
ChromLength = 200       # Длина хромосомы
PopulationSize = 500     # количество агентов в популяции
PCrossover = 0.9        # вроятность скрещивания агентов
PMutation = 0.2         # верояность мутации агента
MaxGenerations = 100     # максимальное колчество поколений
HallOfFameSize = 3      # Эталонные агенты, переходящие в другие поколения, зал славы

HallWinner = tools.HallOfFame(HallOfFameSize)  # создаю зал славы

RandomSpeed = 42
random.seed(RandomSpeed)

# --------------------------------------------------------------------------------------------------------------------
"""Создание особей с хоромосомами и популяции, куда складываются особи"""

"""class FitnessMax():
    def __init__(self):
        self.values = [0]   # приспособленность особи 0"""

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Расчет приспособленности каждого отдельного индивида
def TaskFitness(individual):
    return sum(individual), # Возвращает кортеж

"""class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()"""

toolbox = base.Toolbox()
toolbox.register("ZeroOrOne", random.randint, 0, 1)
toolbox.register("IndividualCreator", tools.initRepeat, creator.Individual, toolbox.ZeroOrOne, ChromLength)
toolbox.register("PopulationCreator", tools.initRepeat, list, toolbox.IndividualCreator)

population = toolbox.PopulationCreator(n=PopulationSize)

# --------------------------------------------------------------------------------------------------------------------

"""Реализация генетического алгоритма"""

generationCounter = 0

fitnessValues = list(map(TaskFitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

maxFitnessValues = []
meanFitnessValues = []


toolbox.register("evaluate", TaskFitness)   # Возвращает кортеж значений приспособленностей каждого отдельного индиаида
"Турнирный отбор. 3 особи учавствуют"
toolbox.register("select", tools.selTournament, tournsize=3)
"Одноточечное скрещивание родителей"
toolbox.register("mate", tools.cxOnePoint)
"Мутайия (интвертирует 1 бит)"
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ChromLength)     # indpb - вероятность мутации гена в хромосоме особи


stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

population, logbook = algorithms.eaSimple(population, toolbox,
                                          cxpb=PCrossover,
                                          mutpb=PMutation,
                                          ngen=MaxGenerations,
                                          stats=stats,
                                          verbose=True)


maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

plt.plot(maxFitnessValues, color="red")
plt.plot(meanFitnessValues, color="green")
plt.xlabel("Поколение")
plt.ylabel("Макс/средняя приспособленность")
plt.title("Зависимость максимальной и срдней приспособленности от поколения")
plt.show()

"""observation = env.reset()

env.render()"""