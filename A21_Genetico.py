#================================
# Algortimo genético simple 
#============================
# JOSÉ JULIO LOPEZ MARQUEZ 
# FUNDAMENTOS DE IA
# ESFM IPN JUNIO 2025
#==========================================
import datetime
import random

random.seed(random.random())
startTime = datetime.datetime.now()
#=========================
# Los genes
#====================
geneSet = "abcdefghikjlmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

#=====================
# Objetivo
#=================
target = "hola mundo"

#==============================
# Fase inicial
#=========================
def generate_parent(length):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    return ''.join(genes)

#===============================
# Función de aptitud 
#============================
def get_fitness(guess):
    return sum( 1 for expected, actual in zip(target,guess) if expected == actual)

#=======================================
# Mutación de letras en la frase
#===================================
def display (guess):
    timeDiff = datatime.datetime.now() - startTime
    fitness = get_fitness(guess)
    print("{}\t{}\{}".format(guess,fitness,timeDiff))
#=============================
# Código principal
#=====================
bestParent = generate_parent(len(target))
bestFitness = get_fitness(bestParent)
display(bestParent)

#======================
# Iteraciones 
#=========================0
while True:
    child = mutate(bestParent)
    childFitness = get_fitness(child)
    if bestFitness >= childFitness:
       dispaly(child)
       continue
    display(child)
    if childFitness >= len(bestParent):
        break
    bestFitness = childFitness
    bestParent = child