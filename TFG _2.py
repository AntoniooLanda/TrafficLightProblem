import numpy as np
import time
import random
import sys

def simulation(total_time, total_intersections, cars, schedule, intersections, streets, extra_points):
    score = 0
    
    for car in cars:
        first_street = car.pop()
        intersections[streets[first_street][1]][first_street].append([0, car, len(car)])
        
    for i in range(total_time):
        for j in range(total_intersections):
            if (len(schedule[j]) == 0):
                continue
            street_name = schedule[j].pop(0)
            schedule[j].append(street_name)
            if len(intersections[j][street_name]) != 0:
                car = intersections[j][street_name][0]
                if car[0] <= i:
                    intersections[j][street_name].pop(0)
                    car[2] -= 1
                    next_street = car[1][car[2]]
                    if car[2] == 0:
                        if streets[street_name][2] + i < total_time:
                            score += extra_points + total_time - i - streets[street_name][2]
                    else:
                        car[0] = i + streets[street_name][2]
                        intersections[streets[next_street][1]][next_street].append(car)
    return score

def better_schedule(cars, streets, total_intersections,n):
    better = [[],[]]
    aux = []
    for i in range(total_intersections):
        aux.append({})
        better[0].append([])
        better[1].append([])
    for car in cars:
        for street in car:
            if aux[streets[street][1]].get(street) == None:
                aux[streets[street][1]][street] = [street, 1]
            else:
                aux[streets[street][1]][street][1] += 1
    for i in range(total_intersections):
        intersection = []
        for street in aux[i].values():
            intersection.append(street)
        random.shuffle(intersection)
        for street in intersection:
            better[0][i].append(street[0])
            better[1][i].append(street[1])
            
    random_number = n * random.random()
    
    for i in range(total_intersections):
        suma = sum(better[1][i])
        
        for j in range(len(better[1][i])):
            #better[1][i][j] = 1 +  (random.randrange(1,3)*better[1][i][j]) // suma
            better[1][i][j] = int(1 +  (random_number *better[1][i][j]) // suma)    

    return better    

#Necesito una lista de tamaño mil en que cada casilla contenga las calles que usan los coches
def optimize_schedule(cars, streets, total_intersections):
    lista1 = []
    lista2 = []
    
    for i in range(total_intersections):
        lista1.append({})
        lista2.append([])
        
    for car in cars:
        for street in car:
            lista1[streets[street][1]][street] = street
    for i in range(total_intersections):
        for street in lista1[i].values():
            lista2[i].append(street)

    return lista2



#Este metodo pasaria un array con un array doble dentro a un array de arrays
def transformador(lista_calles):
    lista = []
    for i in lista_calles[1]:
        lista.append([])
        
    for i in range(len(lista_calles[1])):
        for j in range(len(lista_calles[1][i])):
            for repeticiones in range(lista_calles[1][i][j]):
                lista[i].append(lista_calles[0][i][j])

    return lista

def cruzar(lista1, lista2):
    hijo1 = []
    hijo2 = []
    n = random.randint(0, (len(lista1) * 3) //10)

    for j in range(n):
        hijo1.append(lista1[j])
        hijo2.append(lista2[j])
    for j in range(n, n + (len(lista1) * 7) // 10):
        hijo1.append(lista2[j])
        hijo2.append(lista1[j])
    for j in range(n + (len(lista1) * 7) // 10, len(lista1)):
        hijo1.append(lista1[j])
        hijo2.append(lista2[j])
    return hijo1,hijo2

#Metodo para cruzar codificados de permutación.
#Inputs: dos listas dobles que contiene cada una una lista de con los ciclos de las calles y otras sus tiempos. [ [['a','b'],['c','d']] , [[1,2],[4,2]] ]
#Output: 2 listas con las permutaciones y dos diccionarios con los tiempos del primer y segundo padre
def crucePermutacion(lista1, lista2):
    hijo1 = []
    hijo2 = []
    tiempos1 = {}
    tiempos2 = {}
    for i in range(len(lista1[0])):        
        puntoCruce = random.randint(0, (len(lista1[0][i])))
        permutaciones1 = []
        permutaciones2 = []
                                            
        for j in range(puntoCruce):
            permutaciones1.append(lista1[0][i][j])
            permutaciones2.append(lista2[0][i][j])
            tiempos1[lista1[0][i][j]] = lista1[1][i][j]
            tiempos2[lista2[0][i][j]] = lista2[1][i][j]
            
                                    
        for j in range(puntoCruce,(len(lista1[0][i]))):
            if lista2[0][i][j] in permutaciones1:
                for k in range(len(lista2[0][i])):
                    if lista2[0][i][k] not in permutaciones1:
                        permutaciones1.append(lista2[0][i][k])
                        
            else:
                permutaciones1.append(lista2[0][i][j])
                
                
            if lista1[0][i][j] in permutaciones2:
                for k in range(len(lista1[0][i])):
                    if lista1[0][i][k] not in permutaciones2:
                        permutaciones2.append(lista1[0][i][k])
                        
            else:
                permutaciones2.append(lista1[0][i][j])
                
            tiempos1[lista1[0][i][j]] = lista1[1][i][j]
            tiempos2[lista2[0][i][j]] = lista2[1][i][j]
                
        hijo1.append(permutaciones1)
        hijo2.append(permutaciones2)

    return hijo1,hijo2,tiempos1,tiempos2        
            
#Metodo de cruce en un punto, para cada intersección, se escoge un punto de corte k, los primeros k tiempos son del
#primer padre y el resto del segundo
#Input: dos listas con los ordenes y dos diccionarios con los tiempos de los dos padres.
#Output: array doble con los ordenes y los tiempos de los semaforos
def crucePunto(ordenes1,ordenes2,tiempos1,tiempos2):
    hijo1 = [[],[]]
    hijo2 = [[],[]]
    for i in range(len(ordenes1)):
        auxTiempos1 = []
        auxTiempos2 = []
        puntoCruce = random.randint(0,len(ordenes1[i]))
        for j in range(puntoCruce):
            auxTiempos1.append(tiempos1[ordenes1[i][j]])
            auxTiempos2.append(tiempos2[ordenes2[i][j]])
        for j in range(puntoCruce,len(ordenes1[i])):
            auxTiempos1.append(tiempos2[ordenes2[i][j]])
            auxTiempos2.append(tiempos1[ordenes1[i][j]])
        hijo1[0].append(ordenes1[i])
        hijo1[1].append(auxTiempos1)
        hijo2[0].append(ordenes2[i])
        hijo2[1].append(auxTiempos2)
    return hijo1,hijo2

#Metodo de cruce uniforme, para cada tiempo se esocge aletariamente entre los dos padres
#primer padre y el resto del segundo
#Input: dos listas con los ordenes y dos diccionarios con los tiempos de los dos padres.
#Output: array doble con los ordenes y los tiempos de los semaforos
def cruceUniforme(ordenes1,ordenes2,tiempos1,tiempos2):
    hijo1 = [[],[]]
    hijo2 = [[],[]]
    for i in range(len(ordenes1)):
        auxTiempos1 = []
        auxTiempos2 = []
        for j in range(len(ordenes1[i])):
            flag = random.randint(0,1)
            if flag == 0:
                auxTiempos1.append(tiempos1[ordenes1[i][j]])
                auxTiempos2.append(tiempos2[ordenes2[i][j]])
            else:
                auxTiempos1.append(tiempos2[ordenes1[i][j]])
                auxTiempos2.append(tiempos1[ordenes2[i][j]])
    
        hijo1[0].append(ordenes1[i])
        hijo1[1].append(auxTiempos1)
        hijo2[0].append(ordenes2[i])
        hijo2[1].append(auxTiempos2)
    return hijo1,hijo2

#Metodo de cruce promedio, para cada tiempo se hace el promedio entre los dos padres
#Input: dos listas con los ordenes y dos diccionarios con los tiempos de los dos padres.
#Output: array doble con los ordenes y los tiempos de los semaforos
def crucePromedio(ordenes1,tiempos1,tiempos2):
    hijo1 = [[],[]]
    for i in range(len(ordenes1)):
        auxTiempos1 = []
        for j in range(len(ordenes1[i])):
            if (tiempos1[ordenes1[i][j]]+tiempos2[ordenes1[i][j]])%2 == 1:
                auxTiempos1.append((tiempos1[ordenes1[i][j]]+tiempos2[ordenes1[i][j]])//2  + random.randint(0,1))
                
            else:
                auxTiempos1.append((tiempos1[ordenes1[i][j]]+tiempos2[ordenes1[i][j]])//2)
    
        hijo1[0].append(ordenes1[i])
        hijo1[1].append(auxTiempos1)
    
    return hijo1



def max_elements(lista, num):
    lista_copy = lista.copy()
    positions = []
    
    for j in range(num):
        a = max(lista_copy)
        encontrado = True
        i = 0
        while(encontrado):
            if lista_copy[i] == a:
                positions.append(i)
                lista_copy[i] = 0
                encontrado = False
            i += 1
                
    return positions

def compute_sel_prob(population_fitness):

  n = len(population_fitness)

  rank_sum = n * (n + 1) / 2

  for rank, ind_fitness in enumerate(sorted(population_fitness), 1):
    yield rank, ind_fitness, float(rank) / rank_sum



def selectionTournament(lista):
    winners = []
    parejas = []
    for i in range(len(lista)):
        parejas.append(i)

    random.shuffle(parejas)
    for i in range(len(lista)//2):

        if lista[parejas[i]] >= lista[parejas[i+ len(lista)//2]]:
            winners.append(parejas[i])
            
        else:
            winners.append(parejas[i+ len(lista)//2])
            
    return winners

def linearSelection(scores):
    probs= []  

    for rank, ind_fitness, sel_prob in compute_sel_prob(scores):
        probs.append(sel_prob)
    return random.choices(np.argsort(scores), weights=probs, k=len(scores)//2)

def mutation(hijo, total_intersections):
    number1 = random.randint(0, total_intersections)
    number2 = random.randint(0, len(hijo[1][number1]))
    if hijo[1][number1][number2] == 1:
        hijo[1][number1][number2] = 2
    else:
        hijo[1][number1][number2] = 1
    
def basic_schedule(streets,intersections):
    basic_schedule = [[] for i in range(intersections)]
    for key in streets:
        basic_schedule[streets[key][1]].append(key)

    for intersection in basic_schedule:
        random.shuffle(intersection)
    
    return basic_schedule
    
    

def times_schedule(lista_basica,n):
    lista = [[],[]]
    for i in lista_basica:
        lista[1].append([])
        
    for i in range(len(lista_basica)):
        lista[0].append(lista_basica[i])
        for j in range(len(lista_basica[i])):
            lista[1][i].append(random.randint(1,n))
    return lista

def shuffle_list(lista):
    new_list = []
    for schedule in lista:
        new_list.append(schedule.copy())
    for schedule in new_list:
        random.shuffle(schedule)
    return new_list

def genetic_algorithm(list_name, weight, initial_population, num_generations, mutation_number,metodoCruce, metodoSeleccion, dijkstra):
    data = []
    streets = {}
    intersections = []
    schedule = []
    cars = []

    with open(list_name) as f:
        
        data = f.readline().split()[::-1]

        #Obtenemos los datos iniciales
        total_time = int(data.pop())
        total_intersections = int(data.pop())
        total_streets = int(data.pop())
        total_cars = int(data.pop())
        extra_points = int(data.pop())


        #Guardamos las calles en un mapa: llave: el nombre de la calle
        #                                 valor: array[interseccion inicial, final, tiempo]
        #Lista de intersecciones: cada interseccion tiene un mapa con llave el nombre de la
        #                       calle y de valor una lista (que serán las rutas de los coches)

        #Lista de calles que entran en cada calle
        for i in range(total_intersections):
            intersections.append({})
        for i in range(total_streets):
            data = f.readline().split()[::-1]
            initial_intersection = int(data.pop())
            final_intersection = int(data.pop())
            street_name = data.pop()
            street_duration = int(data.pop())
            
            streets[street_name] = [initial_intersection, final_intersection, street_duration, street_name]
            intersections[final_intersection][street_name] = []

        
        #Ahora metemos las rutas de los coches en la interseccion que toca
        #Esta formada por un array de tamaño dos, el primero con un int que nos ayudara
        #a saber si esta en mitad de una calle y por una cola que es la ruta del coche
        start_time = time.time()
        if dijkstra == 1:
            carsAux = []
            for i in range(total_cars):
                data = f.readline().split()[::-1]
                data.pop()
                carsAux.append(data)
                
            g = [[] for i in range(total_intersections+1)]
            for key in streets:
                g[streets[key][0]].append(key)

            for i in range(total_cars):
                
                path = dijkstraDist(g, int(streets[carsAux[i][-1]][0]),int(streets[carsAux[i][0]][0]),streets)
                if len(path) != 1:
                    cars.append(path)
                
##            for i in range(total_cars):
##                data = f.readline().split()[::-1]
##                path = dijkstraDist(g, int(data[1]),int(data[0]),streets)
##                cars.append(path)
        else:
            for i in range(total_cars):
                data = f.readline().split()[::-1]
                data.pop()
                cars.append(data)

    print("--- %s seconds ---" % (time.time() - start_time))
    random.seed(11)
    #Generamos la poblacion inicial
    schedules = []
    #basicSchedule = optimize_schedule(cars, streets, total_intersections)
    minTimeCars = 0
    for car in cars:
        for street in car:
            minTimeCars += streets[street][2]
    puntuacionMax = (total_time + extra_points) * len(cars) - minTimeCars
    print(puntuacionMax)

    for i in range(initial_population):
        basicSchedule = basic_schedule(streets, total_intersections)
        schedule = times_schedule(shuffle_list(basicSchedule),weight)
        #schedule = better_schedule(cars, streets, total_intersections, weight)
        schedules.append(schedule)
    
    
    shedule_sort = []
    for i in range(len(schedule[0])):
        aux = [schedule[0][i],schedule[1][i]]
        

    
    scores = []
    print(schedules)
    for i in range(initial_population):
        cars_copy = []
        intersections_copy = []
        for intersection in intersections:
            dic = {}
            for street in intersection:
                dic[street] = intersection[street].copy()
            intersections_copy.append(dic)
        for car in cars:
            cars_copy.append(car.copy())

        scores.append(simulation(total_time, total_intersections, cars_copy, transformador(schedules[i]), intersections_copy, streets, extra_points))
    print(scores)
    print(max(scores))

    for i in range(num_generations):
        aux = []
        #best = max_elements(scores, initial_population // 2)
        if metodoSeleccion == 'Torneo de seleccion':
            best = selectionTournament(scores)
        elif metodoSeleccion == 'Ranking lienal':
            best = linearSelection(scores)
        else:
            sys.exit('NOMBRE DE SELECCION NO ENCONTRADO')
        

        for n in range(-1,(initial_population // 2) - 1):
            print(best)
            hijo1,hijo2, tiempos1, tiempos2 = crucePermutacion(schedules[best[n]], schedules[best[n + 1]])
            #CRUCE DE LOS PADRES
            if(metodoCruce == 'Cruce en un Punto'):
                hijo1,hijo2 = crucePunto(hijo1,hijo2, tiempos1, tiempos2)
                
            elif(metodoCruce == 'Cruce Uniforme'):
                hijo1,hijo2 = cruceUniforme(hijo1,hijo2, tiempos1, tiempos2)

            elif(metodoCruce == 'Cruce Promedio'):
                hijo1 = crucePromedio(hijo1,tiempos1, tiempos2)
                hijo2 = schedules[best[n]]

            else:
                sys.exit('NOMBRE DE CRUCE NO ENCONTRADO')
                
            #if random.random() < mutation_number:
            if(False):

                mutation(hijo1, total_intersections)
                print('SE HA PRODUCIDO UNA MUTACION!')

            #if random.random() < mutation_number:
            if(False):
                mutation(hijo2, total_intersections)
                print('SE HA PRODUCIDO UNA MUTACION!')

            aux.append(hijo1)
            aux.append(hijo2)
        scores = []
        schedules = []
        for schedule in aux:
            schedules.append(schedule)

        for j in range(initial_population):
            intersections_copy = []
            for intersection in intersections:
                dic = {}
                for street in intersection:
                    dic[street] = intersection[street].copy()
                intersections_copy.append(dic)
            cars_copy = []
            for car in cars:
                cars_copy.append(car.copy())
            scores.append(simulation(total_time, total_intersections, cars_copy, transformador(schedules[j]), intersections_copy, streets, extra_points))
            #print(schedules[j])   
        print(scores)
        print(max(scores))    

    return schedules[scores.index(max(scores))]

#g = [['A','C','D'],['B'],['E'],[]]
#s = start node 
def dijkstraDist(g, s,final,streets):
    dist = [float('inf') for i in range(len(g))]
    visited = [False for i in range(len(g))]
    rutas = [[] for i in range(len(g))]
    dist[s] = 0
    current = s

    sett = set()

    while (not visited[final]):

        visited[current] = True
        for i in range(len(g[current])):
            v = streets[g[current][i]][1]
            if (visited[v]):
                continue
            sett.add(v)
            alt = dist[current] + streets[g[current][i]][2]
            
            if (alt < dist[v]):
                dist[v] = alt

                rutas[v] = rutas[current].copy()
                rutas[v].append(g[current][i])

        if current in sett:
            sett.remove(current)
        if (len(sett) == 0):
            break

        minDist = float('inf')
        index = 0
        for a in sett:
            if (dist[a] < minDist):
                minDist = dist[a]
                index = a
        current = index
    return rutas[final]



start_time = time.time()
#genetic_algorithm(list_name, weight, initial_population, num_generations, mutation)
#best_schedule = genetic_algorithm("hashcode.in", 2,16, 8, 0.05,'Cruce Promedio','Ranking lienal',1)
#best_schedule = genetic_algorithm("hashcode.in", 3, 6, 3, 0.01)
#4022552
#Hemos tardado:  4724.36921620369 segundos
#Con 10 y 2: 4017564
# Con 10 y 2 con el metodo supuestamente mejor: 4014124 en 1212 segundos
# Con 10 y 2, peso 3 y mutacion 0.05: 4015537
# Con 20 y 3, peso 3 y mutacion 0.05: 4017132
# Cruce en un Punto ,  Cruce Uniforme  , Cruce Promedio
#best_schedule = genetic_algorithm("CallesTFG.in", 3, 8, 3, 0.00001,'Cruce Uniforme',1)
#best_schedule = genetic_algorithm("PruebasTest.in", 5, 14, 6, 0.00001,'Cruce Uniforme',0)
#best_schedule = genetic_algorithm("reducido.in", 3, 10, 3, 0.1)
##
##print('Cruce Uniforme')
##best_schedule = genetic_algorithm("hashcode.in", 2,16, 8, 0.05,'Cruce Uniforme',0)
##print('---------------------------------------------------')
##print('Cruce en un Punto')
##best_schedule = genetic_algorithm("hashcode.in", 2,16, 8, 0.05,'Cruce en un Punto',0)
##print('---------------------------------------------------')
##print('Cruce Promedio')
##best_schedule = genetic_algorithm("hashcode.in", 2,16, 8, 0.05,'Cruce Promedio',0)
##
##
##print('Cruce Uniforme')
##best_schedule = genetic_algorithm("hashcode.in", 2,16, 8, 0.05,'Cruce Uniforme','Ranking lienal',0)
##print('---------------------------------------------------')
##print('Cruce en un Punto')
##best_schedule = genetic_algorithm("hashcode.in", 2,16, 8, 0.05,'Cruce en un Punto','Ranking lienal',0)
##print('---------------------------------------------------')
##print('Cruce Promedio')
##best_schedule = genetic_algorithm("hashcode.in", 2,16, 8, 0.05,'Cruce Promedio','Ranking lienal',0)
##
##
best_schedule = genetic_algorithm("EjemploTFG.in", 4,4,3, 0.05,'Cruce en un Punto','Ranking lienal',0)
