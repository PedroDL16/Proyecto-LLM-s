import numpy as np
import pandas as pd
import time

def newton_method(f,df,x0,N=100,tol = 10**(-6)):
    '''
    Implementación del método de Newton para búsqueda de raíces.
    Parámetros:
    f : Función a encontrar raíces f(x).
    df : Función derivada f’(x).
    x0 : Aproximación inicial.
    N : Número máximo de iteraciones (default: 100).
    tol : Tolerancia de convergencia (default: 1e-6).
    Retorna:
    (x, itera, converged): Tupla con la aproximación final,
    el número de iteraciones usadas y un indicador de convergencia.
    '''
    #se crean las variables que guardan la informacion de las distintas iteraciones
    itera = [x0]
    x_i = x0
    f_i = f(x0)
    df_i = df(x0)
    converged = False
    
    #se verifican las entradas
    if ((type(N) != int) or (tol <= 0)):
        raise Exception('Error en las entradas. N debe ser entero y tol debe ser positivo')
    
    #parte iterativa del codigo
    #se usa ciclo While por compacidad de codigo
    while((np.abs(f_i) > tol) and (len(itera) < N)):
        if (df_i == 0):
            raise Exception('Error en el calculo, la derivada se anula')
        x_i = x_i - (f_i / df_i)
        itera.append(x_i)
        f_i = f(x_i)
        df_i = df(x_i)
    
    #se revisa la convergencia del metodo
    if ((np.abs(f_i) < tol)):
        converged = True
    
    #imprime una advertencia si se llego al maximo de iteraciones
    if (len(itera) == N):
        print('\n***Advertencia: El programa alcanzó el máximo número de iteraciones***')
    return (x_i,itera,converged)


    
def newton2d_numeric(F, x0, tol = 10**(-6), max_iter = 50, h = 10**(-6)):
    '''
    Implementacion metodo de Newton en 2d
    Parámetros:
    F: funcion a encontrar raices F(x,y)
    x0: Aproximacion inicial
    max_iter : Número máximo de iteraciones (default: 100).
    tol : Tolerancia de convergencia (default: 1e-6).
    Retorna:
    (x_i,F_i,itera,converged)
    x_i: Ultima aproximacion de la raíz
    F_i: Funcion evaluada en la ultima aproximacion
    itera: Iteraciones realizadas
    converged: Indicador de convergencia
    '''
    #se crean las variables que guardan la informacion de las distintas iteraciones
    itera = [x0]
    x_i = np.asarray(x0)
    F_i = np.asarray(F(x_i[0],x_i[1]))
    converged = False
    J = np.zeros((2,2))
    J[0][0] = ((F(x_i[0] + h, x_i[1])[0] - F(x_i[0],x_i[1])[0])/h)
    J[0][1] = ((F(x_i[0], x_i[1] + h)[0] - F(x_i[0],x_i[1])[0])/h)
    J[1][0] = ((F(x_i[0] + h, x_i[1])[1] - F(x_i[0],x_i[1])[1])/h)
    J[1][1] = ((F(x_i[0], x_i[1] + h)[1] - F(x_i[0],x_i[1])[1])/h)
    
    #se verifican las entradas
    if ((type(max_iter) != int) or (tol <= 0)):
        raise Exception('Error en las entradas. max_iter debe ser entero y tol debe ser positivo')
    
    #parte iterativa del codigo
    #se usa ciclo While por compacidad de codigo
    while((np.linalg.norm(F_i) >= tol) and (len(itera) < max_iter)):   
        #advertencia en caso de que se anule el determinante del jacobiano
        if (np.linalg.det(J) == 0):
            print(J)
            raise Exception('Error en el calculo, el determinante del jacobiano se anula')
        x_i = x_i - np.linalg.solve(J,F_i)
        itera.append(x_i)
        F_i = F(x_i[0],x_i[1])
        J[0][0] = ((F(x_i[0] + h, x_i[1])[0] - F(x_i[0],x_i[1])[0])/h)
        J[0][1] = ((F(x_i[0], x_i[1] + h)[0] - F(x_i[0],x_i[1])[0])/h)
        J[1][0] = ((F(x_i[0] + h, x_i[1])[1] - F(x_i[0],x_i[1])[1])/h)
        J[1][1] = ((F(x_i[0], x_i[1] + h)[1] - F(x_i[0],x_i[1])[1])/h)
    
    #se verifica la convergencia
    norma = np.linalg.norm(F_i)
    if (norma < tol):
        converged = True
    
    #imprime una advertencia si se llego al maximo de iteraciones
    if (len(itera) == max_iter):
        print('\n***Advertencia: El programa alcanzó el máximo número de iteraciones***')
    return (x_i,F_i,itera,converged)

def newton2d_analytic(F, J, x0, tol = 10**(-6), max_iter = 50):
    '''
    Implementacion metodo de Newton en 2d
    Parámetros:
    F: funcion a encontrar raices F(x,y)
    x0: Aproximacion inicial
    max_iter : Número máximo de iteraciones (default: 100).
    tol : Tolerancia de convergencia (default: 1e-6).
    Retorna:
    (x_i,F_i,itera,converged)
    x_i: Ultima aproximacion de la raíz
    F_i: Funcion evaluada en la ultima aproximacion
    itera: Iteraciones realizadas
    converged: Indicador de convergencia
    '''
    #se crean las variables que guardan la informacion de las distintas iteraciones
    itera = [x0]
    x_i = np.asarray(x0)
    F_i = np.asarray(F(x_i[0],x_i[1]))
    converged = False
    J_i = J(x_i[0],x_i[1])


    #se verifican las entradas
    if ((type(max_iter) != int) or (tol <= 0)):
        raise Exception('Error en las entradas. max_iter debe ser entero y tol debe ser positivo')
    
    
    #parte iterativa del codigo
    #se usa ciclo While por compacidad de codigo
    while((np.linalg.norm(F_i) >= tol) and (len(itera) < max_iter)):        
        #advertencia en caso de que se anule el determinante del jacobiano
        if (np.linalg.det(J_i) == 0):
            print(J_i)
            raise Exception('Error en el calculo, el determinante del jacobiano se anula')
        x_i = x_i - np.linalg.solve(J_i,F_i)
        itera.append(x_i)
        F_i = F(x_i[0],x_i[1])
        J_i = J(x_i[0],x_i[1])
    
    #se verifica la convergencia
    norma = np.linalg.norm(F_i)
    if (norma < tol):
        converged = True
    
    #imprime una advertencia si se llego al maximo de iteraciones
    if (len(itera) == max_iter):
        print('\n***Advertencia: El programa alcanzó el máximo número de iteraciones***')
    return (x_i,F_i,itera,converged)


if __name__ == '__main__':

    '''
    parte del codigo para newton 1d
    '''
    def f(x):
        return x**2 -2    
    def df(x):
        return 2*x
    
    x0 = 0.2
    resultado1d = newton_method(f,df,x0)
    print(f'Raíz aproximada: {resultado1d[0]} \nIteraciones: {len(resultado1d[1])} \nConvergencia: {resultado1d[2]}')
    print('##############################################################################################')
    
    
    '''
    Parte del codigo para newton 2d
    '''
    def F(x,y):
        return [x**2 + y**2 - 1,np.exp(x) - y]

    def J(x,y):
        return np.asarray([[2*x,2*y],[np.exp(x),-1]])
    
    x0 = 0.5
    y0 = 0.5
    #'----------------Númerico-------------------------'
    inicio1 = time.time_ns()
    resultado1 = newton2d_numeric(F,[x0,y0],max_iter = 50)
    ejecucion1 = time.time_ns() - inicio1
        
    
    #'----------------Analitíco------------------------'
    inicio2 = time.time_ns()
    resultado2 = newton2d_analytic(F,J,[x0,y0])
    ejecucion2 = time.time_ns() - inicio2

    resultados = pd.DataFrame({'w*':[resultado1[0],resultado2[0]],'F(x*)':[resultado1[1],resultado2[1]],'#':[resultado1[2],resultado2[2]],'Tiempo':[ejecucion1,ejecucion2]}, index = ['Númerico','Analítico'])
    print(resultados)