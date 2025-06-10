from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Método de Gauss-Jordan
def gauss_jordan(a, b):
    # Verificar si la matriz es invertible
    if np.linalg.det(a) == 0:
        return [float('nan'), float('nan')]  # Devolver NaN si la matriz no es invertible

    n = len(b)
    augmented_matrix = np.hstack([a, b.reshape(-1, 1)])
    
    for i in range(n):
        # Buscar el máximo en la columna
        max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

        for j in range(i + 1, n):
            ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j] -= ratio * augmented_matrix[i]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i][-1] - sum(augmented_matrix[i][j] * x[j] for j in range(i + 1, n))) / augmented_matrix[i][i]
    
    return x

# Método de Eliminación Gaussiana
def gauss_elimination(a, b):
    # Verificar si la matriz es invertible
    if np.linalg.det(a) == 0:
        return [float('nan'), float('nan')]  # Devolver NaN si la matriz no es invertible

    n = len(b)
    augmented_matrix = np.hstack([a, b.reshape(-1, 1)])

    for i in range(n):
        # Buscar el máximo en la columna
        max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

        for j in range(i + 1, n):
            ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j] -= ratio * augmented_matrix[i]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i][-1] - sum(augmented_matrix[i][j] * x[j] for j in range(i + 1, n))) / augmented_matrix[i][i]
    
    return x

# Método de la Matriz Inversa
def inverse_matrix_method(a, b):
    # Verificar si la matriz es invertible
    if np.linalg.det(a) == 0:
        return [float('nan'), float('nan')]  # Devolver NaN si la matriz no es invertible
    
    a_inv = np.linalg.inv(a)  # Inversa de la matriz A
    return np.dot(a_inv, b)

# Método de Cramer
def cramer_method(a, b):
    # Verificar si la matriz es invertible
    det_a = np.linalg.det(a)
    if det_a == 0:
        return [float('nan'), float('nan')]  # Devolver NaN si la matriz no es invertible

    n = len(b)
    solutions = []
    for i in range(n):
        a_copy = a.copy()
        a_copy[:, i] = b  # Sustituir la columna i de A con el vector b
        det_ai = np.linalg.det(a_copy)
        solutions.append(det_ai / det_a)  # Solución para X o Y
    return np.array(solutions)

# Explicaciones para cada caso
def case_optimization_explanation():
    return (
        "Este caso se refiere a la optimización en la industria manufacturera. Los coeficientes de la matriz de "
        "ecuaciones representan las relaciones entre los recursos utilizados en cada parte del proceso de producción. "
        "Por ejemplo, si se producen dos productos diferentes, los coeficientes indican la cantidad de recursos necesarios "
        "para fabricar cada producto. Las raíces representan la cantidad de cada producto que se debe producir para "
        "maximizar la producción mientras se respetan las restricciones de recursos."
    )

def case_population_dynamics_explanation():
    return (
        "Este caso modela el crecimiento de dos especies (por ejemplo, presas y depredadores). Los coeficientes de las "
        "ecuaciones representan las tasas de crecimiento de las especies y las interacciones entre ellas. Las raíces de "
        "las ecuaciones representan las poblaciones equilibradas de las especies, es decir, el número de individuos en "
        "cada especie cuando el sistema está en equilibrio."
    )

def case_traffic_modeling_explanation():
    return (
        "Este caso modela el tráfico en una red de carreteras. Los coeficientes representan el volumen de tráfico en "
        "diferentes rutas o intersecciones. Las raíces de las ecuaciones representan la distribución del tráfico entre las "
        "rutas, lo que ayuda a optimizar el flujo vehicular y evitar congestiones."
    )

# Ruta para la página principal (inicio)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Rutas para cada caso
@app.route("/optimization", methods=["GET", "POST"])
def optimization_case():
    if request.method == "POST":
        a1 = float(request.form["a1"])
        a2 = float(request.form["a2"])
        a3 = float(request.form["a3"])
        b1 = float(request.form["b1"])
        b2 = float(request.form["b2"])
        
        A = np.array([[a1, a2], [a3, b1]])
        B = np.array([b2, b2])
        
        # Calcular las soluciones
        gauss_jordan_solution = gauss_jordan(A, B)
        gauss_elimination_solution = gauss_elimination(A, B)
        inverse_matrix_solution = inverse_matrix_method(A, B)
        cramer_solution = cramer_method(A, B)

        return render_template("optimization_case.html", 
                               gauss_jordan=gauss_jordan_solution.tolist(), 
                               gauss_elimination=gauss_elimination_solution.tolist(), 
                               inverse_matrix=inverse_matrix_solution.tolist(), 
                               cramer=cramer_solution.tolist(),
                               explanation=case_optimization_explanation())

    return render_template("optimization_case.html", explanation=case_optimization_explanation())

@app.route("/population", methods=["GET", "POST"])
def population_case():
    if request.method == "POST":
        a1 = float(request.form["a1"])
        a2 = float(request.form["a2"])
        a3 = float(request.form["a3"])
        b1 = float(request.form["b1"])
        b2 = float(request.form["b2"])
        
        A = np.array([[a1, a2], [a3, b1]])
        B = np.array([b2, b2])
        
        # Calcular las soluciones
        gauss_jordan_solution = gauss_jordan(A, B)
        gauss_elimination_solution = gauss_elimination(A, B)
        inverse_matrix_solution = inverse_matrix_method(A, B)
        cramer_solution = cramer_method(A, B)

        return render_template("population_case.html", 
                               gauss_jordan=gauss_jordan_solution.tolist(), 
                               gauss_elimination=gauss_elimination_solution.tolist(), 
                               inverse_matrix=inverse_matrix_solution.tolist(), 
                               cramer=cramer_solution.tolist(),
                               explanation=case_population_dynamics_explanation())

    return render_template("population_case.html", explanation=case_population_dynamics_explanation())

@app.route("/traffic", methods=["GET", "POST"])
def traffic_case():
    if request.method == "POST":
        a1 = float(request.form["a1"])
        a2 = float(request.form["a2"])
        a3 = float(request.form["a3"])
        b1 = float(request.form["b1"])
        b2 = float(request.form["b2"])
        
        A = np.array([[a1, a2], [a3, b1]])
        B = np.array([b2, b2])
        
        # Calcular las soluciones
        gauss_jordan_solution = gauss_jordan(A, B)
        gauss_elimination_solution = gauss_elimination(A, B)
        inverse_matrix_solution = inverse_matrix_method(A, B)
        cramer_solution = cramer_method(A, B)

        return render_template("traffic_case.html", 
                               gauss_jordan=gauss_jordan_solution.tolist(), 
                               gauss_elimination=gauss_elimination_solution.tolist(), 
                               inverse_matrix=inverse_matrix_solution.tolist(), 
                               cramer=cramer_solution.tolist(),
                               explanation=case_traffic_modeling_explanation())

    return render_template("traffic_case.html", explanation=case_traffic_modeling_explanation())

if __name__ == "__main__":
    app.run(debug=True)
