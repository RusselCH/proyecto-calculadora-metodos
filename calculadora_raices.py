import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Calculadora Visual de Raíces", layout="wide")

# ==================== MÉTODOS NUMÉRICOS ====================

def biseccion(f, a, b, tol=1e-6, max_iter=100):
    iteraciones = []
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return [], None, "El signo de f(a) y f(b) debe ser diferente"
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a) / 2
        iteraciones.append([i+1, a, b, c, fc, error])
        if error < tol or fc == 0:
            return iteraciones, c, None
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return iteraciones, c, None

def regla_falsa(f, a, b, tol=1e-6, max_iter=100):
    iteraciones = []
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return [], None, "El signo de f(a) y f(b) debe ser diferente"
    for i in range(max_iter):
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        error = abs(fc)
        iteraciones.append([i+1, a, b, c, fc, error])
        if abs(fc) < tol:
            return iteraciones, c, None
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return iteraciones, c, None

def secante(f, x0, x1, tol=1e-6, max_iter=100):
    iteraciones = []
    for i in range(max_iter):
        if f(x1) - f(x0) == 0:
            return iteraciones, None, "División por cero"
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        error = abs(x2 - x1)
        iteraciones.append([i+1, x0, x1, x2, f(x2), error])
        if error < tol:
            return iteraciones, x2, None
        x0, x1 = x1, x2
    return iteraciones, x2, None

def newton(f, df, x0, tol=1e-6, max_iter=100):
    iteraciones = []
    for i in range(max_iter):
        dfx = df(x0)
        if dfx == 0:
            return iteraciones, None, "Derivada igual a cero"
        x1 = x0 - f(x0) / dfx
        error = abs(x1 - x0)
        iteraciones.append([i+1, x0, f(x0), dfx, x1, error])
        if error < tol:
            return iteraciones, x1, None
        x0 = x1
    return iteraciones, x1, None

# ==================== INTERFAZ DE USUARIO ====================

st.title("Calculadora Visual de Raíces")
st.markdown(
    """
Esta herramienta te permite calcular raíces de funciones utilizando distintos métodos numéricos:
- **Bisección**
- **Regla Falsa**
- **Secante**
- **Newton-Raphson**

Usa la barra lateral para ingresar los parámetros y elegir el método. En las pestañas de abajo verás el resultado, un gráfico interactivo y la tabla de iteraciones.
"""
)

# Barra lateral para parámetros
st.sidebar.header("Parámetros de Entrada")
func_str = st.sidebar.text_input("Ingresa la función f(x):", "x**3 - x - 2")
metodo = st.sidebar.selectbox("Selecciona el método", ["Bisección", "Regla Falsa", "Secante", "Newton-Raphson"])
tol = st.sidebar.number_input("Tolerancia", value=1e-6, format="%.1e")
max_iter = st.sidebar.number_input("Máximo de iteraciones", value=100, step=1)

if metodo in ["Bisección", "Regla Falsa"]:
    a = st.sidebar.number_input("Valor a", value=1.0)
    b = st.sidebar.number_input("Valor b", value=2.0)
elif metodo == "Secante":
    a = st.sidebar.number_input("x0", value=1.0)
    b = st.sidebar.number_input("x1", value=2.0)
else:  # Newton-Raphson
    a = st.sidebar.number_input("x0", value=1.0)

if st.sidebar.button("Calcular"):
    try:
        # Convertir la función de texto a una función de Python usando sympy
        x = sp.symbols('x')
        func = sp.sympify(func_str)
        f = sp.lambdify(x, func, modules=['numpy'])
        df_func = sp.diff(func, x)
        df = sp.lambdify(x, df_func, modules=['numpy'])
        
        # Ejecutar el método seleccionado
        if metodo == "Bisección":
            iteraciones, raiz, error = biseccion(f, a, b, tol, int(max_iter))
        elif metodo == "Regla Falsa":
            iteraciones, raiz, error = regla_falsa(f, a, b, tol, int(max_iter))
        elif metodo == "Secante":
            iteraciones, raiz, error = secante(f, a, b, tol, int(max_iter))
        else:  # Newton-Raphson
            iteraciones, raiz, error = newton(f, df, a, tol, int(max_iter))
        
        # Crear pestañas para mostrar resultados, gráfico e iteraciones
        tabs = st.tabs(["Resultado", "Gráfico", "Iteraciones"])
        
        with tabs[0]:
            if error:
                st.error(f"Error: {error}")
            else:
                st.success(f"Raíz aproximada: {raiz}")
                st.markdown(f"**Función:** $f(x) = {sp.pretty(func)}$")
                st.markdown(f"**Método:** {metodo}")
        
        with tabs[1]:
            st.subheader("Gráfico Interactivo")
            # Definir un rango para el gráfico
            if metodo in ["Bisección", "Regla Falsa"]:
                rango = abs(b - a)
                x_min, x_max = a - rango, b + rango
            else:
                x_min, x_max = raiz - 5, raiz + 5
            X = np.linspace(x_min, x_max, 400)
            Y = f(X)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=Y, mode='lines', name="f(x)"))
            fig.add_trace(go.Scatter(x=[raiz], y=[f(raiz)], mode='markers', name="Raíz", marker=dict(color='red', size=10)))
            fig.update_layout(
                title="Gráfico de f(x)",
                xaxis_title="x",
                yaxis_title="f(x)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.subheader("Tabla de Iteraciones")
            if iteraciones:
                if metodo in ["Bisección", "Regla Falsa"]:
                    headers = ["Iteración", "a", "b", "c", "f(c)", "Error"]
                elif metodo == "Secante":
                    headers = ["Iteración", "x0", "x1", "x2", "f(x2)", "Error"]
                else:  # Newton-Raphson
                    headers = ["Iteración", "x0", "f(x0)", "f'(x0)", "x1", "Error"]
                df_iter = pd.DataFrame(iteraciones, columns=headers)
                st.dataframe(df_iter)
            else:
                st.info("No se generaron iteraciones.")
    
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
else:
    st.info("Introduce los parámetros y presiona el botón 'Calcular' en la barra lateral.")
