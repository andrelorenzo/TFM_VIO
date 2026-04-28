# Modelo matemático de propagación / preintegración IMU

## Variables

Sea la IMU con medidas

$$
\omega_m \in \mathbb{R}^3, \qquad a_m \in \mathbb{R}^3
$$

y biases

$$
b_g \in \mathbb{R}^3, \qquad b_a \in \mathbb{R}^3
$$

Las medidas corregidas son

$$
\omega = \omega_m - b_g
$$

$$
a = a_m - b_a
$$

La gravedad se representa como

$$
g_w = \gamma \, \hat g
$$

donde:

- $\gamma$ es la magnitud de la gravedad
- $\hat g$ es un vector unitario de dirección

---

## Estado nominal

En la propagación relativa usada por `Propagator.cc`, el estado nominal es

$$
x_k =
\begin{bmatrix}
R_k,\; p_k,\; v_k,\; b_g,\; b_a
\end{bmatrix}
$$

donde:

- $R_k \in SO(3)$: rotación relativa
- $p_k \in \mathbb{R}^3$: posición relativa
- $v_k \in \mathbb{R}^3$: velocidad
- $b_g, b_a$: biases de gyro y accel

Además se usa:

- $v_R$: velocidad inicial en el frame local
- $\hat g_R$: dirección inicial de gravedad en el frame local

---

## Operador skew

Para un vector $x = [x_1, x_2, x_3]^\top$:

$$
[x]_\times =
\begin{bmatrix}
0 & -x_3 & x_2 \\
x_3 & 0 & -x_1 \\
-x_2 & x_1 & 0
\end{bmatrix}
$$

---

## Incremento de rotación

Para un intervalo $\Delta t$, con $\|\omega\| = \omega_n$, la rotación incremental es

$$
\Delta R = \exp(-[\omega]_\times \Delta t)
$$

En forma cerrada:

$$
\Delta R
=
I
-
\frac{\sin(\omega_n \Delta t)}{\omega_n}[\omega]_\times
+
\frac{1-\cos(\omega_n \Delta t)}{\omega_n^2}[\omega]_\times^2
$$

y la actualización es

$$
R_{k+1} = \Delta R \, R_k
$$

### Aproximación para ángulo pequeño

Si $\omega_n < \varepsilon$:

$$
\Delta R \approx I - \Delta t[\omega]_\times + \frac{\Delta t^2}{2}[\omega]_\times^2
$$

---

## Integración de velocidad y posición relativas

Se definen incrementos acumulados

$$
\Delta v, \qquad \Delta p
$$

con inicialización

$$
\Delta v = 0, \qquad \Delta p = 0
$$

La integración discreta usada es

$$
\Delta p \leftarrow \Delta p + \Delta v \Delta t + R_k^\top J_2(\omega,\Delta t)a
$$

$$
\Delta v \leftarrow \Delta v + R_k^\top J_1(\omega,\Delta t)a
$$

donde

$$
J_1(\omega,\Delta t) = \Delta t\,I + f_3[\omega]_\times + f_4[\omega]_\times^2
$$

$$
J_2(\omega,\Delta t) = \frac{\Delta t^2}{2}I + f_1[\omega]_\times + f_2[\omega]_\times^2
$$

con

$$
f_1 = \frac{\omega_n\Delta t \cos(\omega_n\Delta t)-\sin(\omega_n\Delta t)}{\omega_n^3}
$$

$$
f_2 = \frac{(\omega_n\Delta t)^2 - 2\cos(\omega_n\Delta t) - 2\omega_n\Delta t\sin(\omega_n\Delta t) + 2}{2\omega_n^4}
$$

$$
f_3 = \frac{\cos(\omega_n\Delta t)-1}{\omega_n^2}
$$

$$
f_4 = \frac{\omega_n\Delta t-\sin(\omega_n\Delta t)}{\omega_n^3}
$$

### Aproximación para ángulo pequeño

Si $\omega_n < \varepsilon$:

$$
f_1 \approx -\frac{\Delta t^3}{3}, \qquad
f_2 \approx \frac{\Delta t^4}{8}, \qquad
f_3 \approx -\frac{\Delta t^2}{2}, \qquad
f_4 \approx \frac{\Delta t^3}{6}
$$

---

## Estado propagado tras un intervalo total $\Delta T$

Tras acumular todas las muestras IMU entre dos frames:

$$
p_k = v_R \Delta T - \frac{1}{2}\gamma \hat g_R \Delta T^2 + \Delta p
$$

$$
v_k = R_k \left( v_R - \gamma \hat g_R \Delta T + \Delta v \right)
$$

$$
\hat g_k = R_k \hat g_R
$$

y después se normaliza:

$$
\hat g_k \leftarrow \frac{\hat g_k}{\|\hat g_k\|}
$$

---

## Modelo linealizado del error

El estado de error es

$$
\delta x =
\begin{bmatrix}
\delta \hat g &
\delta \theta &
\delta p &
\delta v &
\delta b_g &
\delta b_a
\end{bmatrix}^\top
\in \mathbb{R}^{18}
$$

La dinámica linealizada es

$$
\dot{\delta x} = F\,\delta x + G\,n
$$

con ruido

$$
n =
\begin{bmatrix}
n_g &
n_{wg} &
n_a &
n_{wa}
\end{bmatrix}^\top
$$

y matrices

$$
F =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & -[\omega]_\times & 0 & 0 & -I & 0 \\
0 & -R^\top[v]_\times & 0 & R^\top & 0 & 0 \\
-\gamma R & -\gamma[\hat g]_\times & 0 & -[\omega]_\times & -[v]_\times & -I \\
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

$$
G =
\begin{bmatrix}
0 & 0 & 0 & 0 \\
-I & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
-[v]_\times & 0 & -I & 0 \\
0 & I & 0 & 0 \\
0 & 0 & 0 & I
\end{bmatrix}
$$

---

## Covarianza del ruido IMU

La covarianza continua del ruido es

$$
\Sigma = \mathrm{diag}
\left(
\sigma_g^2,\;
\sigma_{wg}^2,\;
\sigma_a^2,\;
\sigma_{wa}^2
\right)
$$

donde:

- $\sigma_g$: ruido blanco del giróscopo
- $\sigma_{wg}$: random walk del bias del giróscopo
- $\sigma_a$: ruido blanco del acelerómetro
- $\sigma_{wa}$: random walk del bias del acelerómetro

---

## Discretización

La discretización usada es de primer orden:

$$
\Phi = I + \Delta t\,F
$$

$$
Q = \Delta t \, G \Sigma G^\top
$$

$$
P \leftarrow \Phi P \Phi^\top + Q
$$

$$
\Psi \leftarrow \Phi \Psi
$$

donde:

- $P$: covarianza acumulada del error
- $\Psi$: transición acumulada del error

---

## Factor IMU final

Al terminar la integración, se toma el bloque de covarianza del estado actual:

$$
P_c = P_{(3:17,\,3:17)}
$$

y se calcula la información:

$$
\Lambda = P_c^{-1}
$$

Luego, usando Cholesky:

$$
\Lambda = L L^\top
$$

el jacobiano blanqueado del factor IMU queda:

$$
H =
L^\top
\begin{bmatrix}
\Psi_{(3:17,\,0:2)} &
\Psi_{(3:17,\,9:17)} &
-I_{15}
\end{bmatrix}
$$

y el estado nominal propagado es

$$
x_k =
\begin{bmatrix}
q(R_k),\; p_k,\; v_k,\; b_g,\; b_a
\end{bmatrix}
$$

---

## Resumen intuitivo

La idea del método es:

1. corregir gyro y accel con sus biases
2. integrar rotación relativa
3. integrar velocidad y posición relativas
4. propagar la covarianza del error
5. construir un factor IMU entre dos estados consecutivos