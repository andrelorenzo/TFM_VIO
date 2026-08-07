# Guía de presentación del TFM

## Objetivo de esta guía
Esta guía está pensada para ayudarte a montar una presentación de unos 18 minutos, dejando 1 o 2 minutos de margen por si te detienes en alguna diapositiva o surge una transición más lenta. La idea es que no intentes resumir toda la memoria, sino defender con claridad el valor del proyecto y demostrar que entiendes muy bien sus decisiones técnicas y sus resultados.

El error más habitual en este tipo de presentaciones es querer explicarlo todo. En tu caso, funciona mejor centrar la exposición en cinco ideas:

- cuál era el problema,
- qué arquitectura has propuesto,
- cómo funcionan los bloques principales,
- qué resultados objetivos has conseguido,
- y qué limitaciones o mejoras futuras quedan abiertas.

## Estructura recomendada

- Diapositiva 1: título y contexto, 1 minuto
- Diapositiva 2: problema y motivación, 2 minutos
- Diapositiva 3: objetivos y aportación, 2 minutos
- Diapositiva 4: arquitectura general, 2 minutos
- Diapositiva 5: comunicación y sincronización, 2 minutos
- Diapositiva 6: odometría visual-inercial, 2 minutos
- Diapositiva 7: profundidad y evitación de obstáculos, 2 minutos
- Diapositiva 8: planificación y control, 2 minutos
- Diapositiva 9: resultados experimentales, 3 minutos
- Diapositiva 10: conclusiones y trabajo futuro, 2 minutos

Total aproximado: 18 minutos.

---

## Diapositiva 1. Título y contexto

### Qué mostrar

- Título del TFM.
- Tu nombre.
- Tutor.
- Universidad.
- Una imagen del sistema completo o de la plataforma.

### Qué decir

"Buenos días. Voy a presentar mi Trabajo Fin de Máster, titulado 'Navegación autónoma de un UAV en interiores mediante técnicas de visión monocular'. El proyecto se centra en un problema muy relevante dentro de la robótica móvil y aérea: cómo conseguir que un vehículo pueda desplazarse de forma autónoma en interiores, estimando su movimiento y reaccionando ante obstáculos, incluso cuando no disponemos de GPS ni de sensores muy costosos.

En este trabajo he desarrollado un sistema completo que combina visión monocular, información inercial y una arquitectura distribuida entre el vehículo y una estación de tierra. La idea principal ha sido encontrar un equilibrio entre funcionalidad y coste computacional. Es decir, no solo que el sistema sea capaz de navegar y evitar obstáculos, sino que además lo haga con una configuración realista para un UAV con restricciones de peso, energía y procesamiento.

Durante la presentación voy a explicar primero el problema y los objetivos, después la arquitectura y los módulos principales del sistema, y finalmente los resultados y conclusiones obtenidos."

---

## Diapositiva 2. Problema y motivación

### Qué mostrar

- Entorno interior sin GPS.
- Necesidad de estimar pose.
- Necesidad de evitar obstáculos.
- Restricciones de cómputo y sensórica.

### Qué decir

"El problema de partida aparece en navegación interior. En estos entornos no podemos depender de GPS, y además el vehículo debe ser capaz de desplazarse de forma segura en espacios que pueden ser estrechos, dinámicos o visualmente poco estructurados. Eso obliga a resolver dos tareas al mismo tiempo: por un lado, estimar el estado del vehículo, es decir, saber dónde está y cómo se mueve; y por otro, interpretar el entorno para detectar posibles obstáculos y reaccionar a tiempo.

En UAVs pequeños esto es especialmente complicado porque hay restricciones importantes de peso, consumo energético y capacidad de procesamiento. No siempre es viable montar sensores pesados o soluciones de percepción demasiado caras computacionalmente. Por eso resulta atractivo trabajar con una configuración reducida, basada en una cámara monocular y una IMU.

La motivación del proyecto nace precisamente de ahí: estudiar hasta qué punto se puede construir un sistema útil de navegación autónoma interior con una sensórica mínima y con una arquitectura capaz de mantener tiempos de procesamiento compatibles con uso en tiempo real. Esa combinación entre percepción, estimación y viabilidad práctica es realmente el núcleo del trabajo."

---

## Diapositiva 3. Objetivos y aportación del proyecto

### Qué mostrar

- Objetivo general.
- Tres objetivos específicos:
- VIO,
- profundidad monocular,
- integración con navegación y control.

### Qué decir

"A partir de ese problema, el objetivo general del trabajo ha sido diseñar e integrar un sistema de navegación autónoma para un UAV en interiores utilizando únicamente una cámara monocular y una IMU como fuentes sensoriales principales.

Este objetivo general se ha concretado en tres objetivos técnicos. El primero ha sido desarrollar un módulo de odometría visual-inercial, o VIO, capaz de estimar la pose del vehículo combinando información visual e inercial. El segundo ha sido incorporar un estimador de profundidad monocular que permita obtener información útil del entorno para la evitación de obstáculos. Y el tercero ha sido integrar ambas fuentes dentro de un bloque de planificación y control que genere comandos coherentes para el vehículo.

La aportación del proyecto no está en proponer un algoritmo completamente nuevo desde el punto de vista teórico, sino en construir un sistema funcional de extremo a extremo, justificar sus decisiones de diseño y validar que la arquitectura resultante cumple el objetivo temporal del proyecto. Es decir, la contribución está en la integración, adaptación y validación de una solución completa y utilizable."

---

## Diapositiva 4. Arquitectura general del sistema

### Qué mostrar

- Diagrama de arquitectura.
- Nodo embarcado.
- Estación de tierra.
- Flujo de información entre módulos.

### Qué decir

"La arquitectura del sistema se ha planteado de forma distribuida. En el UAV se realiza la adquisición de datos y el envío de la información, mientras que el procesamiento más pesado se desplaza a la estación de tierra. Esta decisión responde a una motivación práctica muy clara: si toda la carga computacional se mantuviera a bordo, el sistema sería mucho más difícil de sostener en una plataforma aérea ligera.

En el nodo embarcado se adquieren las imágenes RGB y las medidas inerciales, y esos datos se transmiten hacia la estación de tierra mediante la pasarela de comunicación. Una vez allí, los datos se sincronizan y alimentan a los módulos principales. Por una parte, la odometría visual-inercial estima la pose y la velocidad del vehículo. Por otra, el estimador de profundidad monocular analiza la escena para apoyar la detección y la evitación de obstáculos.

Finalmente, toda esa información se integra en el planificador y en el controlador, que generan los comandos de movimiento. Esta diapositiva es importante porque resume el flujo completo del proyecto y deja claro que no estamos ante algoritmos aislados, sino ante un sistema conectado de principio a fin."

---

## Diapositiva 5. Comunicación y sincronización temporal

### Qué mostrar

- Pasarela de comunicación.
- Figura o esquema de sincronización.
- Frase destacada: no se sincroniza por tiempo de llegada.

### Qué decir

"Una parte crítica del proyecto ha sido la comunicación y, sobre todo, la sincronización temporal entre vídeo e IMU. Esto es importante porque en el sistema el canal visual y el canal inercial no llegan necesariamente al mismo tiempo ni recorren la misma ruta de comunicación. Por tanto, sincronizar por instante de llegada sería incorrecto y generaría errores en el estimador.

La solución adoptada consiste en conservar las marcas temporales de captura generadas por la propia cámara. En el caso de adquisición remota, el timestamp visual se inserta y se recupera junto al flujo de vídeo, mientras que las medidas IMU también mantienen sus marcas temporales de captura. A partir de ahí, cada imagen se asocia al intervalo inercial correspondiente entre dos fotogramas consecutivos.

Esto permite reconstruir paquetes coherentes para el estimador, evitando depender del jitter de red o del orden de llegada de los mensajes. Aunque puede parecer un detalle de implementación, en realidad es una pieza muy relevante del proyecto, porque la calidad del VIO depende mucho de que la correspondencia temporal entre imagen e inerciales esté bien resuelta."

---

## Diapositiva 6. Odometría visual-inercial

### Qué mostrar

- Diagrama del VIO.
- Detección y seguimiento de características.
- Preintegración IMU.
- Fusión visual-inercial.
- Salida de pose.

### Qué decir

"El primer gran bloque funcional es la odometría visual-inercial. Su misión es estimar el estado del vehículo combinando la información de la cámara y la IMU. La parte visual aporta referencias geométricas de la escena a partir de características detectadas y seguidas entre fotogramas, mientras que la parte inercial aporta información de movimiento a alta frecuencia.

Dentro de este bloque, el sistema realiza detección y seguimiento de características visuales, preintegración de las medidas inerciales y una etapa de fusión que corrige y estabiliza la estimación de pose. También se ha prestado atención al modelado práctico del ruido inercial y a la sincronización de las medidas, porque ambos factores afectan directamente al rendimiento del estimador.

Aquí conviene transmitir una idea clara: el objetivo no era implementar el VIO más sofisticado posible, sino uno que fuera suficientemente robusto y, al mismo tiempo, compatible con las restricciones temporales del sistema completo. Por eso el diseño se ha orientado hacia una solución operativa, optimizada para integrarse con el resto de módulos de navegación y no únicamente para maximizar precisión en un entorno ideal."

---

## Diapositiva 7. Estimación de profundidad y evitación de obstáculos

### Qué mostrar

- Arquitectura del evitador.
- ROI frontal.
- ROI de guiado.
- Ejemplo visual de obstáculo y dirección de evasión.

### Qué decir

"El segundo bloque principal del sistema es la percepción de obstáculos mediante profundidad monocular. Para ello se utiliza un estimador basado en Depth Anything 3, combinado con un postprocesado específico para navegación. La idea aquí no es obtener una reconstrucción métrica perfecta del entorno en todos los casos, sino una representación suficientemente informativa para detectar situaciones de riesgo y orientar la maniobra de evasión.

El método distingue dos funciones. Por un lado, una ROI frontal que se usa para decidir si existe un obstáculo lo bastante relevante como para activar evasión. Por otro, una ROI de guiado que sirve para estimar hacia qué zona de la imagen conviene desplazar la referencia lateral. Es decir, se separa la decisión de 'hay que evitar' de la decisión de 'hacia dónde conviene evitar'.

Esa separación hace el comportamiento más robusto que una estrategia basada únicamente en un gradiente global de profundidad. Además, permite generar variables intermedias interpretables, como la bandera de activación, el ángulo de evasión y la puntuación de obstáculo, que luego usa el planificador."

---

## Diapositiva 8. Planificación y control

### Qué mostrar

- Esquema del planificador local y del controlador.
- Referencia nominal.
- Señales `must_evade` y `evade_angle`.
- Comandos finales de velocidad y guiñada.

### Qué decir

"Una vez estimado el estado del vehículo y analizado el entorno, el siguiente paso es decidir cómo moverse. Aquí intervienen el bloque de planificación y el controlador. La planificación combina la referencia nominal de trayectoria con la información reactiva procedente del módulo de evitación. Si no se detecta riesgo, el vehículo sigue la referencia prevista; si se detecta un obstáculo, la referencia local se modifica para generar una maniobra compatible con la evasión.

Después, el controlador transforma esa referencia en comandos concretos de velocidad lineal y velocidad de guiñada. En esta parte es importante remarcar que el sistema no se limita a detectar obstáculos, sino que integra esa información dentro del flujo de navegación completo hasta producir órdenes de movimiento utilizables.

Por tanto, esta diapositiva representa la conexión entre percepción y acción. Es donde se ve realmente el valor de integrar todos los módulos anteriores: la profundidad y el VIO no quedan como salidas aisladas, sino que afectan de forma directa al comportamiento final del vehículo."

---

## Diapositiva 9. Resultados experimentales

### Qué mostrar

- Un resultado del VIO.
- Un resultado de evitación o coherencia de comandos.
- Tabla o resumen de frecuencia.
- Dato destacado: alrededor de 13 Hz y requisito de 10 Hz cumplido.

### Qué decir

"En cuanto a la validación experimental, aquí conviene centrarse en tres mensajes claros. El primero es que el sistema de odometría visual-inercial proporciona trayectorias coherentes para la navegación interior. Aunque en algunos ensayos la referencia disponible no es un ground truth exacto, sí permite comprobar que la estimación mantiene una evolución razonable y útil para el objetivo del proyecto.

El segundo mensaje es que el módulo de evitación responde de forma consistente ante escenas con obstáculo. En los ensayos se observa coherencia entre la imagen de entrada, la estimación de profundidad, la dirección lateral seleccionada y los comandos resultantes. Es decir, el comportamiento del sistema es interpretable y sigue la lógica esperada de navegación reactiva.

Y el tercer mensaje, que probablemente es el más importante de cara al cumplimiento del proyecto, es el resultado temporal. Tras la optimización del sistema, la frecuencia extremo a extremo alcanzada se sitúa alrededor de 13 Hz, por encima del requisito de 10 Hz fijado inicialmente. Eso significa que el sistema no solo funciona conceptualmente, sino que además cumple la restricción práctica que determinaba su viabilidad en tiempo real."

---

## Diapositiva 10. Conclusiones y trabajo futuro

### Qué mostrar

- Tres conclusiones.
- Dos o tres líneas futuras.

### Qué decir

"Como conclusiones, la primera idea es que el proyecto demuestra la viabilidad de una arquitectura de navegación interior basada en sensórica reducida, apoyándose en una cámara monocular y una IMU. La segunda es que se ha conseguido integrar de manera funcional la estimación de estado, la percepción de obstáculos y la generación de comandos dentro de un único flujo de navegación. Y la tercera, especialmente importante, es que el sistema ha alcanzado una frecuencia de funcionamiento compatible con el requisito temporal marcado para el proyecto.

En cuanto al trabajo futuro, hay varias líneas interesantes. Una de ellas sería reforzar la validación experimental con referencias externas más precisas, de forma que pueda medirse con mayor exactitud el error absoluto de trayectoria. Otra sería seguir mejorando la robustez del VIO y del módulo de evasión en escenarios más exigentes o con maniobras más agresivas. Y una tercera línea sería evolucionar la arquitectura hacia una mayor autonomía embarcada, reduciendo la dependencia de la estación de tierra.

En conjunto, el trabajo deja una base funcional y coherente sobre la que seguir desarrollando soluciones de navegación autónoma para UAVs en interiores."

---

## Diapositiva final. Preguntas

### Qué mostrar

- "Gracias por su atención"
- Una imagen final del sistema o una diapositiva limpia.

### Qué decir

"Esto resume las principales aportaciones del trabajo. He intentado mostrar no solo los módulos por separado, sino también cómo se integran y qué resultados permiten obtener como sistema completo. Muchas gracias por su atención. Quedo a su disposición para cualquier pregunta."

---

## Consejos de estilo para las diapositivas

- No pongas párrafos largos en pantalla.
- Usa una idea principal por diapositiva.
- Enseña diagramas y resultados visuales antes que bloques grandes de texto.
- Evita meter demasiadas ecuaciones salvo que sepas que te las van a valorar especialmente.
- Resalta en resultados solo los números que realmente apoyan tu mensaje.
- Si una figura necesita un minuto entero para entenderse, simplifícala antes de presentarla.

## Consejos para el discurso oral

- Habla algo más despacio de lo que te parezca natural cuando ensayes.
- No memorices palabra por palabra: memoriza la estructura y los mensajes clave.
- En cada bloque, responde siempre a estas dos preguntas:
- para qué sirve,
- y por qué lo resolviste así.
- Cuando llegues a resultados, no enumeres todas las pruebas: selecciona las más fuertes.
- Cierra con una conclusión objetiva: el sistema funciona y cumple el requisito temporal marcado.

## Frase resumen útil para abrir o cerrar

"El proyecto desarrolla un sistema distribuido de navegación autónoma para UAV en interiores que integra odometría visual-inercial, estimación de profundidad monocular y control reactivo, alcanzando una frecuencia de funcionamiento compatible con el requisito de tiempo real planteado."
