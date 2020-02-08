**Guía para el primer examen de Deep Learning**

***

**1.** Definiciones importantes

Relación entre Inteligencia Artificial, Aprendizaje de Máquina y Aprendizaje profundo


<p align="center">
  <image width="600" height="400" src="https://github.com/dapivei/deep-learning-2020/blob/master/images/ai_ml_dl.png">
</p>

+ Inteligencia Artificial(IA): Un esfuerzo por automizatizar tareas intelectuales normalmente desarrolladas por humanos. Como tal, la IA es un campo general que abarca el aprendizaje automático y el aprendizaje profundo, pero que también incluye muchos más enfoques que no implican ningún aprendizaje.

+ Aprendizaje de Máquina: El aprendizaje de máquina surge de esta pregunta: ¿podría una computadora ir más allá de “lo que sabemos cómo ordenar que realice” y aprender por sí misma cómo realizar una tarea específica? ¿Podría una computadora sorprendernos? En lugar de que los programadores elaboren reglas de procesamiento de datos a mano, ¿podría una computadora aprender estas reglas automáticamente al observar los datos?
Esta pregunta abre la puerta a un nuevo paradigma de programación. En la programación clásica, el paradigma de la IA simbólica, los humanos ingresan las reglas (un programa) y los datos que se procesarán de acuerdo con estas reglas, y salen las respuestas. *Con el aprendizaje automático, los humanos ingresan datos, así como las respuestas esperadas de los datos, y salen las reglas. Estas reglas se pueden aplicar a los nuevos datos para producir respuestas originales.*

<p align="center">
  <image width="600" height="250" src="https://github.com/dapivei/deep-learning-2020/blob/master/images/ai_ml.png">
</p>


Un sistema de aprendizaje de máquina se entrena en lugar de programarse explícitamente. Se presenta con muchos ejemplos relevantes para una tarea, y encuentra estructura estadística en estos ejemplos que eventualmente permite que el sistema presente reglas para automatizar la tarea.


El l aprendizaje de máquina descubre reglas para ejecutar una tarea de procesamiento de datos, dados ejemplos de lo que se espera. Entonces, para hacer aprendizaje automático, necesitamos tres cosas:

+ Puntos de datos de entrada: por ejemplo, si la tarea es el reconocimiento de voz, estos puntos de datos podrían ser archivos de sonido de personas que hablan. Si la tarea es el etiquetado de imágenes, podrían ser imágenes.
+ Ejemplos de la salida esperada: en una tarea de reconocimiento de voz, estas podrían ser transcripciones de archivos de sonido generadas por humanos. En una tarea de imagen, los resultados esperados podrían ser etiquetas como "perro", "gato", etc.
+ Una forma de medir si el algoritmo está haciendo un buen trabajo: esto es necesario para determinar la distancia entre la salida actual del algoritmo y su salida esperada. La medición se utiliza como señal de retroalimentación para ajustar la forma en que funciona el algoritmo. Este paso de ajuste es lo que llamamos aprendizaje.

Un modelo de aprendizaje de máquina transforma sus datos de entrada en resultados significativos, un proceso que se "aprende" de la exposición a ejemplos conocidos de entradas y salidas. Por lo tanto, el problema central en el aprendizaje de máquina y el aprendizaje profundo es transformar significativamente los datos: en otras palabras, aprender representaciones útiles de los datos de entrada disponibles, representaciones que nos acercan a la salida esperada.

El aprendizaje, en el contexto del aprendizaje de máquina, describe un proceso de búsqueda automática para obtener mejores representaciones.
Todos los algoritmos de aprendizaje automático consisten en encontrar automáticamente tales transformaciones que convierten los datos en representaciones más útiles para una tarea determinada.

Los algoritmos de aprendizaje de máquina no suelen ser creativos en
encontrar estas transformaciones; simplemente están buscando a través de un conjunto predefinido de operaciones, llamado espacio de hipótesis. Entonces, eso es el aprendizaje automático, técnicamente: *buscar representaciones útiles de algunos datos de entrada, dentro de un espacio predefinido de posibilidades, utilizando la guía de una señal de retroalimentación.*


+ Deep Learning:

El aprendizaje profundo es un subcampo específico del aprendizaje automático: una nueva versión de las representaciones de aprendizaje a partir de datos que pone énfasis en el aprendizaje de capas sucesivas de representaciones cada vez más significativas; representa esta idea de capas sucesivas de representaciones. La cantidad de capas que contribuyen a un modelo de datos se denomina profundidad del modelo. El aprendizaje profundo moderno a menudo implica decenas o incluso cientos de capas sucesivas de representaciones, y todas se aprenden automáticamente de la exposición a los datos de entrenamiento. Mientras tanto, otros enfoques del aprendizaje automático tienden a centrarse en aprender solo una o dos capas de representaciones de los datos; por lo tanto, a veces se les llama aprendizaje superficial.

Entonces, eso es el aprendizaje profundo, técnicamente: una forma de etapas múltiples para aprender representaciones de datos.

La mayor parte de los métodos de aprendizaje emplean arquitecturas de redes neuronales, por lo que, a menudo, los modelos de aprendizaje profundo se denominan redes neuronales profundas.

El término **profundo** suele hacer referencia al número de capas ocultas en la red neuronal. Las redes neuronales tradicionales solo contienen dos o tres capas ocultas, mientras que las redes profundas pueden tener hasta 150.

Los modelos de Deep Learning se entrenan mediante el uso de extensos conjuntos de datos etiquetados y arquitecturas de redes neuronales que aprenden directamente a partir de los datos, sin necesidad de una extracción manual de características.



<p align="center">
  <image width="600" height="300" src="https://github.com/dapivei/deep-learning-2020/blob/master/images/nn_1.png">
</p>

<p align="center">
  <image width="600" height="400" src="https://github.com/dapivei/deep-learning-2020/blob/master/images/nn_2.png">
</p>

<p align="center">
  <image width="600" height="400" src="https://github.com/dapivei/deep-learning-2020/blob/master/images/nn_3.png">
</p>


**2. Relación entre regresión y clasificación. ¿Por qué podemos pensar en el mismo en el mismo modelo para hacer regresión y para hacer clasificación? ¿Por qué habíamos dicho que para fines de Redes Neuronales son problemas equivalentes?**

Podemos pensar en el *problema de clasificación* de la siguiente manera: dado un conjunto de datos, queremos aprender a separarlos y pronosticar una clase o etiqueta que normalmente es -1 y 1.


Para ello, empleamos la ecuación de la recta:

$$Y=\omega_{1}x_{1}+ \omega_{2}x_{2}+\underbrace{b}_{\omega_{0}x_{0}}$$

Donde:

* $Y$ es la etiqueta

* $x_{0}$ es igual a $1$

* queremos aprender un parámetros de $\omega_{0}$

Ecuación que podemos generalizar en términos de algebra lineal:

$$Y=WX$$

Podemos pensar en el *problema de regresión*  de la siguiente manera: tenemos un *set* de datos, que no están clasificados, y tenemos una línea que aproxima a los datos; esa línea se puede estimar con la misma ecuación general, con la única diferencia que $Y$ es un número real.


**3.** La importancia de los datos de entrenamiento y prueba. ¿Qué pasa con uno, qué pasa con el desempeño del otro?


**4.** La diferencia entre parámetros e hiperparámetros. ¿Qué lo hace un parámetro y que lo hace un hiperparámetro? ¿Cuál es su relación con los *datos de entrenamiento* y *datos de validación*? ¿Cuál es su relación el *overfitting* y el *underfitting*?

Los parámetros son los que el modelo emplea para hacer predicciones; por ejemplo, el peso de los coeficientes en una regresión lineal.  


Los hiperparámetros son los que ayudan con el proceso de aprendizaje. Por ejemplo, el número de clusters en K-means, el factor de  For example, number of clusters in K-Means, el factor de penalización en la regresión de Ridge. Los hiperparámetros no aparecen al final de predicción, pero tienen una gran influencia sobre los valores de los parámetros después del aprendizaje.

**5.** Perceptrón, concepto e idea. ¿Qué modela? ¿Cómo calculo el número de parámetros en una red neuronal? ¿Qué tengo que considerar al momento de determinar el número de parámetros de una red neuronal?


**6.** Foward propagation, backward propagation, intuición de la regla de la cadena. Ejemplo: Si tengo una red de este tipo, ¿qué nodos o parámetros debo considerar en el ajuste de tal parámetro?



**7.** Banishing gradient and exploding gradiente.

Es por eso que sigmoide no es una buena función de activación por los problemas de Banishing Gradiente and Exploding Gradient.

**8.** Las funciones de activación. ¿Cuándo conviene usar una, cuándo conviene usar otra?

**9.** Optimizadores.

**10.** Las funciones de pérdidas: ¿cuándo conviene usar una?, ¿cuándo conviene usar otra?

