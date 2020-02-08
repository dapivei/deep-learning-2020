**Guía para el primer examen de Deep Learning**

***

**1.** Definiciones importantes

Relación entre Inteligencia Artificial, Aprendizaje de Máquina y Aprendizaje profundo


<p align="center">
  <image width="600" height="400" src="ttps://github.com/dapivei/deep-learning-2020/tree/master/images">
</p>



+ Inteligencia Artificial: Un esfuerzo por automizatizar tareas intelectuales normalmente desarrolladas por humanos. Como tal, la IA es un campo general que abarca el aprendizaje automático y el aprendizaje profundo, pero que también incluye muchos más enfoques que no implican ningún aprendizaje.

+ Aprendizaje de Máquina:


+ Deep Learning:

- ¿Qué lo hace Deep Learning? ¿Por qué no solo Learning?

La mayor parte de los métodos de aprendizaje emplean arquitecturas de redes neuronales, por lo que, a menudo, los modelos de aprendizaje profundo se denominan redes neuronales profundas.

El término **profundo** suele hacer referencia al número de capas ocultas en la red neuronal. Las redes neuronales tradicionales solo contienen dos o tres capas ocultas, mientras que las redes profundas pueden tener hasta 150.

Los modelos de Deep Learning se entrenan mediante el uso de extensos conjuntos de datos etiquetados y arquitecturas de redes neuronales que aprenden directamente a partir de los datos, sin necesidad de una extracción manual de características.


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


**7.**
