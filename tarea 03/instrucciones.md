**Instrucciones tarea 03**


- El código proporcionado en forma de notebook, carga datos desde un archivo cvs (también proporcionado).
Incluí dos archivos, uno con una sola estación de monitoreo y otra con dos. Si quieren usen el de una sola para que sean menos datos.
- El código da formato a los datos como matrices de numpy X e Y.
- El objetivo es modificar los parámetros (ya conocidos) de la red para obtener el mejor desempeño posible
en un escenario de regresión multivariada (pronóstico de dos variables a partir de otras 18).
- La restricción principal es que el desempeño de validación (y test) debe ser lo más cercano posible al de entrenamiento.
Concretamente, usa la métrica para generalización descrita en la celda debajo de la gráfica con las curvas de desempeño por época, y busca que tal métrica sea menor o igual a 0.16
- Entrega un documento PDF con respuestas a la siguientes preguntas, y el notebook con tu mejor modelo.
- Los parámetros que puedes variar son:
 A) número de capas,
 B) número de perceptrones en cada capa,
 C) activaciones de cada capa,
 D) optimizador (mira la documentación en la pagina de keras o tensorflow para saber como cambiar los valores por defecto de los optimizadores),
 E) pérdida


- Incluye respuestas a las siguientes preguntas en tu documento PDF.

 1) ¿Hay algún efecto si normalizas o dejas de normalizar tus datos?
 2) ¿Cómo se comporta la distancia entre el error de entrenamiento y el de validación al incrementar o disminuir el número de capas?
 3) ¿Y al incrementar o disminuir el número denodos en cada capas?
 4) ¿Hay algún impacto en el desempeño si cambias el tamaño de los lotes?
 5) ¿Cuál es la intuición para usar la métrica de generalización que está en la celda debajo de la gráfica con las curvas de desempeño por época?
 6) Incluye una tabla comparativa para tus 5 mejores métodos. Compáralos en términos de los hiperparámetros seleccionados y las pérdidas de entrenamiento y validación?
 7) ¿La pérdida de test de tu mejor modelo es comparable con la de validación y entrenamiento? ¿Qué sugiere eso?
 8) Reporta el tiempo promedio que tardaba en correr cada uno de tus experimentos.
 9) Incluye gráficas de desempeño de tus mejores modelos.

Sugerencia: Usa colab de google con GPU para evaluar tus modelos.

Entrega:

- Plazo: jueves 13 de febrero, antes de las 4:00 pm.
- Envia PDF y notebook por correo electrónico.
