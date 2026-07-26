---
title: "Tu gusto, leído desde los modelos"
geometry: margin=2.3cm
fontsize: 11pt
---

*Un retrato de cómo escuchás música, trazado no a partir de contar
reproducciones sino desarmando el modelo entrenado de "próxima canción" para
preguntarle qué tuvo que aprender de vos con tal de acertar qué ponés después.
Usamos varios métodos independientes y —lo tranquilizador— todos apuntan al mismo
lado. Los números de reproducción crudos aparecen recién sobre el final, como
contraste. Es exploratorio y cariñoso; no un veredicto.*

## Una nota sobre la matemática (liviana)

No hace falta saber de machine learning: alcanza con un poco de álgebra lineal,
algo de probabilidad y la idea de sistema dinámico.

- **Cada tema es un punto.** El modelo convierte cada canción en un vector dentro
  de un **"espacio de sonido" de 192 dimensiones** que aprendió solo. Una sesión de
  escucha es, entonces, un *recorrido* por ese espacio.
- **El predictor es una recurrencia.** Leyendo la sesión tema por tema, arrastra
  una memoria $h_t$ que actualiza así:
  $$h_t=(1-z_t)\odot n_t + z_t\odot h_{t-1},$$
  es decir, en cada paso interpola —coordenada a coordenada ($\odot$ es
  elemento a elemento)— entre la memoria anterior $h_{t-1}$ y un candidato nuevo
  $n_t$, con una **fracción de retención** $z_t\in(0,1)$ aprendida (la "compuerta de
  actualización"). Ese es todo el mecanismo que después inspeccionamos.
- **Para leer la memoria, la factorizamos.** Buscamos un diccionario $D$ y códigos
  *ralos* $c$ tales que $h\approx Dc$ con muy pocas entradas no nulas (un
  *autoencoder disperso*). Cada columna del diccionario (un "átomo") termina
  comportándose como un concepto: casi siempre un artista o un género.
- **El flujo es una matriz y un grafo.** El modelo de co-ocurrencia ("Markov") no
  es más que la matriz de transición $P(\text{próximo}\mid\text{actual})$; la
  **tasa de permanencia** de un género es su entrada diagonal. Tomando los géneros
  como un grafo dirigido y pesado, **PageRank** (la distribución estacionaria de un
  paseo aleatorio) encuentra los *hubs*, y la **intermediación** (fracción de
  caminos más cortos que pasan por un nodo) encuentra los *puentes*.
- **Las sesiones se agrupan** con k-means (cada sesión → su vector promedio →
  agrupadas por centro más cercano), y los **atractores** salen de *iterar el mapa*
  (poné una semilla, predecí la siguiente, saltá al tema real más cercano, repetí)
  y mirar dónde se asienta la órbita.

Todo lo que sigue se lee de estas construcciones, desde adentro del modelo.

## 1. Un mapa de tu escucha

Si representamos cada sesión por su punto promedio en el espacio de sonido y las
agrupamos, aparecen seis **modos** nítidos. El "terreno" sombreado de abajo es su
densidad: los valles son donde se te amontonan las sesiones.

![Cada punto es una sesión; los valles más oscuros son los modos en los que te instalás. La música argentina (naranja) y los dos modos electrónicos (verde azulado) son grandes y separados; la cuenca de Rammstein/metal (rojo) queda aparte.](figures/taste-landscape-es.pdf){width=88%}

- **Dos modos electrónicos, ~60% de todo.** Uno de club/**UK garage** (Four Tet,
  Joy Orbison, Aphex Twin, Daphni — 31%) y otro de **big-beat/trip-hop** (The
  Avalanches, Underworld, Massive Attack, Air — 29%).
- **La música argentina se parte en dos:** un modo **indie** contemporáneo (Peces
  Raros, Laika Perra Rusa — 17%) y un modo **canon** (Babasonicos, Gustavo Cerati
  — 9%).
- **Un modo metal/rock (Rammstein, 10%) — y es el *corto*:** ~9 temas por sesión
  contra ~14 en el resto. El metal lo escuchás en ráfagas breves y autocontenidas.
- **Un modo chico kraut/ambient (Kraftwerk, Nicolás Jaar — 4%).**

## 2. Qué considera central el modelo

A cada concepto le tocan tantos átomos del diccionario como el modelo necesite
para representarlo; por eso **la cantidad de átomos es su propia medida de
importancia**, independiente de cuánto lo hayas escuchado.

![Artistas ordenados por cuántas direcciones internas les dedica el modelo.](figures/taste-importance-es.pdf){width=78%}

**Four Tet encabeza al modelo** aunque no sea tu más escuchado: es el que más
direcciones distintas exige para quedar capturado. Y un resultado en sí mismo:
aun estirando a **500 átomos**, siguen mapeando a solo **~25 artistas y ~13
géneros**. Pero ese "reparto chico" hay que leerlo con cuidado — ver la sección
que sigue.

## El otro 99,6% — tu amplitud (lo que el modelo *no* ve)

El "reparto chico" se presta a una lectura equivocada. **No** es a cuántos
artistas escuchás; es a cuántos el modelo les puede armar *estructura*. Por
construcción, el autoencoder disperso solo le hace crecer un átomo a un concepto
que se repite lo suficiente como para ayudar a predecir la próxima canción: un
artista de una sola escucha nunca se gana uno. Y lo tuyo es, en su enorme mayoría,
de una sola escucha:

![Vas muy a lo ancho, pero fino. Reproducciones acumuladas vs. artistas acumulados (una curva de Lorenz): cuanto más se arquea lejos de la diagonal, más concentrado. El punto rojo marca tus 25 top — todo el "reparto" del modelo.](figures/taste-breadth-es.pdf){width=74%}

Escuchaste **5.594 artistas**, pero el reparto es extremo (Gini **0,88**): al
**42% lo escuchaste una sola vez y al 56% no más de dos** — la mediana es dos
reproducciones. Tus 25 top —todo el reparto del modelo— son el **0,4% de tus
artistas y, aun así, un tercio de todas las reproducciones**. Así que la foto
honesta es lo contrario de "acotada": sos un **explorador de amplitud enorme** con
un núcleo chico y pesado. Todo el resto de este documento describe ese núcleo,
porque la repetición es lo único sobre lo que el modelo puede razonar; la cola
inmensa y fina de todo-lo-que-probaste-una-vez es igual de real y, sencillamente,
**invisible para esta lente**.

## 3. Valles y pasos de montaña — "bloques" vs "puentes"

El hallazgo más sólido, porque **dos métodos sin relación entre sí lo producen por
separado.** Para cada género calculamos su **tasa de permanencia** de Markov (con
qué frecuencia lleva a sí mismo) y, aparte, el **tamaño del salto en la memoria
neuronal** $\lVert\Delta h\rVert$ cuando llega un tema de ese género. Coinciden:

![A la derecha/abajo, te quedás (bloques); a la izquierda/arriba, el estado se reorganiza (puentes). El modelo de co-ocurrencia y la red neuronal dicen lo mismo.](figures/taste-blocks-bridges-es.pdf){width=80%}

- **Bloques** — los valles donde te instalás: **alternative metal (permanencia
  0,71)**, la dusseldorf-electronic de Kraftwerk (0,55), rock/indie argentino
  (0,51).
- **Puentes** — los pasos entre valles: melodic techno (0,21), deep house (0,23),
  dance pop (0,23), art pop. Su *llegada* fuerza la mayor reorganización de la
  memoria (el estado neuronal salta **1,5×** más en un cambio de género que dentro
  del mismo).

La alegoría es literal y es la misma foto del §1: tu gusto es un **terreno de
valles** (metal, rock argentino, Kraftwerk) unidos por **pasos de montaña
electrónicos** (house/techno). El núcleo electrónico no es solo música que ponés:
es la *red de transporte* entre todo lo demás.

## 4. La gramática de flujo

La misma idea, explícita como grafo: los nodos son géneros, las flechas son
pases, y el tamaño del nodo es su puntaje de "hub" (PageRank).

![Cómo se pasan la posta los géneros. Un clúster argentino muy interconectado (naranja), una red electrónica enlazada (verde azulado) y puentes que los cruzan.](figures/taste-flow-es.pdf){width=88%}

- **Hubs** (donde un caminante pasaría el tiempo): alternative dance, argentine
  indie, downtempo. **Puentes** (cuellos de botella entre regiones, por
  intermediación): alternative dance y —curiosamente— **alternative metal**.
- **El metal te abre y te cierra las sesiones:** es tu forma más habitual tanto de
  **arrancar** (7,8%) como de **terminar** (7,5%) una sesión, pese a ser apenas el
  10% de ellas. Abrís y cerrás con él, y después te vas.
- Puentes con nombre: **Kraftwerk → Babasonicos** (electrónica alemana hacia rock
  argentino) y **Rammstein $\leftrightarrow$ Daft Punk / Grimes**.

## 5. Pozos gravitacionales — hacia dónde "rueda" el modelo

Pensá el modelo entrenado como un sistema dinámico: poné una semilla, dejá que
prediga el siguiente, saltá al tema real más cercano y repetí 30 veces. Donde se
asienta la órbita es una **cuenca de atracción** — literalmente, adónde rodaría una
bolita sobre el terreno del §1.

![Hacé rodar el modelo desde una semilla. Verde = se queda en casa; rojo = fluye hacia otra región. El largo de la barra es la proporción de los 30 pasos que pasa en el género destino.](figures/taste-attractors-es.pdf){width=88%}

- **Pozos profundos (de los que no se sale):** **Kraftwerk → 100% dusseldorf
  electronic**, **Aphex Twin → 100% ambient**, **Radiohead → 100% alternative
  rock**, **Rammstein → 70% metal**: arrancás ahí y el modelo se queda, igual que
  las sesiones cortas y autocontenidas de "bloque".
- **Trampolines (fluyen):** **Four Tet desemboca en ambient (90%)**; **The xx,
  DIIV y Underworld derivan hacia un sumidero de album-rock / alternative-rock.**
  Así que tu núcleo electrónico es donde *empiezan* los viajes; una cuenca genérica
  de rock/ambient es donde *terminan* los poco guiados.

## 6. Adentro del mecanismo — cómo decide quedarse o cambiar

La compuerta de actualización $z_t$ se puede leer exacta (reconstruir la
recurrencia desde los pesos reproduce el modelo a $5\times10^{-8}$). Dos cosas, una
de ellas un negativo limpio:

- **No hay "celdas de memoria" de prendido/apagado.** La fracción de retención
  $z_t$ se queda en **~0,51 para cada una de las 256 coordenadas** (rango
  0,42–0,63) y casi no se mueve: no existe una unidad que se *prenda* para "metal"
  y se *apague* para "house". La memoria está **distribuida**, no es un tablero de
  interruptores etiquetados.
- **Pero el estado se mueve 1,5× más en los cambios** — y esa reorganización es
  justo lo que miden los puentes del §3. Así que la conducta de "decidir cambiar"
  es real; solo que vive en el movimiento de todo el estado, no en ninguna celda
  interpretable.

## 7. ¿Coincide con lo que hacés en la práctica?

Como contraste contra los registros crudos (el único lugar donde entran los
conteos): el núcleo electrónico de mayor relevancia para el modelo es también tu
música **más escuchada del último año**; sus "bloques" (Rammstein, rock argentino)
son tus mayores totales **históricos**; y la cuenca de Rammstein —aislada y corta—
encaja con tu propia observación de que *hace rato que no lo ponés*: el modelo, que
aprende cronológicamente, **bajó la fase apagada** aunque el conteo de toda la vida
lo siga poniendo #1.

## En una línea

Desde adentro, tu gusto es un **núcleo** chico y pesado —~25 artistas que se
llevan un tercio de tus reproducciones, flotando sobre una cola ancha y fina de
~5.600 artistas que en su mayoría probaste una vez— y ese núcleo está dispuesto
como un **terreno**: valles profundos donde te instalás —Rammstein/metal
industrial (en
ráfagas cortas, aparte de todo), rock argentino en el linaje Cerati/Babasonicos,
Kraftwerk— unidos por una **red electrónica de pasos de montaña** (Four Tet,
Caribou, Aphex Twin, house/techno) que todos los métodos coinciden en señalar como
la verdadera estructura conectiva, sobre un núcleo dream-pop (DIIV, The xx); y,
como el modelo aprende en el tiempo, codifica el **presente electrónico** al que te
mudaste.

---

*Salvedades y método: las figuras salen de un modelo sobre una única partición
cronológica; la cantidad de átomos, la tasa de permanencia, $\lVert\Delta h\rVert$
y "atractor" son constructos interpretativos (un autoencoder disperso sobre el
estado oculto de la GRU, la matriz de transición de co-ocurrencia, la
reconstrucción de las compuertas y k-means sobre vectores de sesión); la huella
neuronal incorpora los nombres de artista, así que los conceptos tienden a tener
forma de artista; solo se representan conceptos frecuentes y predictivos, no toda
tu biblioteca. Método completo: `sae-interpretability-note.pdf`.*
