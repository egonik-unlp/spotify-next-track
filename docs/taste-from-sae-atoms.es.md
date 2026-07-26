# Lo que los átomos del modelo dicen sobre tu gusto

*Una lectura de tu forma de escuchar, inferida a partir de los conceptos que el
modelo de "próxima canción" aprendió a representar. Es exploratorio — tómalo como
"en torno a qué está organizada la memoria del modelo", no como un veredicto.*

## Cómo se obtiene esta lectura (en términos simples)

El modelo que adivina tu próxima canción va armando una pequeña **memoria** a
medida que escucha una sesión. Por sí sola esa memoria es una maraña — miles de
números, ninguno con un significado propio. Así que hacemos dos cosas:

1. **Abrimos la memoria y la obligamos a ordenarse.** Entrenamos un pequeño
   ayudante (un "diccionario disperso") que debe volver a describir la memoria del
   modelo usando solo un puñado de **interruptores** internos a la vez. Obligado a
   ser económico, cada interruptor deja de ser un borrón difuso y pasa a
   representar *una cosa reconocible*.
2. **Leemos los interruptores.** Después miramos a qué reacciona cada interruptor,
   y casi todos coinciden con un **artista** o un **género** concreto — aquello
   cuyas canciones predice mejor como la que viene a continuación.

Entonces, las listas de abajo son, literalmente, **las cosas para las que el
modelo consideró que valía la pena tener un interruptor dedicado** con tal de
predecir qué reproducís después. Y cuando el modelo gasta *varios* interruptores
en un mismo artista, ese artista no solo es frecuente: le da forma a tus sesiones
de varias maneras distintas. No se guardan canciones; son los patrones.

Dos aclaraciones honestas: la "huella" del modelo incluye el **nombre del
artista**, así que los interruptores tienden a tener forma de artista; y estos son
los conceptos **frecuentes y predictivos**, no todo tu gusto — un favorito
aislado no se gana un interruptor. Es un mapa de lo que domina tu escucha, no un
juicio sobre ella.

*(Método, para quien tenga curiosidad: un autoencoder disperso top-k sobre el
estado oculto de la GRU de próxima canción, con k=32 átomos activos — ver
`sae-interpretability-note.pdf`. Acá se resumen los 80 átomos más fuertes de esa
corrida.)*

## El mapa de tu gusto

**1. Un núcleo electrónico profundo, guiado por productores — de lejos la región
más grande.** El modelo gasta acá la mayoría de sus interruptores, y abarcan todo
el rango de texturas en vez de un solo subgénero:

- **UK garage / house:** Joy Orbison (8 interruptores — empatado como el máximo de
  cualquier artista), Jamie xx, DJ Koze, Four Tet, Underworld
- **Sampledelia / plunderphonics:** The Avalanches (6 interruptores)
- **Trip-hop / downtempo / balearic:** Massive Attack, Thievery Corporation, Air,
  Tosca
- **Dance melódico / leftfield:** Caribou y su alias de house Daphni (8
  interruptores entre los dos)
- **IDM / pioneros de la electrónica:** Aphex Twin, Kraftwerk — más interruptores
  dedicados a **minimal techno**

Esta es la columna vertebral de tu escucha: no es "EDM", sino una paleta
electrónica de *cavador de bateas* — cálida, melódica, guiada por productores
(un linaje Warp / Ninja Tune / XL / Border Community).

**2. Un eje indie / dream-pop atmosférico.** El otro gran polo es de guitarras y
texturas: **DIIV** (7 interruptores), **The xx**, y una fuerte serie de
interruptores dedicados a **indie rock** (6). Más soñador y bañado en reverb que
ruidoso.

**3. Un hilo argentino fuerte y personal.** Distinto de los dos anteriores, y
claramente una parte grande de *tu* escucha en particular: **Laika Perra Rusa** se
gana la **mayor cantidad de interruptores de todos los artistas (8)**. Esta escena
local no es un detalle: es una de las regiones mejor representadas del modelo.

**4. Art-rock como puente, y un caso más pesado aparte.** **Radiohead** (más
interruptores de **album rock**) y **Gorillaz** se ubican entre los polos
electrónico e indie — tejido conectivo de art-rock que difumina géneros. Y un
hilo claramente distinto: **Rammstein** (3 interruptores) — un bolsón de metal
industrial que se separa de todo lo demás, y justamente por eso el modelo le da
sus propios interruptores.

## Tus anclas

Los artistas en los que el modelo gasta **más** capacidad — varios interruptores
distintos cada uno — son tus verdaderas anclas: los sonidos que se repiten a lo
largo de muchas sesiones y en más de un modo.

> **Laika Perra Rusa** y **Joy Orbison** (8 interruptores cada uno), **DIIV** (7),
> **The Avalanches** (6), luego **Massive Attack, Daphni, Caribou, Air,
> Thievery Corporation** (4 cada uno), **DJ Koze, Rammstein** (3).

Que un artista reciba muchos interruptores significa que ocupa varios submodos en
tu escucha — profundidad, no solo cantidad de reproducciones.

## En una línea

Tu gusto, tal como lo ve el modelo, es un **núcleo electrónico melódico guiado por
productores** (UK garage / house → downtempo → IDM) entrelazado con **indie /
dream-pop atmosférico**, que lleva un hilo **indie-rock argentino** fuerte y
personal, con **art-rock a lo Radiohead / Gorillaz** como puente — y un bolsón
aparte de **metal industrial a lo Rammstein**. Está anclado por un puñado de
artistas (Laika Perra Rusa, Joy Orbison, DIIV, The Avalanches) lo bastante
profundos como para ocupar varias dimensiones del modelo a la vez.
