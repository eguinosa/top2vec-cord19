# Using the Top2Vec Topic Model with the CORD-19 dataset.


## Introduccion

En este proyecto trabajamos con los documentos dentro de CORD-19 que contengan más de una página. Para garantizar que los documentos cumplan estas caracteristicas realizamos el proceso ya explicado en el proyecto 'LDA con CORD-19', donde solo ultilizas los documentos de CORD-19 que contengan 3,001 caracteres o más.

Para la implementacion del proyecto utilizamos la libreria del modelo de topicos Top2Vec, implementado en <https://github.com/ddangelov/Top2Vec>.

Para ultilizar este modelo de topicos los creadores del proyecto recomiendan utilizar la clase Top2Vec dentro de la libreria instalada usando:

  $ pip install top2vec

pero esta implementacion da error, sin poder encontrarle solucion a este problema, decidimos utilizar las otras dos implementaciones disponibles para Top2Vec, una utilizando un modelo pre-entrenado de Universal Sentence Encoder:

  $ pip install top2vec[sentence_encodores]

y otra utilizando un modelo BERT pre-entrenado, especificamente el modelo 'distiluse-base-multilingual-cased', este un modelo multilingüe de Universal Sentence Encoder para 15 lenguajes, Arabico, Chino, Holandes, Ingles, Frances, Aleman, Italiano, Koreano, Polaco, Portugues, Ruso, Español y Turco:

  $ pip install top2vec[sentence_transformers]


## Implementación

En el proyecto usamos varias de las clases y métodos utilizados en anterior proyecto 'LDA con CORD-19', dentro de estos
la clase Papers_Analizer() que separa los papers del CORD-19 por su tamaño, y nos entrega la cantidad de documentos grandes que deseemos.

### main.py

Es en 'main.py' donde llamamos y utilizamos los modelos embedded pre-entrenados para generar los embeddings conjuntos de los documentos y las palabras del CORD-19. La libreria Top2Vec brinda 3 opciones:

- universal-sentence-encoder
- universal-sentence-encoder-multilingual
- distiluse-base-multilingual-cased

Nosotros ultilizamos 'universal-sentence-encoder' y 'distiluse-base-multilingual-cased'. Para utilizar ambos modelos se debe llamar a la clase Top2Vec de la siguiente forma:

- Top2Vec(paper_texts, embedding_model='universal-sentence-encoder')
- Top2Vec(paper_texts, embedding_model='distiluse-base-multilingual-cased')

## Resultados

Para probar el modelo de tópicos Top2Vec tomamos una muestra de 3,000 documentos grandes de los papers disponibles dentro de CORD-19.

Los resultados obtenidos varían dependiendo de la muestra tomada, en algunos casos el numero de tópicos encontrados es grande, pero en otros casos el numero de tópicos es pequeño resultando en una gran concentración de papers dentro de un solo tópico.

En cuanto al rendimiento de los modelos utilizados ambos, BERT y Universal Sentence Encoder, tienen resultados similares con respecto al numero y la calidad de los topicos similares, siendo Universal Sentence Encoder mucho mas rapido en cuanto a la ejecución.

### Test Result 1 - 4 Tópicos
3,000 Random Papers
[0 h : 2 min : 6 sec : 573 mill]

Top2Vec found 4 topics.

Topic Sizes:
  Topic #0: 2881
  Topic #1: 51
  Topic #2: 37
  Topic #3: 31

Topic #0 Words:
[(0.29470897, 'bacteriophage'),
 (0.29097623, 'igm'),
 (0.26823637, 'retroviruses'),
 (0.26007655, 'antigens'),
 (0.25583568, 'igg'),
 (0.25344843, 'interferon'),
 (0.2519625, 'antiviral'),
 (0.2518179, 'unvaccinated'),
 (0.25033516, 'antigen'),
 (0.24626926, 'microbiology'),
 (0.2442199, 'virulence'),
 (0.24331522, 'encephalitis'),
 (0.24127308, 'bacteriophages'),
 (0.24105015, 'lymphocyte'),
 (0.23886725, 'neutrophil'),
 (0.23682985, 'immunized'),
 (0.23590967, 'staphylococcus'),
 (0.2342084, 'microorganism'),
 (0.23231453, 'sars'),
 (0.23143217, 'erythromycin')]

Topic #1 Words:
[(0.307088, 'wie'),
 (0.30480343, 'geht'),
 (0.3003263, 'keine'),
 (0.29598194, 'zum'),
 (0.29431742, 'auch'),
 (0.2936045, 'denn'),
 (0.29296622, 'dieses'),
 (0.28990754, 'richtig'),
 (0.2893827, 'aber'),
 (0.28932035, 'jetzt'),
 (0.2881305, 'nicht'),
 (0.28751588, 'gibt'),
 (0.28730744, 'genau'),
 (0.2841134, 'wegen'),
 (0.28332675, 'ihr'),
 (0.28308326, 'kein'),
 (0.28305984, 'immer'),
 (0.2819091, 'diese'),
 (0.28094625, 'welche'),
 (0.28010386, 'doch')]

Topic #2 Words:
[(0.3677531, 'latex'),
 (0.2680273, 'lemma'),
 (0.26364845, 'svm'),
 (0.25183085, 'usepackage'),
 (0.24908891, 'bayesian'),
 (0.2456662, 'classifiers'),
 (0.23587006, 'simplex'),
 (0.23309943, 'stata'),
 (0.22583783, 'classifier'),
 (0.22488171, 'vim'),
 (0.21707726, 'slides'),
 (0.21648356, 'homology'),
 (0.21390918, 'plt'),
 (0.21034974, 'pairwise'),
 (0.20765847, 'superscript'),
 (0.20126633, 'auc'),
 (0.20102882, 'elliptic'),
 (0.19989628, 'normalization'),
 (0.19930884, 'mcmc'),
 (0.19895214, 'confluence')]

Topic #3 Words:
[(0.3341459, 'puede'),
 (0.31878644, 'respuesta'),
 (0.31723252, 'otro'),
 (0.29805365, 'pueden'),
 (0.29762226, 'caso'),
 (0.28059542, 'informacion'),
 (0.27699804, 'estos'),
 (0.27400738, 'existen'),
 (0.25892663, 'resultados'),
 (0.25680283, 'candidiasis'),
 (0.25197896, 'estas'),
 (0.24951085, 'sobre'),
 (0.2485342, 'sintomas'),
 (0.24421525, 'tienen'),
 (0.24103782, 'menos'),
 (0.2326436, 'todos'),
 (0.22388175, 'deben'),
 (0.22369796, 'este'),
 (0.22283733, 'gastritis'),
 (0.22040346, 'meningitis')]

### Test Result 2 - 18 Tópicos - Universal Sentence Encoder
3,000 Papers
[0 h : 1 min : 58 sec : 532 mill]

Top2Vec found 18 topics.

Topic Sizes:
  Topic #0: 765
  Topic #1: 307
  Topic #2: 239
  Topic #3: 196
  Topic #4: 169
  Topic #5: 137
  Topic #6: 132
  Topic #7: 127
  Topic #8: 122
  Topic #9: 121
  Topic #10: 103
  Topic #11: 101
  Topic #12: 92
  Topic #13: 92
  Topic #14: 81
  Topic #15: 77
  Topic #16: 77
  Topic #17: 62

### Test Result 3 - 18 Tópicos - BERT
3,000 Papers (los utilizados en Test 2)
[0 h : 8 min : 21 sec : 595 mill]

Top2Vec found 18 topics.

Topic Sizes:
  Topic #0: 340
  Topic #1: 330
  Topic #2: 267
  Topic #3: 231
  Topic #4: 228
  Topic #5: 202
  Topic #6: 175
  Topic #7: 169
  Topic #8: 168
  Topic #9: 144
  Topic #10: 140
  Topic #11: 107
  Topic #12: 105
  Topic #13: 90
  Topic #14: 81
  Topic #15: 80
  Topic #16: 75
  Topic #17: 68
