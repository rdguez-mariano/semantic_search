PROMPT_ANSWER_FROM_DOCS_WITH_SOURCES = """\
En utilisant les informations contenues dans les documents ci-dessous, donnez \
une réponse précise à la question. Ne répondez qu'à la question posée, \
la réponse doit être concise et pertinente par rapport à la question. \
Finisez par fournir la liste des documents qui ont servis à votre réponse où \
les premiers élements sont les plus pertinents. \
S'il n'est pas possible de répondre à la \
question en utilisant les informations fournies, répondez par \
'Je ne trouve pas la réponse' et une liste vide. \
La liste doit suivre le format suivant.
```
Documents utilisés:
- Document 7
- Document 1
- Document 3
...
- Document N
```


{context}

---
Voici maintenant la question à laquelle vous devez répondre.

Question: {question}

Réponse:
"""

PROMPT_EXTRACT_EXACT_ANSWER_FROM_DOCS = """
En utilisant les documents ci-dessous créez une liste qui servirait à \
formuler une réponse courte et claire à la question. \
Chaque élément de la liste est le résultat d'un copier-coller exact \
d'une ou plusieurs lignes consécutives d'un des documents. \
S'il n'est pas possible de répondre à la question en utilisant les \
informations fournies, répondez par une liste vide. \
La liste suit le format:
- élément 1
- élément 2
...
- élément N


{context}

---
Voici maintenant la question pour laquelle vous devez créer la liste.

Question: {question}

Réponse:
"""

PROMPT_FRENCH_GRADER = """Vous êtes un examinateur chargé \
d'évaluer la pertinence d'un document par rapport à une question \
posée par un utilisateur.

Voici le document:

---

{context}

---

Voici la question de l'utilisateur : {question}


Si le document contient des mots-clés liés à la question de l'utilisateur, \
évaluez-le comme étant pertinent. \
Il n'est pas nécessaire que ce test soit rigoureux. \
L'objectif est de filtrer les documents erronés. \
Donner un score binaire 'oui' ou 'non' pour indiquer si le document \
est pertinent par rapport à la question. \
Répondez oui ou non, et rien d'autre.


Réponse:
"""

PROMPT_FRENCH_GENERATE_QUESTIONS = """Générez 3 questions sur des fait \
ou des concepts qui trouveront leur réponses dans les documents ci-dessous.


---

{context}

---


Soyez original, précis et évitez les questions de connaissance publique. \
Chaque question doit tenir en une phrase. \
Les questions supposent que le lecteur ne connaîtra pas les documents. \
Répondez seulement avec la liste de vos questions proposées. \
{format_instructions}

Réponse:
"""
