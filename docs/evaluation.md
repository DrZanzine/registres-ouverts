# Évaluation

Le module `registres_ouverts.evaluate` mesure la qualité des sorties de la
chaîne par rapport au jeu de référence. Il est écrit en Python pur (aucune
dépendance) pour que chaque chiffre publié soit reproductible : mêmes données,
mêmes règles, même script.

## Transcription : CER et WER

- **Distance d'édition** (Levenshtein) : nombre minimal de substitutions,
  insertions et suppressions pour passer de la référence à l'hypothèse. Chaque
  opération coûte 1. L'alignement est conservé pour l'analyse d'erreurs.
- **CER** (taux d'erreur par caractère) = distance en caractères / nombre de
  caractères de la référence.
- **WER** (taux d'erreur par mot) = même calcul sur les mots, séparés par les
  espaces.
- Le taux peut dépasser 100 % si l'hypothèse ajoute beaucoup de texte.
- **Référence vide** : taux de 0 si l'hypothèse est vide aussi, *indéfini*
  sinon ; les insertions sont tout de même comptées dans le total.
- **Total global (micro)** : somme des distances / somme des longueurs de
  référence, sur toutes les lignes et toutes les pages. Ce n'est pas la
  moyenne des taux par ligne, qui donnerait autant de poids à une ligne de
  trois caractères qu'à une ligne entière.

### Formats et appariement

Référence et hypothèse peuvent être :

- un fichier texte `.txt` en UTF-8 : une ligne du fichier par ligne de
  l'image ;
- un fichier PAGE XML `.xml` (toutes versions, dont celle produite par
  kraken) : pour chaque `TextLine`, dans l'ordre du document, le texte de son
  `TextEquiv/Unicode` direct (celui des `Word` est ignoré ; s'il y a plusieurs
  `TextEquiv`, celui d'`index` le plus petit).

On peut comparer deux fichiers, ou deux dossiers : les pages sont alors
appariées par nom de fichier sans extension (`vue_0002.txt` ↔
`vue_0002.xml`).

Les lignes sont appariées **par identifiant** quand les deux côtés ont des
identifiants uniques (PAGE XML), sinon **par ordre** (option `--par-ordre`
pour l'imposer). Une ligne de référence sans correspondant est **manquante**
(comptée comme entièrement supprimée) ; une ligne d'hypothèse sans
correspondant est **en trop** (comptée comme entièrement insérée). Une page de
référence sans hypothèse est comptée comme entièrement manquante ; une page
d'hypothèse sans référence est signalée mais non comptée. Tous ces cas sont
signalés dans le rapport.

Le rapport liste aussi les **confusions** les plus fréquentes (« n » lu « u »,
accent omis…), calculées sur les lignes appariées seulement.

## Normalisation

La normalisation est appliquée identiquement à la référence et à
l'hypothèse avant toute mesure. Le rapport indique toujours celle qui a été
appliquée (`Normalisation : nfc, casse…`, ou clé `normalisation` en JSON).

| Option | Effet | Par défaut |
| --- | --- | --- |
| — | Unicode **NFC** : « é » composé et « e » + accent combinant sont identiques | toujours |
| `--casse` | ignore la casse (`casefold`) | non |
| `--diacritiques` | supprime accents, cédilles, trémas (« é » → « e ») ; les ligatures « œ » restent | non |
| `--ponctuation` | supprime la ponctuation Unicode (`.,;:'-«»`…) sans la remplacer : « l'an » → « lan » | non |
| `--espaces` | réduit les suites d'espaces à une seule, supprime celles du début et de la fin | non |

Ordre d'application : NFC, casse, diacritiques, ponctuation, espaces. Les
options sont désactivées par défaut parce qu'elles masquent des erreurs
réelles ; tout chiffre publié doit préciser celles qui ont été utilisées.

## Extraction : précision, rappel et F1 par champ

Les actes sont des fichiers JSON conformes à `configs/acte.schema.json`. Un
champ est tout objet `{brut, normalise, confiance}` (`enfant.nom`,
`pere.age`, `temoins[1].profession`…). La valeur comparée est `normalise`, à
défaut `brut`, convertie en texte puis normalisée comme ci-dessus ; une valeur
`null` ou vide est **absente**.

| Référence | Hypothèse | Issue | Compte |
| --- | --- | --- | --- |
| présente | égale | `correct` | vrai positif |
| présente | différente | `faux` | faux positif **et** faux négatif |
| présente | absente | `manquant` | faux négatif |
| absente | présente | `en_trop` | faux positif |
| absente | absente | `absent` | non compté |

- **Précision** = VP / (VP + FP) : part des valeurs produites qui sont justes.
- **Rappel** = VP / (VP + FN) : part des valeurs de référence retrouvées.
- **F1** = 2 × P × R / (P + R).
- *Indéfini* quand le dénominateur est nul.

Les témoins sont regroupés sous un même nom (`temoins.nom`), appariés par
rang. Dans des dossiers, les actes sont appariés par `id` (si tous en ont un)
ou par nom de fichier. Un acte de référence sans hypothèse compte tous ses
champs comme manquants. La liste complète des écarts figure dans la sortie
JSON.

## Courbe de seuil

À partir de couples (confiance, correct), pour chaque seuil *s* :

- les valeurs de confiance ≥ *s* sont **acceptées** (validées
  automatiquement), les autres partent en relecture ;
- **couverture** = acceptées / total ;
- **précision** = justes parmi les acceptées.

Le **seuil retenu** est le plus petit seuil dont la précision atteint la
cible (99 % par défaut) : c'est celui qui envoie le moins de valeurs en
relecture. Sans `--pas`, les seuils sont les confiances observées (courbe
exacte).

Les couples viennent d'un CSV (`confiance,correct`, avec `correct` en 1/0,
vrai/faux, true/false ou oui/non), d'un JSON (liste de
`{"confiance": …, "correct": …}` ou de paires), ou directement d'actes avec
`--actes REFERENCE HYPOTHESE` : chaque valeur produite avec une confiance est
correcte si elle est égale à la référence.

## Commandes

```sh
# CER/WER d'un dossier de pages, détail par ligne
uv run python -m registres_ouverts.evaluate cer data/reference/pages outputs/kraken --lignes

# Même mesure, sans tenir compte de la casse ni des accents, en JSON
uv run python -m registres_ouverts.evaluate cer ref.txt hyp.xml --casse --diacritiques --json

# Échec (code 1) si le CER global dépasse 5 %
uv run python -m registres_ouverts.evaluate cer ref/ hyp/ --max 0.05

# Précision, rappel et F1 par champ ; export des couples de confiance
uv run python -m registres_ouverts.evaluate champs ref_actes/ sortie_actes/ \
    --export-confiances outputs/confiances.csv

# Courbe de seuil et seuil retenu pour 99 % de précision
uv run python -m registres_ouverts.evaluate seuil outputs/confiances.csv --cible 0.99
uv run python -m registres_ouverts.evaluate seuil --actes ref_actes/ sortie_actes/ --pas 0.05
```

Codes de sortie : `0` succès, `1` critère non atteint (`cer --max`),
`2` erreur d'usage ou fichier illisible.

Exemple de sortie (données fictives) :

```
Normalisation : nfc
Pages : 1
CER : 27.16 %  (22 erreurs / 81 caractères ; sub. 2, ins. 0, sup. 20)
WER : 46.15 %  (6 erreurs / 13 mots ; sub. 3, ins. 0, sup. 3)
  ! ligne manquante dans l'hypothèse : n° 3

Confusions les plus fréquentes (référence → hypothèse) :
  «,» → ∅ : 1
  «é» → «e» : 1
  «à» → «a» : 1
```
