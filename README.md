# registres-ouverts

**Outillage publique et banc d'essai pour la lecture automatique et indexations de registres d'état civil anciens.**

Lire un registre manuscrit par IA est devenu courant ; publier des résultats dont le taux d’erreur est vérifiable reste rare, et les index existants sont souvent fermés ou payants. registres-ouverts met en libre accès son code, ses données et ses mesures, au service des généalogistes, des services d’archives et de la recherche.
`registres-ouverts` produit, à partir des images numérisées par les Archives départementales :

1. **Un jeu de référence ouvert** : des actes transcrits et vérifiés à la main, aux formats standard ALTO/PAGE, pour entraîner et évaluer n'importe quel modèle.
2. **Un banc d'essai** : les taux d'erreur par champ de plusieurs moteurs de reconnaissance d'écriture, mesurés sur le même échantillon.
3. **Un index complet et ouvert** : tous les champs de l'acte (enfant, date, parents, âges, professions, témoins), chaque valeur reliée à l'image d'origine avec son score de confiance.
4. **À terme, les liens entre actes** : naissance, mariage et décès d'une même personne.

> **Statut : en cours de cadrage.** Rien n'est encore utilisable.

---

## Pilote

| | |
| --- | --- |
| Commune | Manosque (Alpes-de-Haute-Provence) |
| Actes | Naissances |
| Période | 1873 – 1882 |
| Source des images | Archives départementales des Alpes-de-Haute-Provence (Licence Ouverte) |
| Objectif | ≥ 99 % de précision sur les actes validés automatiquement ; les autres partent en relecture humaine |

## Chaîne de traitement

```
Acquisition → Segmentation → Transcription (HTR) → Extraction des champs
                                                          │
                                       Contrôle (tables décennales, confiance)
                                               │ doute          │ validé
                                         Relecture humaine       │
                                               └──────► Base de données ──► Publication
```

- Le modèle de transcription est adapté à l'écriture des registres, à partir d'un modèle de base public.
- L'extraction des champs est contrainte par un schéma : chaque valeur doit figurer mot pour mot dans la transcription.
- Le nom et la date de chaque acte sont vérifiés automatiquement contre les tables décennales.

## Principes

1. **Ouvert.** Code libre, données réutilisables par tous, institutions comprises.
2. **Mesuré.** Chaque résultat est publié avec ses taux d'erreur et ses règles d'évaluation.
3. **Traçable.** Chaque donnée renvoie à l'image de l'acte original.
4. **Respectueux de la vie privée.** Seuls les actes de plus de 100 ans, déjà numérisés et publiés par les archives, sont traités.

## Feuille de route

- [ ] **Jalon 0 — Cadrage** : vérification de l'existant, chaîne complète sur 20 pages, choix du moteur de transcription
- [ ] **Jalon 1 — Prototype** : jeu de référence, banc d'essai et index des naissances de Manosque 1873-1882
- [ ] **Jalon 2 — Partenariat** : mariages et décès, collaboration avec les Archives départementales
- [ ] **Jalon 3 — Liens et recherche** : liens entre actes d'une même personne, recherche publique

## Contribuer

Les contributions sont bienvenues : code, relecture d'actes, retours d'archivistes, de paléographes ou de généalogistes.

Toute contribution au code nécessite la signature préalable d'un **accord de contribution**, afin de garantir la cohérence des licences du projet. Ouvrez une *issue* pour en discuter.

## Licences

- **Code** : [GNU AGPL-3.0](LICENSE)
- **Données produites** : [Licence Ouverte / Open Licence 2.0 (Etalab)](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)
- **Images des registres** : produites par les services d'archives, réutilisées selon leurs conditions, source citée pour chaque acte.

Copyright © 2026 Ziane Menni

## Contact

Ziane Menni — [portfolio.zianemenni.fr](https://portfolio.zianemenni.fr)
