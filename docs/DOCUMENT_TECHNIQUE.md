Document technique — registres-ouverts
17 nov. 2026 · @Ziane
La chaîne transforme des images de registres d'état civil du XIXe siècle en données structurées, vérifiables et mesurées. Chaque brique existe déjà en open source : le travail consiste à les assembler, à les adapter à un fonds réel (naissances de Manosque, 1873-1882) et à évaluer rigoureusement le résultat.
Architecture
Les actes validés vont directement en base ; les autres passent par la relecture, dont les corrections réentraînent le modèle de transcription. Le module d'évaluation compare toutes les sorties au jeu de référence.
Modules
Huit modules indépendants, chacun testable seul et remplaçable ; la segmentation est le plus incertain.
Module
Rôle
Outil
Entrée → sortie
Difficulté
acquire
Télécharger les images et leurs références (cote, vue)
Python (requests) ou envoi en lot par les archives
URL ou lot → images + manifeste
Faible
segment
Découper chaque page en actes puis en lignes
kraken (détection des lignes de base) + règles sur le numéro d'acte en marge
Image → zones d'actes et lignes (PAGE XML)
Élevée
transcribe
Transcrire chaque ligne
kraken, modèle de base CATMuS Modern affiné sur les mains de Manosque
Lignes → texte + confiance par caractère
Moyenne
extract
Extraire les champs de l'acte
Règles (expressions régulières) et LLM local 7B via Ollama, sortie JSON contrainte
Texte de l'acte → champs structurés
Moyenne
check
Calculer le score de confiance
Python : tables décennales, accord règles/LLM, confiance HTR
Champs → score + statut validé / à relire
Faible
review
Faire relire les cas douteux
Export vers eScriptorium ou tableur, réimport des corrections
Actes douteux → actes corrigés
Faible
store
Stocker et publier
PostgreSQL ; exports CSV/JSON et ALTO/PAGE
Actes → base + fichiers ouverts
Faible
evaluate
Mesurer la qualité
Python (jiwer pour le CER) + métriques par champ
Sorties + jeu de référence → rapport chiffré
Moyenne
Garde-fou de l'extraction : toute valeur produite par le LLM doit apparaître dans la transcription (correspondance exacte ou approchée). Sinon le champ est rejeté et l'acte part en relecture.
Boucle d'amélioration : les corrections de la relecture enrichissent le jeu d'entraînement, et le modèle HTR est réaffiné régulièrement.
Modèle de données
Un acte est un document JSON ; chaque champ porte sa valeur lue, sa valeur normalisée et sa confiance, et l'acte renvoie à son image.
{
  "id": "59168-N-1879-035",
  "source": {"archives": "AD59", "cote": "à compléter", "vue": 12, "zone": [120, 340, 1480, 2210]},
  "type": "naissance",
  "numero_acte": 35,
  "date_acte": "1879-05-13",
  "enfant": {"nom": "Druon", "prenoms": "Jean Baptiste", "sexe": "M", "date_naissance": "1879-05-13T05:30"},
  "pere": {"nom": "Druon", "prenoms": "Jean Baptiste", "age": 29, "profession": "tisserand"},
  "mere": {"nom": "Dufetel", "prenoms": "Edoxie Célestine", "age": 24, "profession": "tisseuse"},
  "temoins": [
    {"nom": "Meurisse", "prenoms": "Bernard Clovis", "age": 30, "profession": "officier de santé"}
  ],
  "mentions_marginales": {"presence": true},
  "controle": {"table_decennale": "concordant", "accord_regles_llm": 0.94, "confiance_htr": 0.97, "statut": "valide"}
}
L'exemple reprend un acte de Cysoing (1879) déjà transcrit, pour illustration. Chaque champ conserve aussi la chaîne brute lue, utile pour l'évaluation et la traçabilité.
Méthode d'évaluation
Tout chiffre publié est reproductible : mêmes données, mêmes règles, même script.
• Jeu de référence : environ 50 actes, construits en corrigeant à la main une sortie automatique, validés par un généalogiste. Il sert uniquement à l'évaluation, jamais à l'entraînement.
• Transcription : taux d'erreur par caractère (CER) et par mot (WER), par moteur, sur les lignes du jeu de référence.
• Extraction : précision et rappel par champ (nom, prénoms, dates, parents, âges, professions, témoins), après normalisation.
• Normalisation fixée avant toute mesure et publiée : abréviations (« J. Bte » = « Jean-Baptiste »), accents, majuscules, graphies anciennes.
• Validation automatique : courbe précision / part d'actes validés selon le seuil de confiance ; seuil retenu pour ≥ 99 % de précision.
• Contrôle global : concordance nom + date avec les tables décennales sur l'ensemble des actes.
• Comparaison externe : mêmes métriques sur les index existants (Filae, Geneanet, FamilySearch), quand ils sont consultables pour l'échantillon.
Limite assumée : 50 actes donnent une marge d'erreur de plusieurs points ; les résultats sont publiés avec leur intervalle de confiance.
Environnement technique et organisation du dépôt
Tout tourne dans une VM dédiée du homelab, isolée du reste : c'est là que vivent le code, les données et Claude Code.
Élément
Choix
Machine
VM Ubuntu dédiée, GTX 1070 8 Go en passthrough (VM console éteinte pendant les traitements)
Langage
Python 3.12, dépendances gérées par uv
Transcription
kraken (segmentation + HTR)
Extraction
Règles Python + Ollama (modèle 7B quantifié, sortie JSON contrainte)
Base
PostgreSQL dans Docker
Qualité
pytest, ruff ; GitHub Actions sur chaque push
Entraînement lourd
GPU OVH loué à l'heure si la GTX 1070 est trop lente
Assistant de code
Claude Code dans la VM, guidé par un CLAUDE.md local, non versionné
Arborescence
registres-ouverts/
├── README.md, LICENSE, pyproject.toml
├── src/registres_ouverts/
│   ├── acquire/  segment/  transcribe/  extract/
│   └── check/  review/  store/  evaluate/
├── configs/        # schéma JSON d'un acte, règles de normalisation
├── data/
│   ├── raw/        # images téléchargées (non versionnées)
│   └── reference/  # jeu de référence (versionné)
├── models/         # modèles entraînés (non versionnés)
├── tests/
├── docs/
└── .github/workflows/ci.yml
Conventions
• En-tête de chaque fichier source : Copyright (C) 2026 Ziane Menni et SPDX-License-Identifier: AGPL-3.0-or-later.
• Branche main protégée : tout passe par une branche et une demande de fusion, validée par le porteur du projet.
• Les images et les modèles ne sont jamais versionnés ; un script les récupère ou les régénère.
• Chaque module s'exécute seul en ligne de commande, et la chaîne complète avec une seule commande.
Points durs et parades
Deux difficultés réelles, la segmentation et les noms propres ; les autres sont maîtrisables.
Point dur
Pourquoi
Parade
Segmentation des doubles pages
Deux actes par page, marges, mentions ajoutées d'une autre main, parfois tournées
S'appuyer sur le numéro et le nom d'acte en marge ; corriger à la main les 20 premières pages pour mesurer le problème
Noms propres
Les formules se lisent bien, les erreurs tombent sur les noms et professions
Modèle affiné sur les mains de la décennie ; contrôle par les tables décennales
LLM qui invente ou normalise
Un modèle de langage « corrige » volontiers un nom rare
Valeur obligatoirement présente dans la transcription ; comparaison avec les règles
Confiance mal calibrée
Un score élevé n'est pas une garantie
Seuil fixé par mesure sur le jeu de référence, pas à l'intuition
Mentions marginales
Autre main, autre encre, ajouts tardifs
2027 : détection seulement ; transcription reportée
Mémoire de la GTX 1070
8 Go : suffisant pour l'inférence, lent pour l'entraînement
Entraînement ponctuel sur GPU loué à l'heure
Choix du moteur HTR
Transkribus rapide mais propriétaire, kraken ouvert mais plus de travail
Décision fin décembre sur mesure, un seul moteur ensuite
Plan de développement du jalon 0
Objectif fin décembre 2026 : une chaîne qui tourne de bout en bout sur 20 pages, même imparfaite.
[ ] Créer la VM de traitement (Ubuntu, GPU en passthrough, Docker, uv, Ollama)
[ ] Cloner le dépôt dans la VM, ajouter un CLAUDE.md local (exclu de Git), pyproject.toml, l'arborescence et la CI
[ ] acquire : télécharger 20 pages de naissances de Manosque 1879 avec leurs références
[ ] segment + transcribe : premier passage avec un modèle de base public, sans affinage
[ ] extract : première version à base de règles, puis version LLM
[ ] evaluate : calcul du CER et de la précision par champ sur une dizaine d'actes corrigés à la main
[ ] Tester un second moteur (Transkribus) sur les mêmes pages et comparer
[ ] Note de bilan : moteur retenu, erreurs observées, priorités du jalon 1