# Modèles HTR utilisés

Les modèles ne sont jamais versionnés : ils sont téléchargés dans `models/` (ignoré par Git).
Cette page note leur origine, leur licence et leur empreinte, pour que chaque résultat soit reproductible.

Vérifié le 17 novembre 2026 sur Zenodo (API `https://zenodo.org/api/records/<id>`) : la version citée est la dernière publiée.

## Segmentation

| | |
| --- | --- |
| Modèle | `blla.mlmodel`, modèle de segmentation par défaut livré avec kraken |
| Version | kraken 7.1.1 |
| Licence | celle de kraken (Apache-2.0) |
| Téléchargement | aucun, inclus dans le paquet |

## Transcription

### McCATMuS

| | |
| --- | --- |
| Titre | McCATMuS - Transcription model for handwritten, printed and typewritten documents from the 16th century to the 21st century |
| Auteur | Alix Chagué (ALMAnaCH, Inria ; Université de Montréal ; EPHE) |
| DOI | [10.5281/zenodo.13788177](https://doi.org/10.5281/zenodo.13788177) (version), concept 10.5281/zenodo.13788176 |
| Publication | 19 septembre 2024 |
| Licence | **CC-BY-4.0** (Zenodo et `metadata.json` du modèle) |
| Fichier | `models/mccatmus/McCATMuS_nfd_nofix_V1.mlmodel` (16 173 802 octets) |
| MD5 | `e531463f631303c700750784b4f9ed63` (identique à Zenodo) |
| Données d'entraînement | 22 jeux de données au format CATMuS (surtout du français), du XVIe siècle à 2023, manuscrits, imprimés et dactylographiés ; jeu [CATMuS/modern](https://huggingface.co/datasets/CATMuS/modern) |
| Exactitude annoncée | 92,8 % (sur le jeu de test de l'auteur, pas sur nos registres) |
| Normalisation Unicode | NFD |

C'est le modèle que le document technique appelle « CATMuS Modern ».

### LECTAUREP Contemporary French Model (Administration)

| | |
| --- | --- |
| Titre | LECTAUREP Contemporary French Model (Administration) |
| Auteur | Alix Chagué (ALMAnaCH, Inria), données d'Aurélia Rostaing, Françoise Limon-Bonnet, Nathalie Denis et Marc Durand |
| DOI | [10.5281/zenodo.6542744](https://doi.org/10.5281/zenodo.6542744) |
| Publication | 12 mai 2022, version 1.0.0 |
| Licence | **CC-BY-4.0** (Zenodo et `metadata.json` du modèle) |
| Fichier | `models/lectaurep/lectaurep_base.mlmodel` (16 120 811 octets) |
| MD5 | `f6c7f613931dce656a163756eb2b56de` (identique à Zenodo) |
| Données d'entraînement | Documents administratifs français de 1742 à 1928 (projet LECTAUREP, Inria et Archives nationales) : répertoires de notaires, actes de mariage et de divorce ; 19 364 lignes |
| Exactitude annoncée | 90,9 % (sur 12 pages de test de l'auteur, pas sur nos registres) |
| Normalisation Unicode | NFD |
| Conventions | Abréviations non développées, majuscules d'époque, exposants signalés par `^`, signatures transcrites `¥` |

## Compatibilité avec kraken 7

Les deux modèles ont été entraînés avec d'anciennes versions de kraken (4.x).
Ils se chargent sans erreur avec `kraken.tasks.RecognitionTaskModel.load_model` dans kraken 7.1.1 (Windows, Python 3.12, torch CPU).

## Attribution (CC-BY-4.0)

Toute publication de résultats produits avec ces modèles cite le modèle utilisé, son auteur et son DOI.

## Récupérer les modèles

```sh
mkdir -p models/mccatmus models/lectaurep
curl -L -o models/mccatmus/McCATMuS_nfd_nofix_V1.mlmodel \
  https://zenodo.org/api/records/13788177/files/McCATMuS_nfd_nofix_V1.mlmodel/content
curl -L -o models/lectaurep/lectaurep_base.mlmodel \
  https://zenodo.org/api/records/6542744/files/lectaurep_base.mlmodel/content
md5sum models/*/*.mlmodel
```
