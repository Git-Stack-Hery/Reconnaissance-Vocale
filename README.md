# Reconnaissance vocale par spectrogrammes

Projet de reconnaissance de la personne qui prononce « bonjour » à partir d’un enregistrement WAV. Le prototype distingue trois classes :

| Indice | Classe | Interprétation |
|---:|---|---|
| 0 | `Moi` | La voix du propriétaire du modèle |
| 1 | `Invites` | Une voix invitée représentée dans les données |
| 2 | `Autres` | Les autres sons et voix présents dans le jeu de données |

Le système utilise un MLP appliqué à des spectrogrammes Mel. Il s’agit d’un prototype pédagogique, et non d’un système biométrique validé pour un usage critique.

## Auteur

NOM : RAZAFIARISON Hery Solo

N° D'INSCRIPTION : 265-80-22

NIVEAU : M1 / AEII

## Fonctionnement

1. Les fichiers WAV de `Donnees/train` et `Donnees/test` sont convertis en images PNG par `Scripts/preparer_spectrogrammes.py`. Les PNG sont enregistrés dans `Donnees_MFCC`, avec les mêmes noms que les fichiers WAV.
2. Chaque WAV doit être mono-compatible et déjà échantillonné à 22 050 Hz; le convertisseur vérifie cette fréquence, ajuste le signal à 1,5 seconde, puis calcule un spectrogramme Mel de 64 bandes et 128 trames.
3. `entrainer_modele.py` normalise les pixels, crée une validation à partir du dossier train en conservant ensemble les variantes identifiées par leur nom de fichier, puis entraîne le MLP.
4. Le jeu test est évalué après la sélection du modèle. Le script affiche la perte, la précision, la matrice de confusion et les scores par classe. Il enregistre aussi les courbes et un rapport texte.
5. `Scripts/tester_modele.py` charge le modèle sauvegardé dans `Modele` et prédit la classe du fichier WAV indiqué dans `CHEMIN_AUDIO` (par défaut `Donnees/test/0_Moi/80_Moi.wav`).

### Architecture

- Réduction moyenne spatiale `2 × 2`, puis aplatissement du spectrogramme.
- Couches denses de 64 puis 32 neurones, avec activation ReLU.
- Dropout de `0,35` et `0,15`, régularisation L2, puis sortie softmax à trois classes.
- Optimiseur Adam, entropie croisée catégorielle sparse, arrêt anticipé et réduction du taux d’apprentissage lorsque la validation plafonne.

## Pourquoi un MLP ?

Le MLP a été choisi comme **baseline simple sur des spectrogrammes Mel de taille fixe**. Après une réduction moyenne `2 × 2`, le spectrogramme est aplati en un vecteur de 2 048 valeurs, puis traité par deux couches denses de 64 et 32 neurones. Cette approche fournit un premier point de comparaison direct pour une classification à trois classes.

Un CNN est aussi un candidat techniquement pertinent, et souvent mieux adapté aux spectrogrammes : ses filtres peuvent apprendre des motifs locaux dans le temps et en fréquence tout en partageant leurs paramètres. Le MLP, lui, ne conserve pas explicitement cette structure locale après l’aplatissement et comporte environ **133 315 paramètres**.

Un essai comparatif a ensuite été réalisé sur les spectrogrammes actuels, avec le même découpage train/validation/test. Le MLP existant a obtenu un F1 macro de validation de `0,7964` et un F1 macro de test de `0,8333`. Un CNN compact précis (8, 16 et 24 filtres, 5 627 paramètres) a obtenu `0,5216` en validation et `0,4864` sur le test. Pour ces données et cette configuration, le MLP est donc le choix retenu selon les résultats mesurés, et non parce qu’un MLP serait théoriquement supérieur.

Ce résultat ne permet pas d’affirmer que tous les CNN seraient moins bons : quelques essais seulement ont été évalués. Le CNN reste une architecture pertinente pour des spectrogrammes, car ses convolutions apprennent des motifs locaux dans le temps et les fréquences. Une comparaison plus générale demanderait plusieurs configurations et plusieurs graines aléatoires.

## Organisation

```text
ReconnaissanceVocale/
├── Donnees/
│   ├── train/              # WAV d’entraînement, rangés par classe
│   └── test/               # WAV de test, rangés par classe
├── Donnees_MFCC/
│   ├── train/              # Spectrogrammes PNG d’entraînement
│   └── test/               # Spectrogrammes PNG de test
├── Modele/
│   ├── modele_reconnaissance_vocale.h5
│   ├── courbes_entrainement.png
│   └── resultats_entrainement.txt
├── Scripts/
│   ├── preparer_spectrogrammes.py
│   ├── entrainer_modele.py
│   └── tester_modele.py
└── README.md
```

Les chemins utilisés par les scripts sont construits relativement à la racine du projet, et ne dépendent pas du répertoire courant du terminal. Les PNG de `Donnees_MFCC` sont des fichiers générés. Le convertisseur supprime les anciens PNG lorsqu’ils ne correspondent plus à un WAV source; il faut donc le relancer après toute modification des dossiers WAV.

## Installation et exécution

Les scripts nécessitent Python, TensorFlow et Matplotlib. Depuis un terminal, se placer dans le dossier du projet :

```text
python -m pip install tensorflow matplotlib
python Scripts/preparer_spectrogrammes.py
python Scripts/entrainer_modele.py
```

Pour tester un audio avec le modèle entraîné, modifier `CHEMIN_AUDIO` au début de `Scripts/tester_modele.py` (chemin relatif à la racine du projet), puis exécuter :

```text
python Scripts/tester_modele.py
```

Le modèle est enregistré au format HDF5 (`.h5`) dans `Modele/modele_reconnaissance_vocale.h5`. Keras signale que HDF5 est un format historique, mais ce format est conservé pour compatibilité avec les habitudes de projet.

## Résultat de référence

Lors de la comparaison MLP/CNN, le test contenait 60 spectrogrammes, soit 20 par classe. Le MLP a obtenu :

- Accuracy : **83,33 %**
- F1 macro : **0,8337**
- Accuracy de validation : **79,17 %**
- F1 macro de validation : **0,7964**

La matrice de confusion du test MLP était :

| Classe réelle | Prédit `Moi` | Prédit `Invites` | Prédit `Autres` |
|---|---:|---:|---:|
| `Moi` | 18 | 0 | 2 |
| `Invites` | 1 | 17 | 2 |
| `Autres` | 4 | 1 | 15 |

Les résultats peuvent varier si les fichiers ou la séparation train/test changent. La validation est également sensiblement plus faible que le test; davantage d’enregistrements réels et une séparation par personne sont nécessaires pour améliorer la capacité.

## Fichiers produits

- `Modele/modele_reconnaissance_vocale.h5` : modèle entraîné.
- `Modele/courbes_entrainement.png` : évolution de la perte et de la précision en entraînement et validation.
- `Modele/resultats_entrainement.txt` : résultats des entraînements, matrices de confusion et métriques par classe.
