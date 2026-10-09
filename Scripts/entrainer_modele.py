from pathlib import Path
from datetime import datetime
import random
import re

import matplotlib.pyplot as plt
import tensorflow as tf

layers = tf.keras.layers


# =========================================================
# PARAMÈTRES
# =========================================================

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_DONNEES = DOSSIER_PROJET / "Donnees_MFCC"
DOSSIER_TRAIN = DOSSIER_DONNEES / "train"
DOSSIER_TEST = DOSSIER_DONNEES / "test"
DOSSIER_MODELE = DOSSIER_PROJET / "Modele"
CHEMIN_MODELE = DOSSIER_MODELE / "modele_reconnaissance_vocale.h5"
CHEMIN_COURBES = DOSSIER_MODELE / "courbes_entrainement.png"
CHEMIN_RAPPORT = DOSSIER_MODELE / "resultats_entrainement.txt"

CLASSES = ["0_Moi", "1_Invites", "2_Autres"]
TAILLE_IMAGE = (64, 128)
BATCH_SIZE = 16
EPOCHS = 150
PART_VALIDATION = 0.20
GRAINE_ALEATOIRE = 42
CONFIGURATIONS_MLP = [
    {
        "nom": "mlp_dropout_0.35_0.15",
        "type": "MLP",
        "dropout_1": 0.35,
        "dropout_2": 0.15,
    },
]


# =========================================================
# PRÉPARATION DES DONNÉES
# =========================================================

def trouver_fichiers_par_classe(dossier):
    fichiers_par_classe = {}

    for nom_classe in CLASSES:
        dossier_classe = dossier / nom_classe
        if not dossier_classe.is_dir():
            raise FileNotFoundError(f"Dossier de classe introuvable : {dossier_classe}")

        fichiers = sorted(
            chemin for chemin in dossier_classe.iterdir()
            if chemin.is_file() and chemin.suffix.lower() == ".png"
        )
        if not fichiers:
            raise FileNotFoundError(f"Aucun spectrogramme PNG dans : {dossier_classe}")

        fichiers_par_classe[nom_classe] = fichiers

    return fichiers_par_classe


def separer_train_validation(fichiers_par_classe):
    generateur = random.Random(GRAINE_ALEATOIRE)
    chemins_train = []
    etiquettes_train = []
    chemins_validation = []
    etiquettes_validation = []

    for indice_classe, nom_classe in enumerate(CLASSES):
        fichiers = fichiers_par_classe[nom_classe]
        groupes = {}
        for chemin in fichiers:
            nom_source = re.sub(r"_(?:var\d+|orig|ajustee)$", "", chemin.stem)
            groupes.setdefault(nom_source, []).append(chemin)

        noms_groupes = list(groupes)
        generateur.shuffle(noms_groupes)
        cible_validation = max(1, round(len(fichiers) * PART_VALIDATION))
        groupes_validation = set()
        nb_validation = 0

        for nom_groupe in noms_groupes[:-1]:
            if nb_validation >= cible_validation:
                break
            groupes_validation.add(nom_groupe)
            nb_validation += len(groupes[nom_groupe])

        for nom_groupe, fichiers_groupe in groupes.items():
            if nom_groupe in groupes_validation:
                chemins_validation.extend(str(chemin) for chemin in fichiers_groupe)
                etiquettes_validation.extend([indice_classe] * len(fichiers_groupe))
            else:
                chemins_train.extend(str(chemin) for chemin in fichiers_groupe)
                etiquettes_train.extend([indice_classe] * len(fichiers_groupe))

    return chemins_train, etiquettes_train, chemins_validation, etiquettes_validation


def charger_spectrogramme(chemin, etiquette):
    contenu = tf.io.read_file(chemin)
    image = tf.io.decode_png(contenu, channels=1)
    image = tf.ensure_shape(image, [TAILLE_IMAGE[0], TAILLE_IMAGE[1], 1])
    return image, etiquette


def creer_dataset(chemins, etiquettes, melanger=False):
    dataset = tf.data.Dataset.from_tensor_slices((chemins, etiquettes))
    dataset = dataset.map(charger_spectrogramme, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()

    if melanger:
        dataset = dataset.shuffle(buffer_size=len(chemins), seed=GRAINE_ALEATOIRE)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# =========================================================
# MODÈLE MLP
# =========================================================

def creer_modele(configuration):
    regularisation = tf.keras.regularizers.l2(1e-4)

    return tf.keras.Sequential([
        layers.Input(shape=(TAILLE_IMAGE[0], TAILLE_IMAGE[1], 1)),

        # Réduire les paramètres sans utiliser de convolution
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),

        # Couches denses avec régularisation contre le surapprentissage
        layers.Dense(64, activation="relu", kernel_regularizer=regularisation),
        layers.Dropout(configuration["dropout_1"]),
        layers.Dense(32, activation="relu", kernel_regularizer=regularisation),
        layers.Dropout(configuration["dropout_2"]),

        # Une sortie par classe : Moi, Invités, Autres
        layers.Dense(len(CLASSES), activation="softmax"),
    ])


def creer_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
    ]


def afficher_statistiques_et_courbes(historique):
    valeurs = historique if isinstance(historique, dict) else historique.history
    pertes_validation = valeurs["val_loss"]
    meilleur_indice = min(
        range(len(pertes_validation)), key=pertes_validation.__getitem__
    )
    meilleure_epoque = meilleur_indice + 1

    print("\n===== MEILLEURE EPOQUE DE VALIDATION =====")
    print(f"Epoque : {meilleure_epoque}")
    print(f"Loss entrainement : {valeurs['loss'][meilleur_indice]:.4f}")
    print(f"Accuracy entrainement : {valeurs['accuracy'][meilleur_indice]:.4f}")
    print(f"Loss validation : {pertes_validation[meilleur_indice]:.4f}")
    print(f"Accuracy validation : {valeurs['val_accuracy'][meilleur_indice]:.4f}")

    epoques = range(1, len(valeurs["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epoques, valeurs["loss"], label="Entrainement")
    axes[0].plot(epoques, pertes_validation, label="Validation")
    axes[0].axvline(meilleure_epoque, color="gray", linestyle="--")
    axes[0].set_title("Evolution de la perte")
    axes[0].set_xlabel("Epoque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoques, valeurs["accuracy"], label="Entrainement")
    axes[1].plot(epoques, valeurs["val_accuracy"], label="Validation")
    axes[1].axvline(meilleure_epoque, color="gray", linestyle="--")
    axes[1].set_title("Evolution de la precision")
    axes[1].set_xlabel("Epoque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(CHEMIN_COURBES, dpi=150)
    print(f"\nCourbes sauvegardees : {CHEMIN_COURBES.name}")
    plt.show()
    plt.close(figure)


def calculer_statistiques(etiquettes, probabilites):
    predictions = tf.argmax(probabilites, axis=1, output_type=tf.int32).numpy()
    matrice = tf.math.confusion_matrix(
        etiquettes,
        predictions,
        num_classes=len(CLASSES),
    ).numpy()

    statistiques = []

    for indice, nom_classe in enumerate(CLASSES):
        vrais_positifs = int(matrice[indice, indice])
        faux_positifs = int(matrice[:, indice].sum() - vrais_positifs)
        faux_negatifs = int(matrice[indice, :].sum() - vrais_positifs)
        support = int(matrice[indice, :].sum())

        precision = vrais_positifs / (vrais_positifs + faux_positifs) if vrais_positifs + faux_positifs else 0.0
        rappel = vrais_positifs / support if support else 0.0
        score_f1 = (
            2 * precision * rappel / (precision + rappel)
            if precision + rappel
            else 0.0
        )
        statistiques.append({
            "classe": nom_classe,
            "precision": precision,
            "rappel": rappel,
            "f1": score_f1,
            "support": support,
        })

    return matrice, statistiques


def formater_statistiques(titre, matrice, statistiques):
    lignes = [
        f"===== {titre} : MATRICE DE CONFUSION =====",
        "Lignes : classes reelles | Colonnes : classes predites",
        f"{'':<14}" + "".join(f"{nom:>14}" for nom in CLASSES),
    ]
    for indice, nom_classe in enumerate(CLASSES):
        valeurs_ligne = "".join(f"{int(valeur):>14}" for valeur in matrice[indice])
        lignes.append(f"{nom_classe:<14}{valeurs_ligne}")

    lignes.extend([
        "",
        f"===== {titre} : RESULTATS PAR CLASSE =====",
        f"{'Classe':<14}{'Precision':>12}{'Rappel':>12}{'F1-score':>12}{'Support':>12}",
    ])
    for resultat in statistiques:
        lignes.append(
            f"{resultat['classe']:<14}{resultat['precision']:>12.4f}"
            f"{resultat['rappel']:>12.4f}{resultat['f1']:>12.4f}"
            f"{resultat['support']:>12}"
        )

    lignes.append(
        f"{'Moyenne macro':<14}"
        f"{sum(item['precision'] for item in statistiques) / len(CLASSES):>12.4f}"
        f"{sum(item['rappel'] for item in statistiques) / len(CLASSES):>12.4f}"
        f"{sum(item['f1'] for item in statistiques) / len(CLASSES):>12.4f}"
        f"{sum(item['support'] for item in statistiques):>12}"
    )
    return lignes


def afficher_statistiques_test(etiquettes, probabilites):
    matrice, statistiques = calculer_statistiques(etiquettes, probabilites)
    print("\n" + "\n".join(formater_statistiques("TEST", matrice, statistiques)))
    return matrice, statistiques


def ajouter_au_rapport(lignes):
    with CHEMIN_RAPPORT.open("a", encoding="utf-8") as fichier:
        fichier.write("\n".join(str(ligne) for ligne in lignes) + "\n")


# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================

def main():
    tf.keras.utils.set_random_seed(42)

    fichiers_train = trouver_fichiers_par_classe(DOSSIER_TRAIN)
    chemins_train, etiquettes_train, chemins_validation, etiquettes_validation = (
        separer_train_validation(fichiers_train)
    )
    fichiers_test = trouver_fichiers_par_classe(DOSSIER_TEST)
    chemins_test = []
    etiquettes_test = []
    for indice_classe, nom_classe in enumerate(CLASSES):
        fichiers = fichiers_test[nom_classe]
        chemins_test.extend(str(chemin) for chemin in fichiers)
        etiquettes_test.extend([indice_classe] * len(fichiers))

    dataset_validation = creer_dataset(chemins_validation, etiquettes_validation)
    dataset_test = creer_dataset(chemins_test, etiquettes_test)

    print("\nClasses : ")
    for indice, nom_classe in enumerate(CLASSES):
        print(f"  {indice} : {nom_classe}")
    print(f"\nImages d'entrainement : {len(chemins_train)}")
    print(f"Images de validation : {len(chemins_validation)}")
    print(f"Images de test : {len(chemins_test)}")

    # =========================================================
    # NORMALISATION
    # =========================================================

    normalisation = layers.Rescaling(1.0 / 255)
    dataset_validation = dataset_validation.map(
        lambda x, y: (normalisation(x), y)
    )
    dataset_test = dataset_test.map(
        lambda x, y: (normalisation(x), y)
    )

    ajouter_au_rapport([
        "",
        "=" * 72,
        f"SESSION : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Modele : MLP retenu, dropout 0.35 / 0.15",
        f"Classes : {CLASSES}",
        f"Images train : {len(chemins_train)}",
        f"Images validation : {len(chemins_validation)}",
        f"Images test : {len(chemins_test)}",
    ])

    print("\n===== ENTRAINEMENT DU MLP RETENU =====")
    meilleur_essai = None

    for numero, configuration in enumerate(CONFIGURATIONS_MLP, start=1):
        print(
            f"\n===== ESSAI {numero}/{len(CONFIGURATIONS_MLP)} : "
            f"{configuration['nom']} ====="
        )
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(GRAINE_ALEATOIRE)

        modele = creer_modele(configuration)
        modele.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("\n===== RÉSUMÉ DU MODÈLE =====")
        modele.summary()

        dataset_train_essai = creer_dataset(
            chemins_train, etiquettes_train, melanger=True
        )
        dataset_train_essai = dataset_train_essai.map(
            lambda x, y: (normalisation(x), y)
        )

        historique = modele.fit(
            dataset_train_essai,
            validation_data=dataset_validation,
            epochs=EPOCHS,
            callbacks=creer_callbacks(),
        )

        valeurs = historique.history
        meilleur_indice = min(
            range(len(valeurs["val_loss"])), key=valeurs["val_loss"].__getitem__
        )
        meilleure_epoque = meilleur_indice + 1
        perte_validation, precision_validation = modele.evaluate(
            dataset_validation, verbose=0
        )
        probabilites_validation = modele.predict(dataset_validation, verbose=0)
        matrice_validation, statistiques_validation = calculer_statistiques(
            etiquettes_validation, probabilites_validation
        )
        f1_macro_validation = sum(
            resultat["f1"] for resultat in statistiques_validation
        ) / len(CLASSES)

        resultat = {
            "configuration": configuration,
            "historique": valeurs,
            "meilleure_epoque": meilleure_epoque,
            "perte_validation": perte_validation,
            "precision_validation": precision_validation,
            "f1_macro_validation": f1_macro_validation,
            "matrice_validation": matrice_validation,
            "statistiques_validation": statistiques_validation,
            "poids": [poids.copy() for poids in modele.get_weights()],
        }

        print(f"\nMeilleure epoque : {meilleure_epoque}")
        print(f"Validation loss : {perte_validation:.4f}")
        print(f"Validation accuracy : {precision_validation:.4f}")
        print(f"Validation F1 macro : {f1_macro_validation:.4f}")

        lignes_essai = [
            "",
            f"ESSAI {numero} : {configuration['nom']}",
            f"Configuration : {configuration}",
            f"Parametres du modele : {modele.count_params()}",
            f"Meilleure epoque : {meilleure_epoque}",
            f"Epoques executees : {len(valeurs['loss'])}",
            f"Loss train a la meilleure epoque : {valeurs['loss'][meilleur_indice]:.4f}",
            f"Accuracy train a la meilleure epoque : {valeurs['accuracy'][meilleur_indice]:.4f}",
            f"Validation loss : {perte_validation:.4f}",
            f"Validation accuracy : {precision_validation:.4f}",
            f"Validation F1 macro : {f1_macro_validation:.4f}",
            *formater_statistiques(
                "VALIDATION", matrice_validation, statistiques_validation
            ),
        ]
        ajouter_au_rapport(lignes_essai)

        if (
            meilleur_essai is None
            or f1_macro_validation > meilleur_essai["f1_macro_validation"]
            or (
                f1_macro_validation == meilleur_essai["f1_macro_validation"]
                and perte_validation < meilleur_essai["perte_validation"]
            )
        ):
            meilleur_essai = resultat

    configuration = meilleur_essai["configuration"]
    print(f"\n===== MLP RETENU : {configuration['nom']} =====")
    print(f"F1 macro validation : {meilleur_essai['f1_macro_validation']:.4f}")
    print(f"Loss validation : {meilleur_essai['perte_validation']:.4f}")

    tf.keras.backend.clear_session()
    modele = creer_modele(configuration)
    modele.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    modele.set_weights(meilleur_essai["poids"])

    # Le jeu test n'est consulte qu'apres le choix du meilleur essai.
    print("\n===== EVALUATION FINALE SUR LE TEST =====")
    perte_test, precision_test = modele.evaluate(dataset_test)
    probabilites_test = modele.predict(dataset_test, verbose=0)
    matrice_test, statistiques_test = afficher_statistiques_test(
        etiquettes_test, probabilites_test
    )
    print(f"\nLoss : {perte_test:.4f}")
    print(f"Accuracy : {precision_test:.4f}")

    modele.save(str(CHEMIN_MODELE))
    print(f"\nModele sauvegarde : {CHEMIN_MODELE.name}")

    ajouter_au_rapport([
        "",
        f"MEILLEUR MODELE SELECTIONNE : {configuration['nom']}",
        f"Type de modele : {configuration['type']}",
        f"F1 macro validation : {meilleur_essai['f1_macro_validation']:.4f}",
        f"Loss validation : {meilleur_essai['perte_validation']:.4f}",
        f"Loss test : {perte_test:.4f}",
        f"Accuracy test : {precision_test:.4f}",
        *formater_statistiques("TEST", matrice_test, statistiques_test),
        f"Modele sauvegarde : {CHEMIN_MODELE.name}",
        f"Courbes sauvegardees : {CHEMIN_COURBES.name}",
        f"Rapport : {CHEMIN_RAPPORT.name}",
    ])

    afficher_statistiques_et_courbes(meilleur_essai["historique"])
    print(f"\nRapport texte sauvegarde : {CHEMIN_RAPPORT.name}")


if __name__ == "__main__":
    main()