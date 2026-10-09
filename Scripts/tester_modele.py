from pathlib import Path

import tensorflow as tf

from preparer_spectrogrammes import convertir_en_png


# =========================================================
# PARAMÈTRES
# =========================================================

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
CHEMIN_MODELE = DOSSIER_PROJET / "Modele" / "modele_reconnaissance_vocale.h5"
CHEMIN_AUDIO = DOSSIER_PROJET / "Donnees" / "test" / "0_Moi" / "80_Moi.wav"

CLASSES = ["Moi", "Invites", "Autres"]
TAILLE_IMAGE = (64, 128)


# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================

if not CHEMIN_MODELE.is_file():
    raise FileNotFoundError(f"Modele introuvable : {CHEMIN_MODELE}")

if not CHEMIN_AUDIO.is_file():
    raise FileNotFoundError(
        f"Audio introuvable : {CHEMIN_AUDIO}\n"
        "Modifiez CHEMIN_AUDIO avec le chemin d'un fichier WAV a tester."
    )

modele = tf.keras.models.load_model(CHEMIN_MODELE)


# =========================================================
# CONVERSION DU WAV EN SPECTROGRAMME
# =========================================================

dataset_audio = tf.data.Dataset.from_tensor_slices((
    [str(CHEMIN_AUDIO)],
    [str(CHEMIN_AUDIO)],
))
dataset_spectrogramme = dataset_audio.map(convertir_en_png)
_, contenu_png = next(iter(dataset_spectrogramme))

img = tf.io.decode_png(contenu_png, channels=1)
img = tf.ensure_shape(img, [TAILLE_IMAGE[0], TAILLE_IMAGE[1], 1])


# =========================================================
# NORMALISATION ET DIMENSION BATCH
# =========================================================

img_array = tf.cast(img, tf.float32) / 255.0
img_array = tf.expand_dims(img_array, axis=0)


# =========================================================
# PRÉDICTION
# =========================================================

prediction = modele.predict(img_array, verbose=0)[0]
indice_classe = int(tf.argmax(prediction).numpy())

print("\n===== PRÉDICTION =====")
print("Probabilites :", prediction)
for indice, nom_classe in enumerate(CLASSES):
    print(f"{nom_classe} : {prediction[indice]:.4f}")

print(f"\nRésultat : {CLASSES[indice_classe]}")
print(f"Confiance : {prediction[indice_classe]:.2%}")
print(f"Fichier testé : {CHEMIN_AUDIO.name}")