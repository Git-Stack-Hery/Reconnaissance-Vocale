from pathlib import Path

import tensorflow as tf


# =========================================================
# PARAMETRES
# =========================================================

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_DONNEES = DOSSIER_PROJET / "Donnees"
DOSSIERS_SOURCE = {
    "train": DOSSIER_DONNEES / "train",
    "test": DOSSIER_DONNEES / "test",
}
DOSSIER_SORTIE = DOSSIER_PROJET / "Donnees_MFCC"

CLASSES = ["0_Moi", "1_Invites", "2_Autres"]
FREQUENCE_ECHANTILLONNAGE = 22050
DUREE_AUDIO = 1.5
NB_ECHANTILLONS = int(FREQUENCE_ECHANTILLONNAGE * DUREE_AUDIO)
NB_MELS = 64
TAILLE_FENETRE = 512
PAS_FENETRE = 256
TAILLE_FFT = 512
NB_TRAMES = 1 + (NB_ECHANTILLONS - TAILLE_FENETRE) // PAS_FENETRE
PLAGE_DB = 80.0


def trouver_fichiers(dossier_source):
    chemins = []
    chemins_sortie = []

    for nom_classe in CLASSES:
        dossier_classe = dossier_source / nom_classe
        if not dossier_classe.is_dir():
            raise FileNotFoundError(f"Dossier de classe introuvable : {dossier_classe}")

        fichiers = sorted(
            chemin for chemin in dossier_classe.iterdir()
            if chemin.is_file() and chemin.suffix.lower() == ".wav"
        )
        if not fichiers:
            raise FileNotFoundError(f"Aucun fichier WAV dans : {dossier_classe}")

        dossier_destination = DOSSIER_SORTIE / dossier_source.name / nom_classe
        dossier_destination.mkdir(parents=True, exist_ok=True)
        png_attendus = {f"{chemin.stem}.png" for chemin in fichiers}
        for ancien_png in dossier_destination.glob("*.png"):
            if ancien_png.name not in png_attendus:
                ancien_png.unlink()

        chemins.extend(str(chemin) for chemin in fichiers)
        chemins_sortie.extend(
            str(dossier_destination / f"{chemin.stem}.png") for chemin in fichiers
        )

    return chemins, chemins_sortie


def convertir_en_png(chemin_audio, chemin_png):
    audio, frequence = tf.audio.decode_wav(
        tf.io.read_file(chemin_audio), desired_channels=1
    )
    verification_frequence = tf.debugging.assert_equal(
        frequence,
        FREQUENCE_ECHANTILLONNAGE,
        message="Les fichiers WAV doivent etre en 22050 Hz.",
    )

    with tf.control_dependencies([verification_frequence]):
        audio = tf.squeeze(audio, axis=-1)

    audio = audio[:NB_ECHANTILLONS]
    remplissage = NB_ECHANTILLONS - tf.shape(audio)[0]
    audio = tf.pad(audio, [[0, remplissage]])

    spectrogramme = tf.signal.stft(
        audio,
        frame_length=TAILLE_FENETRE,
        frame_step=PAS_FENETRE,
        fft_length=TAILLE_FFT,
    )
    spectrogramme_puissance = tf.square(tf.abs(spectrogramme))
    banque_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NB_MELS,
        num_spectrogram_bins=TAILLE_FFT // 2 + 1,
        sample_rate=FREQUENCE_ECHANTILLONNAGE,
        lower_edge_hertz=80.0,
        upper_edge_hertz=FREQUENCE_ECHANTILLONNAGE / 2,
        dtype=tf.float32,
    )
    spectrogramme_mel = tf.matmul(spectrogramme_puissance, banque_mel)
    spectrogramme_db = 10.0 * tf.math.log(spectrogramme_mel + 1e-10) / tf.math.log(10.0)
    spectrogramme_db -= tf.reduce_max(spectrogramme_db)
    spectrogramme_db = tf.clip_by_value(spectrogramme_db, -PLAGE_DB, 0.0)

    pixels = tf.cast(
        tf.round((spectrogramme_db + PLAGE_DB) * (255.0 / PLAGE_DB)),
        tf.uint8,
    )
    pixels = tf.transpose(pixels, perm=[1, 0])[..., tf.newaxis]
    pixels = tf.ensure_shape(pixels, [NB_MELS, NB_TRAMES, 1])
    return chemin_png, tf.io.encode_png(pixels)


def main():
    total = 0

    for nom_split, dossier_source in DOSSIERS_SOURCE.items():
        chemins_audio, chemins_png = trouver_fichiers(dossier_source)
        dataset = tf.data.Dataset.from_tensor_slices((chemins_audio, chemins_png))
        dataset = dataset.map(
            convertir_en_png,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        compteur_split = 0
        for chemin_png, contenu_png in dataset:
            Path(chemin_png.numpy().decode("utf-8")).write_bytes(contenu_png.numpy())
            compteur_split += 1

        total += compteur_split
        print(f"{nom_split} : {compteur_split} spectrogrammes crees")

    print(f"\nTotal : {total} spectrogrammes dans {DOSSIER_SORTIE}")
    print(f"Taille des images : {NB_TRAMES} x {NB_MELS} pixels (temps x frequences)")


if __name__ == "__main__":
    main()