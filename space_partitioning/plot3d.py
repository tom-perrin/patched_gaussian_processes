import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_field_3d(
    X_field, Y_field,
    X_obs=None, Y_obs=None,
    X_pred=None, Y_pred=None,
    title="Champ et observations"
):
    """
    X_field : (N, 2) coordonnées du champ
    Y_field : (N,) valeurs du champ analytique

    X_obs   : (N_obs, 2) points observés
    Y_obs   : (N_obs,) valeurs observées

    X_pred  : (N_pred, 2) points prédits
    Y_pred  : (N_pred,) valeurs prédites
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- Champ analytique (surface ou nuage dense)
    surf = ax.plot_trisurf(
        X_field[:, 0],
        X_field[:, 1],
        Y_field,
        cmap="viridis",
        alpha=0.6,
        linewidth=0
    )

    # --- Observations
    if X_obs is not None and Y_obs is not None:
        ax.scatter(
            X_obs[:, 0],
            X_obs[:, 1],
            Y_obs,
            c="red",
            s=40,
            label="Observations"
        )

    # --- Prédictions
    if X_pred is not None and Y_pred is not None:
        ax.scatter(
            X_pred[:, 0],
            X_pred[:, 1],
            Y_pred,
            c="black",
            s=60,
            marker="x",
            label="Prédictions"
        )

    # --- Mise en forme
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Valeur du champ")
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Champ analytique")

    if (X_obs is not None) or (X_pred is not None):
        ax.legend()

    plt.tight_layout()
    plt.show()
