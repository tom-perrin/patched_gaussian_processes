import plotly.graph_objects as go
import plotly.colors as pc

import numpy as np


def plot_field_3d_plotly(
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

    fig = go.Figure()

    # --- Champ analytique (nuage 3D coloré → très fluide)
    fig.add_trace(go.Scatter3d(
        x=X_field[:, 0],
        y=X_field[:, 1],
        z=Y_field,
        mode="markers",
        marker=dict(
            size=3,
            color=Y_field,
            colorscale="Viridis",
            opacity=0.7,
            colorbar=dict(title="Champ")
        ),
        name="Champ analytique"
    ))

    # --- Observations
    if X_obs is not None and Y_obs is not None:
        fig.add_trace(go.Scatter3d(
            x=X_obs[:, 0],
            y=X_obs[:, 1],
            z=Y_obs,
            mode="markers",
            marker=dict(
                size=6,
                color="red"
            ),
            name="Observations"
        ))

    # --- Prédictions
    if X_pred is not None and Y_pred is not None:
        fig.add_trace(go.Scatter3d(
            x=X_pred[:, 0],
            y=X_pred[:, 1],
            z=Y_pred,
            mode="markers",
            marker=dict(
                size=8,
                color="black",
                symbol="x"
            ),
            name="Prédictions"
        ))

    # --- Mise en forme
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Valeur du champ",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0.02,
            y=0.98
        )
    )

    fig.show()

def plot_predicted_field_3d_plotly(
    Xg, Yg,
    Z_pred,
    Z_region,
    valid_mask,
    title="Champ krigé – représentation 3D"
):
    fig = go.Figure()

    regions = np.unique(Z_region[valid_mask])

    # Palette discrète (autant de couleurs que de régions)
    colors = pc.qualitative.Set3
    n_colors = len(colors)

    for i, reg in enumerate(regions):
        mask = (Z_region == reg) & valid_mask

        if not np.any(mask):
            continue

        # Surface uniquement sur la région
        Z_reg = np.where(mask, Z_pred, np.nan)

        fig.add_trace(go.Surface(
            x=Xg,
            y=Yg,
            z=Z_reg,

            # Couleur uniforme pour toute la région
            surfacecolor=np.zeros_like(Z_reg),
            colorscale=[[0, colors[i % n_colors]],
                        [1, colors[i % n_colors]]],

            showscale=False,
            opacity=1.0,
            name=f"Région {reg}"
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Valeur du champ",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0.02,
            y=0.98
        )
    )

    fig.show()

def plot_regions_2d_plotly(
    Xg, Yg,
    Z_region,
    title="Partition de l'espace (régions)"
):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=Xg[0, :],
        y=Yg[:, 0],
        z=Z_region,
        colorscale="tab10",
        colorbar=dict(title="Région"),
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=60, r=20, b=60, t=60)
    )

    fig.show()
