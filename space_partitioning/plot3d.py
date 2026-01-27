import plotly.graph_objects as go


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
