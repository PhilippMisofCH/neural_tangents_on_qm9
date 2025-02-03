import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import utils
import jax.numpy as jnp
import jax
import dataset


def create_molecule_configuration_plot(positions: jax.Array, charges: jax.Array, atom_types:
                                       list[float], select_id: int) -> go.Figure():
    num_atoms = jnp.count_nonzero(charges)
    positions = positions[:num_atoms]
    charges = charges[:num_atoms]

    rel_pos = positions - positions[select_id]
    color_table = {}
    for i, t in enumerate(atom_types):
        color_table[int(t)] = px.colors.qualitative.Plotly[i]

    colors = [color_table[int(c)] for c in charges]

    trace = go.Scatter3d(x=rel_pos[:, 0], y=rel_pos[:, 1], z=rel_pos[:, 2],
                         mode='markers',
                         marker={
        'size': 12,
        'color': colors,
    }
    )
    return trace


def create_per_atom_feature_plot(bandlimit: int, scalar_field: jax.Array, n_elements: int,
                                 close_azimut: bool = True) -> list[go.Figure()]:
    thetas, phis = utils.get_sph_grid(bandlimit)
    sph_vecs = utils.create_sphere_vecs(bandlimit)
    features_per_elem = scalar_field.shape[-1] // n_elements

    if scalar_field.ndim == 2:
        scalar_field = jnp.expand_dims(scalar_field, -1)

    if close_azimut:
        sph_vecs = jnp.concatenate([sph_vecs, jnp.expand_dims(sph_vecs[:, 0, :], 1)], axis=1)
        scalar_field = jnp.concatenate([scalar_field, jnp.expand_dims(scalar_field[:, 0], 1)], axis=1)

    traces = []
    for i in range(n_elements):
        for j in range(features_per_elem):
            traces.append((
                go.Surface(x=sph_vecs[..., 0], y=sph_vecs[..., 1], z=sph_vecs[..., 2],
                           surfacecolor=scalar_field[..., i * features_per_elem + j],
                           coloraxis='coloraxis'),
                (i + 1, j + 1)
            ))
    return traces


def create_molecule_plots(positions: jax.Array, charges: jax.Array, powers: list[int], bandlimit:
                          int, atom_types: list[float], select_id: int):
    n_elements = len(atom_types)
    scalar_field = dataset.create_spherical_potentials(positions, charges, bandlimit, atom_types,
                                                       powers)[select_id]
    features_per_elem = scalar_field.shape[-1] // n_elements
    fig = make_subplots(rows=n_elements + 1, cols=features_per_elem,
                        specs=[[{'is_3d': True, 'colspan': 2}, None]] + [[{'is_3d': True}] * features_per_elem] * n_elements
                        )

    conf_trace = create_molecule_configuration_plot(positions, charges, atom_types, select_id)
    fig.add_trace(conf_trace, 1, 1)

    traces = create_per_atom_feature_plot(bandlimit, scalar_field, n_elements)
    for trace, (i, j) in traces:
        fig.add_trace(trace, i + 1, j)

    max_val = jnp.max(scalar_field).item()
    min_val = jnp.min(scalar_field).item()

    fig.update_layout(
        coloraxis={
            # 'colorscale': 'viridis',
            'colorscale': 'plasma',
            'cmin': min_val,
            'cmax': max_val
        },
        coloraxis_colorbar={
            'title': 'feature value'
        },
        autosize=True,
        height=3000,
    )
    fig.update_xaxes(rangeslider=dict(visible=False))
    fig.write_image('molecule_plot.png')
    fig.write_html('molecule_plot.html')
    fig.show()
