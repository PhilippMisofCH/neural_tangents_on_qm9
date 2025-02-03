import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import s2fft
from jax.typing import ArrayLike


def get_sph_grid(bandlimit: int) -> tuple[jax.Array, jax.Array]:
    sampling = 'dh'
    phis = s2fft.sampling.s2_samples.phis_equiang(bandlimit, sampling)
    thetas = s2fft.sampling.s2_samples.thetas(bandlimit, sampling)
    return thetas, phis


def sph_to_cart(thetas: ArrayLike, phis: ArrayLike) -> jax.Array:
    ths, phs = jnp.meshgrid(thetas, phis, sparse=False, indexing='ij')
    cart_vec = jnp.stack([jnp.sin(ths) * jnp.cos(phs),
                          jnp.sin(ths) * jnp.sin(phs),
                          jnp.cos(ths)], axis=-1)
    return cart_vec


def create_sphere_vecs(bandlimit: int) -> jax.Array:
    thetas, phis = get_sph_grid(bandlimit)
    sph_vecs = sph_to_cart(thetas, phis)
    return sph_vecs
