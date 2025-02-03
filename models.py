from gntk.utils.containers import Module
from neural_tangents import stax, Kernel
from neural_tangents.stax import layer, requires, Diagonal, Bool
import jax


class ResNet(Module):
    __name__ = "ResNet"

    def __init__(self, max_num_atoms: int):
        resnet_block = stax.serial(
            stax.FanOut(2),
            stax.parallel(
                stax.serial(
                    stax.Dense(512),
                    stax.Relu(),
                    stax.Dense(512),
                ),
                stax.Identity(),
            ),
            stax.FanInSum())
        single_atom_network = stax.serial(
            stax.Flatten(),
            stax.Dense(512),
            stax.repeat(resnet_block, 2),
            stax.Dense(512),
            stax.Relu(),
            stax.Dense(256),
        )

        atom_networks = stax.parallel(*([single_atom_network] * max_num_atoms))

        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(
            _split_atom_features(max_num_atoms),
            atom_networks,
            stax.FanInSum(),
            stax.Dense(50),
            stax.Relu(),
            stax.Dense(1)
        )


class MLP(Module):
    __name__ = "MLP"

    def __init__(self, max_num_atoms: int, n_layers: int):
        self.n_layers = n_layers
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        perceptron = stax.serial(
            stax.Dense(512),
            stax.Relu(),
        )
        single_atom_network = stax.serial(
            stax.Flatten(),
            stax.Dense(512),
            stax.Relu(),
            stax.repeat(perceptron, n_layers - 2) if n_layers > 2 else
            stax.Identity(),
            stax.Dense(10),
        )

        atom_networks = stax.parallel(*([single_atom_network] * max_num_atoms))

        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(
            _split_atom_features(max_num_atoms),
            atom_networks,
            stax.FanInSum(),
            stax.Dense(1)
        )

    def __str__(self) -> str:
        return f"MLP(n_layers_{self.n_layers})"

    def get_architecture(self) -> str:
        return f"n_layers-{self.n_layers}"


@layer
def _split_atom_features(max_num_atoms: int):
    def init_fn(rng, input_shape: tuple[int, ...]) -> tuple[list, tuple]:
        per_atom_shape = (input_shape[0], *input_shape[2:])
        return [per_atom_shape] * max_num_atoms, ()

    def apply_fn(params, inputs: jax.Array, **kwargs) -> list:
        return [inputs[:, i] for i in range(max_num_atoms)]

    def extract_all_subkernels(k: jax.Array, idx: int) -> Kernel:
        nngp, ntk, cov1, cov2 = k.nngp, k.ntk, k.cov1, k.cov2
        return k.replace(
            nngp=extract_subkernel(nngp, idx),
            ntk=extract_subkernel(ntk, idx),
            cov1=extract_subkernel(cov1, idx),
            cov2=extract_subkernel(cov2, idx),
        )

    def extract_subkernel(kernel: jax.Array, idx: int) -> jax.Array:
        # TODO: should also handle diagonal_batch and diagonal_spatial
        if kernel is None or kernel.ndim == 0:
            return kernel
        if kernel.ndim == 8:
            return kernel[:, :, idx, idx, :, :, :, :]
        elif kernel.ndim == 7:
            return kernel[:, idx, idx, :, :, :, :]
        elif kernel.ndim == 5:
            return kernel[:, :, idx, :, :]
        elif kernel.ndim == 4:
            return kernel[:, idx, :, :]
        else:
            raise ValueError(f"Invalid kernel shape: {kernel.shape}")

    @requires(batch_axis=0,
              channel_axis=4,
              diagonal_spatial=Diagonal(
                  output=Bool.MAYBE)
              )
    def kernel_fn(k: jax.Array, **kwargs) -> list[Kernel]:
        ks = [extract_all_subkernels(k, i) for i in range(max_num_atoms)]
        return ks

    return init_fn, apply_fn, kernel_fn
