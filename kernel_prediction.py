import os
import argparse
from models import MLP, ResNet
import neural_tangents as nt
import dataset
import jax.numpy as jnp
import jax
import time
from datetime import datetime
jax.config.update("jax_enable_x64", True)

if 'SLURM_PROCID' in os.environ:
    jax.distributed.initialize()

LINE_WIDTH = 80

setup = {
    'powers': [2, 6],
    'targets': ['U0_atomization'],
    'num_rot': 1,
    'shuffle': True,
    'gpu_mem_frac': 0.4,
}


class Logger():
    def __init__(self):
        # print(os.environ['SLURM_PROCID'])
        self.is_main_proc = (int(os.environ['SLURM_PROCID']) == 0
                             if 'SLURM_PROCID' in os.environ else True)
        self.log = "Starting logging at "
        self.log += datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
        self.write(self.log)

    def write(self, txt, newsection=False):
        if newsection:
            txt = '\n' + '-' * LINE_WIDTH + "\n" + txt
        if self.is_main_proc:
            print(txt)
        self.log += txt + "\n"

    def to_file(self, setup):
        if self.is_main_proc:
            with open(get_output_filename(setup), "w") as f:
                f.write(self.log)

    def __str__(self):
        return self.log


class Walltimer:
    def start(self):
        self.start_time = time.time()
        return "Starting timer..."

    def stop(self, run=None):
        self.stop_time = time.time()
        elapsed_time = self.stop_time - self.start_time
        output = (f"Time elapsed: {elapsed_time:.0f} s")
        if run:
            run.log({"time_elapsed": elapsed_time})
        return output


def create_parser():
    parser = argparse.ArgumentParser(
        description=("Toolkit for equivariant NTK predictions on the QM9"
                     " dataset")
    )
    parser.add_argument("--bandlimit", type=int, default=6)
    parser.add_argument("--n_train", type=int, default=3)
    parser.add_argument("--n_test", type=int, default=1)
    parser.add_argument("--kn_reg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp",
                        "resnet"])
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--kernel", type=str, default="ntk", choices=["ntk",
                        "nngp"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device_count", type=int, default=1)
    parser.add_argument("--max_gpu_mem", type=float, default=45.0)
    parser.add_argument("--force_kn_recompute", action="store_true")
    return parser


def compute_rmse_loss_scalar(preds, targets):
    return jnp.sqrt(jnp.mean((preds - targets) ** 2))


def compute_mae_loss_scalar(preds, targets):
    return jnp.mean(jnp.abs(preds - targets))


def predictions_to_output(preds, target_vals, run=None, unit_conv=1.0):
    n_samples = preds.shape[0]
    output = f"\n{'predictions':>20}\t{'targets':>20}\n"
    preds = jnp.squeeze(preds, axis=1) * unit_conv
    target_vals = jnp.squeeze(target_vals, axis=1) * unit_conv
    for i in range(n_samples):
        output += f"{preds[i]:20.3f}\t{target_vals[i]:20.3f}\n"

    rmse = compute_rmse_loss_scalar(preds, target_vals)
    relative_rmse = rmse / (dataset.qm9_meta['stats'][setup['targets'][0]][-1] * dataset.Ha_to_meV)
    mae = compute_mae_loss_scalar(preds, target_vals)
    output += f"\nRMSE: {rmse:.3f}\n"
    output += f"\nrelative RMSE: {relative_rmse:.3f}\n"
    output += f"\nMAE: {mae:.3f}\n"
    if run:
        data = [[preds[i], target_vals[i]] for i in range(n_samples)]
    return output


def get_output_filename(setup):
    slurm_id = setup.get('slurm_id', '')
    filename = (f"{setup['model']}_L-{setup['bandlimit']:02d}_"
                f"n-train-{setup['n_train']:04d}_n-test-{setup['n_test']:03d}")
    if slurm_id:
        filename += f"_slurm-{slurm_id}"
    return filename + ".log"



def add_args_to_setup(args, setup):
    arg_dict = vars(args)
    for k, v in arg_dict.items():
        setup[k] = v
    return setup


def setup_to_output(setup):
    output = "\nComputing with\n"
    for k, v in setup.items():
        output += f"\t{k}: {v}\n"

    return output


def batch_predict_fn(predict_fn, setup, data_source, rotations):
    # batch_size = min(max_batch_size, setup['n_test'])
    batch_size = 8
    n_batches = setup['n_test'] // batch_size

    all_preds = []
    all_y_test = dict([(t, []) for t in setup['targets']])
    for i in range(n_batches):
        n_next_samples = min(batch_size, setup['n_test'] - i * batch_size)
        x_test, y_test = dataset.load_sphere_data(setup['targets'], data_source,
                                                  setup['shuffle'],
                                                  setup['bandlimit'],
                                                  dataset.qm9_meta['atom_types'],
                                                  setup['powers'],
                                                  setup['seed'],
                                                  max_samples=n_next_samples,
                                                  offset=i * batch_size,
                                                  rotations=rotations)

        all_preds.append(predict_fn(x_test=x_test, get=setup['kernel']))
        for k in all_y_test:
            all_y_test[k].append(y_test[k])

    all_preds = jnp.concatenate(all_preds, axis=0)
    for k in all_y_test:
        all_y_test[k] = jnp.concatenate(all_y_test[k], axis=0)
    return all_preds, all_y_test


def normalize_y_values(y, mean, std):
    return (y - mean) / std


def denormalize_y_values(y, mean, std):
    return y * std + mean


def check_args(args):
    if args.model == 'mlp':
        if args.n_layers is None:
            raise ValueError("n_layers missing for MLP model")
    elif args.n_layers is not None:
        raise ValueError("n_layers is only for MLP model")


def main(args, setup):
    setup = add_args_to_setup(args, setup)
    setup['n_gpus'] = (
        jax.device_count() if setup['device_count'] == -1 else
        setup['device_count']
    )
    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        slurm_id = (f"{os.environ['SLURM_ARRAY_JOB_ID']}_"
                    f"{os.environ['SLURM_ARRAY_TASK_ID']}")
    elif 'SLURM_JOB_ID' in os.environ:
        slurm_id = os.environ['SLURM_JOB_ID']
    else:
        slurm_id = None
    setup['slurm_id'] = slurm_id


    if args.model == "mlp":
        model = MLP(dataset.qm9_meta['max_num_atoms'], setup['n_layers'])
    elif args.model == "resnet":
        model = ResNet(dataset.qm9_meta['max_num_atoms'])
    else:
        raise ValueError(f"Unknown model {args.model}")
    setup['model_architecture'] = model.get_architecture()

    logger = Logger()
    if setup['batch_size'] is None:
        logger.write("No batch size specified, using optimal one.")
        setup['batch_size'] = 8

    logger.write(setup_to_output(setup), newsection=True)

    logger.write(
        f"\nJAX device count: {jax.device_count()}"
        f"\nJAX local device count: {jax.local_device_count()}\n",
        newsection=True
    )

    data_source = dataset.create_data_source(dataset.name, dataset.data_dir, None)

    x_train, y_train = dataset.load_sphere_data(setup['targets'], data_source['train'],
                                                True, setup['bandlimit'],
                                                dataset.qm9_meta['atom_types'],
                                                setup['powers'],
                                                setup['seed'],
                                                max_samples=setup['n_train'])

    key = jax.random.key(setup['seed'])
    key, subkey = jax.random.split(key)

    timer = Walltimer()
    logger.write(timer.start())
    kernel_fn_batched = nt.batch(model.kernel_fn,
                                 device_count=setup['device_count'],
                                 batch_size=setup['batch_size'])
    sel_targ = setup['targets'][0]


    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn_batched, x_train,
                                                          normalize_y_values(y_train[sel_targ],
                                                                             *dataset.qm9_meta['stats'][sel_targ]),
                                                          diag_reg=setup['kn_reg'])

    x_test, y_test = dataset.load_sphere_data(setup['targets'], data_source['test'],
                                              True, setup['bandlimit'],
                                              dataset.qm9_meta['atom_types'],
                                              setup['powers'],
                                              setup['seed'],
                                              max_samples=setup['n_test'])
    preds = predict_fn(x_test=x_test, get='ntk')

    preds = denormalize_y_values(preds, *dataset.qm9_meta['stats'][sel_targ])

    if sel_targ in dataset.qm9_meta['unit_convs']:
        unit_conv = dataset.qm9_meta['unit_convs'][sel_targ]
    else:
        unit_conv = 1.0
    logger.write(predictions_to_output(preds, y_test[sel_targ], unit_conv), newsection=True)

    logger.write(timer.stop(), newsection=True)
    logger.write("Done.")
    logger.to_file(setup)


args = create_parser().parse_args()
check_args(args)
main(args, setup)
