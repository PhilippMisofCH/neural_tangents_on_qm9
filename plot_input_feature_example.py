import dataset
import visualisation


data_source = dataset.create_data_source(dataset.name, dataset.data_dir, 'train')
powers = [2, 6]
bandlimit = 32
positions, charges, energies = dataset.load_dataset(['U'], data_source, shuffle=True, max_samples=1)

sample_id = 3
select_id = 6
visualisation.create_molecule_plots(positions[sample_id], charges[sample_id], powers, bandlimit,
                                    dataset.qm9_meta['atom_types'], select_id)
