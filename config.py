weight_decay = 0.0001
gpu_config = {
    'ids': '0'
}
dataset_config = {
    'dataset_name': 'cityscapes',
    'num_readers': 1,
    'num_threads': 1,
    'batch_size': 1,
    'input_shape_original': [1024, 2048]

}


def print_config(dataset):
    import util
    from pprint import pprint
    from tensorflow.contrib.slim.python.slim.data import parallel_reader

    def do_print(stream=None):
        print(util.log.get_date_str(), file=stream)
        print('\n# =========================================================================== #', file=stream)
        print('# Training flags:', file=stream)
        print('# =========================================================================== #', file=stream)

        def print_ckpt(path):
            ckpt = util.tf.get_latest_ckpt(path)
            if ckpt is not None:
                print('Resume Training from : %s' % (ckpt), file=stream)
                return True
            return False


        print('\n# =========================================================================== #', file=stream)
        print('# pixel_link net parameters:', file=stream)
        print('# =========================================================================== #', file=stream)
        vars = globals()
        for key in vars:
            var = vars[key]
            if util.dtype.is_number(var) or util.dtype.is_str(var) or util.dtype.is_list(var) or util.dtype.is_tuple(
                    var):
                pprint('%s=%s' % (key, str(var)), stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        data_files = parallel_reader.get_data_files(dataset.data_sources)
        pprint(sorted(data_files), stream=stream)
        print('', file=stream)

    do_print(None)