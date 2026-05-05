from utilities import get_file_name, combine_and_save_files

import logging
module_logger = logging.getLogger('evaluate_forecasts')

def combine_model_predictions(config):

    """Function to combine model predictions.

    Args:
        config (dict): configuration dictionary.
    """

    module_logger.info('===============================================================')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    frequency = config['dataset']['frequency']
    ext = config['dataset']['ext']
    # fitting parameters    
    retrain_scenarios = config['fitting']['retrain_scenarios']
    combine_only =config['fitting']['combine_only']
    # model parameters
    model_names = config['model_names']

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        module_logger.info(f'[ Model name: {m} ]')

        if not combine_only:

            for rs in retrain_scenarios:

                f_list_tmp = get_file_name(
                    path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'], 
                    name_list = None,
                    ext = ext, remove_ext=False, add_path=True
                )
                f_list_tmp.sort(key = lambda x: int("".join([i for i in x if i.isdigit()])))

                module_logger.info(f'Combining predictions for retrain scenario: {rs}')
                combine_and_save_files(
                    path_list_to_read = None,
                    path_list_to_write = ['results', dataset_name, frequency, m, 'preds', 'byretrain'],
                    name_list = [dataset_name, frequency, m, rs, 'outsample'],
                    ext = ext,
                    files_to_read = f_list_tmp
                )

        f_list_tmp = get_file_name(
            path_list = ['results', dataset_name, frequency, m, 'preds', 'byretrain'], 
            name_list = None,
            ext = ext, remove_ext=False, add_path=True
        )
        f_list_tmp.sort(key = lambda x: int("".join([i for i in x if i.isdigit()])))
        
        combine_and_save_files(
            path_list_to_read = None,
            path_list_to_write = ['results', dataset_name, frequency, m, 'preds'],
            name_list = [dataset_name, frequency, m, 'preds'],
            ext = ext,
            files_to_read = f_list_tmp
        )

        module_logger.info('----------------------------- END -----------------------------')

    module_logger.info('===============================================================')

    return

def combine_dataset_predictions(config):

    """Function to combine dataset predictions.

    Args:
        config (dict): configuration dictionary.
    """

    module_logger.info('===============================================================')
    module_logger.info('---------------------------- START ----------------------------')

    dataset_names = config['dataset']['dataset_names']
    frequencies = config['dataset']['frequencies']
    ext = config['dataset']['ext']
    model_names = config['model_names']

    for i in range(len(dataset_names)):

        dataset_name_tmp = dataset_names[i]
        freq_tmp = frequencies[i]
        module_logger.info(f'[ Dataset: {dataset_name_tmp} | Frequency: {freq_tmp} ]')

        # get file paths and names of evaluation and time samples
        preds_f_list = []
        for m in model_names:
            preds_f_list += get_file_name(
                path_list = ['results', dataset_name_tmp, freq_tmp, m, 'preds'], 
                name_list = [dataset_name_tmp, freq_tmp, m, 'preds'],
                ext = ext, remove_ext=False, add_path=True
            )

        combine_and_save_files(
            path_list_to_read = None,
            path_list_to_write = ['results', dataset_name_tmp, freq_tmp, 'preds'],
            name_list = [dataset_name_tmp, freq_tmp, 'preds'],
            ext = ext,
            files_to_read = preds_f_list
        )

    module_logger.info('===============================================================')

    return