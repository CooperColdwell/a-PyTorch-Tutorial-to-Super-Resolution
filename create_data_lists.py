from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['../Super-resolution/train2014'],
                      test_folders=['../Super-resolution/val2014',
                                    '../Super-resolution/BSDS100',
                                    '../Super-resolution/Set5',
                                    '../Super-resolution/Set14'],
                      min_size=100,
                      output_folder='data_lists')
