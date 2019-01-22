config = {}

config['batch_size'] = 8
config['thread_num'] = 10
config['img_inp_shape'] = [None, None, 1]
config['min_size'] = 512
config['img_pad_val'] = 0
config['label_pad_val'] = 0

config['data_root'] = '/content/gdrive/My Drive/viz/takehome'
config['log_root'] = '/content/gdrive/My Drive/viz/gan_logs'
# config['samples_root'] = config['data_root'] + '/jpg'
# config['labels_root'] = config['data_root'] + '/bmp'

config['train_val_lists'] = config['data_root'] + '/train_val_lists.pickle'