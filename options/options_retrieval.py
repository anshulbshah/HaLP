import socket

hostname = socket.gethostname()
if hostname == 'r26':
  dataroot = "/cis/net/r26a/data/Datasets/"
elif hostname == 'io63':
  dataroot = "/cis/net/io63b/data/ashah/Datasets/"
elif hostname == 'io76':
  dataroot = "/cis/net/io76b/data/ashah/datasets/"
elif hostname == "io65":
  dataroot = "/cis/net/io65b/data/ashah/Datasets/"
elif hostname == "r11":
  dataroot = "/cis/net/r11_ssd/data/ashah/Datasets/"
elif hostname == "io62":
  dataroot = "/cis/net/io62a/data/ashah/Datasets/"
elif hostname == "io64":
  dataroot = "/cis/net/io64a/data/ashah/Datasets/"
elif hostname == "io92":
  dataroot = "/export/io92/data/ashah/datasets/"
elif hostname == "io35":
  dataroot = "/export/r35a/data/ashah/datasets/"
elif hostname == 'io79':
  dataroot = "/cis/net/io79a/data/ashah/Datasets"
elif hostname == 'r25':
  dataroot = "/cis/net/r25/data/ashah/datasets"
elif hostname == 'r35':
  dataroot = "/export/r35a/data/ashah/datasets"
elif hostname == 'io90':
  dataroot = "/export/io90/data/ashah"
elif hostname == 'r28':
  dataroot = "/cis/net/r28a/data/ashah"
elif hostname == 'r29':
  dataroot = '/cis/net/r29/data/Anshul/datasets'
elif 'umiacs' in hostname:
  dataroot = "/fs/vulcan-projects/jigsaw_selfsup_shlokm/PosNeg_CMD/"
else:
  dataroot = "./data"


class  opts_ntu_60_cross_view():

  def __init__(self):

  # graph based model
   self.agcn_model_args = {
      "num_class": 60,
      "num_point": 25,
      "num_person": 2,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

   #image based model
   self.hcn_model_args = {
      "in_channel":3,
      "out_channel":64,
      "window_size":64,
      "num_joint":25,
      "num_person":2,
      "num_class":60
    }

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":150,
      "en_hidden_size":1024,
      "en_num_layers":3,
      "num_class":60
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy',
     'label_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xview/train_label.pkl',
     'num_frame_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xview/val_data_joint.npy',
     'label_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xview/val_label.pkl',
     'num_frame_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xview/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }

class  opts_ntu_60_cross_subject():

  def __init__(self):

  # graph based model
   self.agcn_model_args = {
      "num_class": 60,
      "num_point": 25,
      "num_person": 2,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

   #image based model
   self.hcn_model_args = {
      "in_channel":3,
      "out_channel":64,
      "window_size":64,
      "num_joint":25,
      "num_person":2,
      "num_class":60
    }

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":150,
      "en_hidden_size":1024,
      "en_num_layers":3,
      "num_class":60
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy',
     'label_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xsub/train_label.pkl',
     'num_frame_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xsub/val_data_joint.npy',
     'label_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xsub/val_label.pkl',
     'num_frame_path': f'{dataroot}/NTU/NTU-RGB-D-60-AGCN/xsub/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }

class  opts_ntu_120_cross_subject():
  def __init__(self):

  # graph based model
   self.agcn_model_args = {
      "num_class": 120,
      "num_point": 25,
      "num_person": 2,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

   #image based model
   self.hcn_model_args = {
      "in_channel":3,
      "out_channel":64,
      "window_size":64,
      "num_joint":25,
      "num_person":2,
      "num_class":120
    }

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":150,
      "en_hidden_size":1024,
      "en_num_layers":3,
      "num_class":120
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsub/train_data_joint.npy',
     'label_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsub/train_label.pkl',
     'num_frame_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsub/train_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsub/val_data_joint.npy',
     'label_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsub/val_label.pkl',
     'num_frame_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsub/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }

class  opts_ntu_120_cross_setup():

  def __init__(self):

  # graph based model
   self.agcn_model_args = {
      "num_class": 120,
      "num_point": 25,
      "num_person": 2,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

   #image based model
   self.hcn_model_args = {
      "in_channel":3,
      "out_channel":64,
      "window_size":64,
      "num_joint":25,
      "num_person":2,
      "num_class":120
    }

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":150,
      "en_hidden_size":1024,
      "en_num_layers":3,
      "num_class":120
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsetup/train_data_joint.npy',
     'label_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsetup/train_label.pkl',
     'num_frame_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsetup/train_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsetup/val_data_joint.npy',
     'label_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsetup/val_label.pkl',
     'num_frame_path': f'{dataroot}/NTU/NTU-RGB-D-120-AGCN-fixed2/xsetup/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }


class  opts_pku_v2_cross_subject():

  def __init__(self):

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":150,
      "en_hidden_size":1024,
      "en_num_layers":3,
      "num_class":120
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': f'{dataroot}/PKU/xsub/train_data_joint.npy',
     'label_path': f'{dataroot}/PKU/xsub/train_label.pkl',
     'num_frame_path': f'{dataroot}/PKU/xsub/train_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': f'{dataroot}/PKU/xsub/val_data_joint.npy',
     'label_path': f'{dataroot}/PKU/xsub/val_label.pkl',
     'num_frame_path': f'{dataroot}/PKU/xsub/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }

   self.agcn_model_args = {
      "num_class": 120,
      "num_point": 25,
      "num_person": 2,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

