from pathlib import Path


dataset_paths = dict(
    colab='/content/wm',
    pc_thijs='G:/wm/',
    laptop_thijs='C:/wm',
    pc_francesco=r'/home/fmannella/Documents/Dataset',
    laptop_francesco=r'/home/fm/tmp',
    hbp_cluster='/opt/app-root/src/shared/WP2_T2.1_LivePaper_rootCollab',
)


# Loop over the project paths
user = None
for u, dpath in dataset_paths.items():
    if Path(dpath).is_dir():
        user = u

assert user is not None

if user == 'pc_thijs':
    datasetpath = 'G:/wm/Dataset'
    codepath = 'C:/Users/fpga/Desktop/WP2-T2.1-whisker-modelling'
    writepath = 'G:/wm/output'
    original_datadir = 'G:/data-vita'

elif user == 'colab':
    datasetpath = '/content/wm/Dataset'
    codepath = '/content/wm/Code'
    writepath = '/content'
    original_datadir = '/content/wm/Dataset'

elif user == 'laptop_thijs':
    datasetpath = r'C:\wm'
    codepath = 'C:/users/truikes/WP2-T2.1-whisker-modellig'
    writepath = 'C:/wm/output'
    original_datadir = None

elif user == 'pc_francesco':
    datasetpath = "/home/fmannella/Projects/current/whisk-modelling/Dataset"
    codepath = "/home/fm/wm/Code"
    writepath = "/home/fm/wm/tmp"
    original_datadir = None

elif user == 'laptop_francesco':
    datasetpath = "/home/fm/tmp"
    codepath = "/home/fm/tmp"
    writepath = "/home/fm/tmp"
    original_datadir = None

elif user == 'hbp_cluster':
    datasetpath = "/home/fm/wm/Dataset"
    codepath = "/home/fm/wm/Code"
    writepath = "/home/fm/wm/tmp"
    original_datadir = None


else:
    raise ValueError('Cannot find datasetpath, update params.py!')

datasetpath = Path(datasetpath)
codepath = Path(codepath)
writepath = Path(writepath)

