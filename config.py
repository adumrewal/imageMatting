img_rows, img_cols = 320, 320
# img_rows_half, img_cols_half = 160, 160
channel = 4
batch_size = 8
epochs = 1000
patience = 50
num_samples = 30
num_train_samples = 20
# num_samples - num_train_samples
num_valid_samples = 10
unknown_code = 128
epsilon = 1e-6
epsilon_sqr = epsilon ** 2

##############################################################
# Set your paths here

# path to dataset folder
dataset_path = './data/'

# Path to folder where you want the composited images to go
image_path = dataset_path+'input/'

# path to provided alpha mattes
a_path = dataset_path+'mask/'

##############################################################
