import argparse
import os
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.backend.tensorflow_backend import set_session

from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator import train_gen, valid_gen
from migrate import migrate_model
from segnet import build_encoder_decoder, build_refinement
from utils import alpha_prediction_loss, overall_loss, get_available_cpus, get_available_gpus

if __name__ == '__main__':
    # Defines amount of logging to do while training
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    checkpoint_models_path = 'models/'
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]

    # Callbacks
    # Tensorboard is currently excluded from our training
    # Incase you wish to use it, uncomment the following line and include tensorboard in callbacks variable
    # tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'final.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'final.%02d-%.4f.hdf5'
            self.model_to_save.save(fmt % (epoch, logs['val_loss']))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#     config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)
    
    # Load our model, added support for Multi-GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate X MB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14336)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    num_gpu = len(get_available_gpus())
    print("Number of GPUs:",num_gpu)
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            model = build_encoder_decoder()
            model = build_refinement(model)
            if pretrained_path is not None:
                model.load_weights(pretrained_path)
            else:
                migrate_model(model)

        final = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        model = build_encoder_decoder()
        final = build_refinement(model)
        if pretrained_path is not None:
            final.load_weights(pretrained_path)
        else:
            # If you don't have pretrained VGG dataset, comment out the following line
            migrate_model(final)
            
    sgd = keras.optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    decoder_target = tf.placeholder(dtype='float32', shape=(None, None, None, None))
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = False)
    final.compile(optimizer=sgd, loss=overall_loss, target_tensors=[decoder_target],options=run_opts)

    print(final.summary())
    
    # Currently the training has been set to use only 1 worker
    # However you can uncomment the following lines and pass
    # workers as a parameter to fit_generator and put use_multiprocssing as true
    # num_cpu = get_available_cpus()
    # workers = int(round(num_cpu / 2))
    
    # Final callbacks
    callbacks = [model_checkpoint, early_stop, reduce_lr]#, tensor_board]

    # Start Fine-tuning
    final.fit_generator(train_gen(),
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_gen(),
                        validation_steps=num_valid_samples // batch_size,
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        use_multiprocessing=False,
                        workers=1
                        )
    K.clear_session()


