from src.models.model_preparation_saving import prepare_model_target, prepare_model_source, create_upload_zip, save_pred_model
from src.models.tf_generators_models_kfold import create_model, compute_class_weights
import numpy as np
from src.evaluation.auc_evaluation import calculate_AUC
from tensorflow.keras import callbacks
from numpy.random import seed
import tensorflow as tf
import csv

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


seed(1)
tf.random.set_seed(2)

# Source STL10
target = True
# define src data
source_data = "textures"
# define target dataset
target_data = 'isic'
x_col = 'path'
y_col = 'class'
augment = True
k = 5
img_length = 112
img_width = 112
learning_rate = 0.0001
batch_size = 128
epochs = 50
color = True
dropout = 0.5
scheduler_bool = False
home = './data'
imagenet = True


class MetricsLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, _run):
        super().__init__()
        self._run = _run

    def on_epoch_end(self, _, logs):
        self._run.log_scalar("training.loss", logs.get('loss'))
        self._run.log_scalar("training.acc", logs.get('accuracy'))
        self._run.log_scalar("validation.loss", logs.get('val_loss'))
        self._run.log_scalar("validation.acc", logs.get('val_accuracy'))


def scheduler(epochs, learning_rate):
    if epochs < 20:
        return learning_rate
    else:
        return learning_rate * 0.5

'''
if scheduler_bool:
    # add learning rate scheduler in callbacks of model
    callbacks_settings = [MetricsLoggerCallback(_run),
                          callbacks.LearningRateScheduler(scheduler)]
else:
    callbacks_settings = [MetricsLoggerCallback(_run)]
'''

if target:
    # collect all objects needed to prepare model for transfer learning
    dataframe, num_classes, x_col, y_col, class_mode, skf, train_gen, valid_gen = prepare_model_target(home,
                                                                                                       source_data,
                                                                                                       target_data,
                                                                                                       x_col,
                                                                                                       y_col,
                                                                                                       augment,
                                                                                                       k)

    # initialize empty lists storing accuracy, loss and multi-class auc per fold
    acc_per_fold = []
    loss_per_fold = []
    auc_per_fold = []

    fold_no = 1  # initialize fold counter

    for train_index, val_index in skf.split(np.zeros(len(dataframe)), y=dataframe[['class']]):
        print(f'Starting fold {fold_no}')

        train_data = dataframe.iloc[train_index]  # create training dataframe with indices from fold split
        valid_data = dataframe.iloc[val_index]  # create validation dataframe with indices from fold split

        train_generator = train_gen.flow_from_dataframe(dataframe=train_data,
                                                        x_col=x_col,
                                                        y_col=y_col,
                                                        target_size=(img_length, img_width),
                                                        batch_size=batch_size,
                                                        class_mode=class_mode,
                                                        seed=2,
                                                        validate_filenames=False)

        valid_generator = valid_gen.flow_from_dataframe(dataframe=valid_data,
                                                        x_col=x_col,
                                                        y_col=y_col,
                                                        target_size=(img_length, img_width),
                                                        batch_size=batch_size,
                                                        class_mode=class_mode,
                                                        validate_filenames=False,
                                                        seed=2,
                                                        shuffle=False)

        model = create_model(source_data, target_data, num_classes, learning_rate, img_length, img_width, color,
                             dropout)  # create model using the compilation settings and image information

        class_weights = compute_class_weights(train_generator.classes)  # get class model_weights to balance classes

        model.fit(train_generator,
                  steps_per_epoch=train_generator.samples // batch_size,
                  epochs=epochs,
                  class_weight=class_weights,
                  validation_data=valid_generator,
                  validation_steps=valid_generator.samples // batch_size)
                  #callbacks=callbacks_settings)

        # compute loss and accuracy on validation set
        valid_loss, valid_acc = model.evaluate(valid_generator, verbose=1)
        print(f'Validation loss for fold {fold_no}: {valid_loss}', f' and Validation accuracy for fold {fold_no}: '
                                                                   f'{valid_acc}')
        acc_per_fold.append(valid_acc)
        loss_per_fold.append(valid_loss)

        predictions = model.predict(valid_generator)  # get predictions

        # calculate auc-score using y_true and predictions
        auc = calculate_AUC(target_data, valid_generator, predictions)
        auc_per_fold.append(auc)

        # save predictions and models_base in local memory
        save_pred_model(source_data, target_data, fold_no, model, predictions)

        fold_no += 1

    # create zip file with predictions and models_base and upload to OSF
    create_upload_zip(k, source_data, target_data)

    # compute average scores for accuracy, loss and auc
    print('Accuracy, loss and AUC per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%, - AUC: {auc_per_fold[i]}')
    print('Average accuracy, loss and AUC for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)})')
    print(f'> AUC: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)})')

    print( acc_per_fold, loss_per_fold, auc_per_fold)
    
    #auc_per_fold.append(np.mean(auc_per_fold))
    #auc_per_fold.append(np.std(auc_per_fold))
            
    #with open('../results/aucs_rep.csv','a') as f:
        #wr = csv.writer(f, dialect='excel')
        #wr.writerow([source_data, target_data]+auc_per_fold)
    
else:
    # collect all objects needed to prepare model for pretraining
    num_classes, train_generator, valid_generator, test_generator = prepare_model_source(home,
                                                                                         source_data,
                                                                                         target_data,
                                                                                         augment,
                                                                                         batch_size,
                                                                                         img_length,
                                                                                         img_width)

    model = create_model(source_data, target_data, num_classes, learning_rate, img_length, img_width, color,
                         dropout)  # create model using the compilation settings and image information

    # class_weights = compute_class_weights(train_generator.classes)  # get class model_weights to balance classes

    model.fit(train_generator,
              epochs=epochs,
              #class_weight=class_weights,
              validation_data=valid_generator,
              #callbacks=callbacks_settings
              )

    # compute loss and accuracy on validation set
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f'Test loss:', test_loss, f' and Test accuracy:', test_acc)

    # save model model_weights
    model.save(f'model_weights_resnet_pretrained={source_data}.h5')

    create_upload_zip(k, source_data, target_data)

    print(test_loss, test_acc)
