import os
import json
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc


def test(test_filenames, params, scene_name, model_path):
    models_module = __import__('.'.join(['models', params.model, 'estimator_def']))
    model = getattr(models_module, params.model)
    estimator_def = model.estimator_def
    
    input_fn_test = estimator_def.get_eval_input_fn(test_filenames, 1)
    test_inputs_op, test_labels_op = input_fn_test()
    test_preds_op = estimator_def.model_fn(
        test_inputs_op, 
        test_labels_op, 
        tf.estimator.ModeKeys.PREDICT, 
        params.model_params
        ).predictions


    pred_list = []
    target_list = []
    label_list = []

    saver = tf.train.Saver()
    # Counter just so we know what example is being evaluated. 
    ex_count = 0
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        while True:
            try:
                test_inputs, test_preds, test_labels= sess.run([
                    test_inputs_op, # The targets are the inputs
                    test_preds_op, 
                    test_labels_op
                    ]
                )
                
                target_list.append(test_preds['targets'])
                pred_list.append(test_preds['predictions'])
                label_list.append(test_labels[0])
                ex_count+=1
                print('Testing example', ex_count)
            except tf.errors.OutOfRangeError:
                break



    np.save(
        os.path.join(params.preds_dir, 'preds_test_{}_{}'.format(params.model, scene_name)), 
        np.array(pred_list)
        )   

    np.save(
        os.path.join(params.targets_dir, 'targets_test_{}_{}'.format(params.model, scene_name)), 
        np.array(target_list)
        )

    np.save(
        os.path.join(params.labels_dir, 'label_list_{}_{}'.format(params.model,scene_name)), 
        np.array(label_list)
        )
