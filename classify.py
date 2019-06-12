import os
import json
import argparse
import pandas as pd
import numpy as np
from fastai.vision import ImageList, open_image
from fastai.basic_train import load_learner
from fastai.basic_data import DatasetType


BASE_FOLDER = 'datasets/stanford-cars'

learner = load_learner(path=BASE_FOLDER, file='export.pkl')
with open('{}/car_id_to_car_class_mapping.json'.format(BASE_FOLDER)) as f:
    car_id_to_car_class_mapping = json.load(f)

with open('{}/car_class_to_car_id_mapping.json'.format(BASE_FOLDER)) as f:
    car_class_to_car_id_mapping = json.load(f)


def classify_folder(input_folder, output):
    learner.data.add_test(ImageList.from_folder(input_folder))
    print('==============================')
    print('Classifying folder..')
    preds, _ = learner.get_preds(ds_type=DatasetType.Test)
    prediction_idx = np.argmax(preds, axis=1)
    data_class = learner.data.classes
    predicted_classes = [data_class[idx] for idx in prediction_idx]
    predicted_class_id = [car_class_to_car_id_mapping[
        pred_class] for pred_class in predicted_classes]
    confidence_level = [prob[prediction_idx[
        i]].item() for i, prob in enumerate(preds)]

    filenames = [str(fn).split('/')[-1] for fn in learner.data.test_ds.x.items]

    prob_df = pd.DataFrame(np.asarray(preds), columns=data_class)

    index_df = pd.DataFrame({
        'fname': filenames,
        'Predicted Car': predicted_classes,
        'Predicted Car ID': predicted_class_id,
        'Confidence Level': confidence_level
    })

    final_df = pd.concat([index_df, prob_df], axis=1)
    final_df = final_df.sort_values(by=['fname'])

    print('Output prediction dataframe to {}'.format(output))
    final_df.to_csv(output, index=False)
    print('==============================')


def classify_image(input_image):
    print('==============================')
    print('Classifying image..')
    prediction, idx, probs = learner.predict(open_image(input_image))
    print('Prediction: {}'.format(prediction))
    print('Confidence score: {}'.format(probs[idx]))
    print('==============================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to classify car make and model'
    )
    parser.add_argument(
        '--input',
        help='Input image or directory of images to be classified'
    )
    parser.add_argument(
        '--output',
        help='Output filename when classify a whole folder',
        default='predictions.csv'
    )
    args = parser.parse_args()

    inp = args.input
    out = args.output
    if os.path.isdir(inp):
        classify_folder(inp, out)

    if os.path.isfile(inp):
        classify_image(inp)

