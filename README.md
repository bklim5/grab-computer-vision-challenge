# grab-computer-vision-challenge
Car make/model recognition challenge. Dataset from https://ai.stanford.edu/~jkrause/cars/car_dataset.html

Using **fastai** library to finetune **resnet50** pretrained model on imagenet datasets, the highest accuracy obtain on the Stanford Car dataset is **86.69%**.

Technique used during training:
1. Learning rate finder
2. Fine-tuning of last layer
3. Unfreezing of earlier layers
4. Progressive resizing (from 244 -> 299 -> 480 -> (400, 600))

All these techniques / approaches were taught in fast.ai MOOC course.

## Run
To run the classifier, 
1. First download the trained weights and metadata from datasets/stanford-cars folder
    - export.pkl
    - car_id_to_car_class_mapping.json
    - car_class_to_car_id_mapping.json
    
2. Use **classify.py** to classify an single image or entire folder. 

```
ubuntu@ip-172-31-28-240:~/bk$ python classify.py --help

usage: classify.py [-h] [--input INPUT] [--output OUTPUT]

Script to classify car make and model

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Input image or directory of images to be classified
  --output OUTPUT  Output filename when classify a whole folder
```

Example:
1. To classify a single image
```
ubuntu@ip-172-31-28-240:~/bk$ python classify.py --input datasets/stanford-cars/cars_test/00001.jpg

==============================
Classifying image..
Prediction: Suzuki Aerio Sedan 2007
Confidence score: 0.9107944369316101
==============================
```

2. To classify entire folder and output predictions and confidence score to pred.csv
```
ubuntu@ip-172-31-28-240:~/bk$ python classify.py --input datasets/stanford-cars/cars_test/ --output pred.csv

==============================
Classifying folder..
Output prediction dataframe to pred.csv
==============================
```
The output CSV will contain the filename, predicted class, predicted class ID, confidence score for the predicted class and all other classes

| fname  | Predicted Car | Predicted Car ID | Confidence Level | AM General Hummer SUV 2000 | Acura Integra Type R 2001 | ... |  
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 00001.jpg  | Suzuki Aerio Sedan 2007  | 181 | 0.950445 | 7.04e-09 | 4.86e-04 | ... |
| 00002.jpg  | Ferrari California Convertible 2012  | 102 | 0.266090 | 5.81e-05 | 8.66e-04 | ... |
| ... | ...| ...| ...| ...| ...| ...|

## Web app
TODO

## Potential future improvement
1. To utilize the bounding boxes provided in training dataset to focus on the car image. Though that would require an object detection model to be trained as well to recognize the car object in test.
2. To test with different architecture than Resnet.
3. To add more training data especially for classes that the model confused the most. This can be easily found out using fastai utility function
   ```
   interp = ClassificationInterpretation.from_learner(learner)
   interp.plot_top_losses()
   interp.plot_confusion_matrix()
   ```
4. Add more training data from other sources, eg: http://vmmrdb.cecsresearch.org/


