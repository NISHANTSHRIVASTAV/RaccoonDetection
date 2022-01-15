# Vehicle Plate Detection 

Vehicle Plate Detector aims to classify and detect vehicle plate in an image with state of the art Mobilenet-SSD neural network architecture using PyTorch framework. 

## 🚀&nbsp; Installation

1. Clone the repo
```
$ git clone https://github.com/NISHANTSHRIVASTAV/VehiclePlateDetection.git
```

2. Change your directory to the cloned repo 
```
$ cd VehiclePlateDetection
```

3. Now, run the following command in the terminal/command prompt to install the required libraries
```
$ pip install -r requirements.txt
```

## :bulb: Working

### Generate Training Files

Go to dataset directory i.e. dataset\VOC2007 and run
```
$ python vision\datasets\generate_vocdata.py labels.txt
```
### Train the model with multiple GPUs
```
$ python train_ssd.py --dataset-type=voc --data=dataset\VOC2007\ --model-dir=models\custom_trained_model\ --batch-size=36 --epochs=150 --workers=0 --use-cuda=True --pretrained-ssd=models\pretrained_model\mobilenet-v1-ssd-mp-0_675.pth --gpu-devices 0 1 2 
```
### Inference/Prediction
To run the Vehicle Plate Detector Inference Engine which saves the response with bounding boxes in \results directory
```
$ python VehiclePlateDetection.py 
```

### Evaluation
To run the Vehicle Plate Detector Inference Engine
```
$ python eval_ssd.py --net=mb1-ssd --trained_model=models\custom_trained\mb1-ssd-Epoch-211-Loss-2.275852680206299.pth --dataset_type=voc --dataset=dataset\VOC2007\ --label_file=labels\labels.txt --use_cuda=True --eval_dir=eval 
```
## Results


Accuracy on Test Images - 78 % <br />

Total Test Images - 63
### On CPUs

Inference Time for each image - 0.140625 <br />
Batch wise Prediction Time with batch-size=30 
```json
{'batch_0': 4.4845263957977295, 'batch_1': 4.359519004821777, 'batch_2': 0.29688358306884766}
```

### On GPUs

Inference Time for each image - 0.015690 <br />
Batch wise Prediction Time with batch-size=30
```json
{'batch_0': 1.7811949253082275, 'batch_1': 0.45837974548339844, 'batch_2': 0.031238794326782227}
```