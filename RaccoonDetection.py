import cv2
import uuid
import numpy
import logging
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

MODEL_FILE_PATH = 'models\\custom_trained_model\\mb1-ssd-Epoch-165-Loss-1.3442652225494385.pth'
LABEL_FILE_PATH = 'labels\\labels.txt'
LOG_FILE_PATH = 'logs\\raccoon_detector_logger.log'
RUNTIME_DEVICE_TYPE = 'cpu'

logging.basicConfig(filename=LOG_FILE_PATH, filemode='a', level=logging.INFO,\
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info('Raccoon Detector Inference Engine')

class RaccoonDetector():

    def __init__(self, image=None, nms_threshold=0.50, net_type='mb1-ssd', device='cpu'):
        
        try:
            self.image = image
            self.net_type = net_type
            self.nms_threshold = nms_threshold
            self.class_names = [name.strip() for name in open(LABEL_FILE_PATH).readlines()]
            self.timer = Timer()

            if self.net_type == 'vgg16-ssd':
                self.net = create_vgg_ssd(len(self.class_names ), is_test=True, device=device)
            elif self.net_type == 'mb1-ssd':
                self.net = create_mobilenetv1_ssd(len(self.class_names ), is_test=True, device=device)
            elif self.net_type == 'mb1-ssd-lite':
                self.net = create_mobilenetv1_ssd_lite(len(self.class_names ), is_test=True, device=device)
            elif self.net_type == 'mb2-ssd-lite':
                self.net = create_mobilenetv2_ssd_lite(len(self.class_names ), is_test=True, device=device)
            elif self.net_type == 'sq-ssd-lite':
                self.net = create_squeezenet_ssd_lite(len(self.class_names ), is_test=True, device=device)
            else:
                print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")

            self.net.load(MODEL_FILE_PATH)

            if self.net_type == 'vgg16-ssd':
                self.predictor = create_vgg_ssd_predictor(self.net, candidate_size=200, device=device)
            elif self.net_type == 'mb1-ssd':
                self.predictor = create_mobilenetv1_ssd_predictor(self.net, candidate_size=200, device=device)
            elif self.net_type == 'mb1-ssd-lite':
                self.predictor = create_mobilenetv1_ssd_lite_predictor(self.net, candidate_size=200, device=device)
            elif self.net_type == 'mb2-ssd-lite':
                self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200, device=device)
            elif self.net_type == 'sq-ssd-lite':
                self.predictor = create_squeezenet_ssd_lite_predictor(self.net, candidate_size=200, device=device)
            else:
                self.predictor = create_vgg_ssd_predictor(self.net, candidate_size=200, device=device)
        except Exception as e:
            logging.error("Caught an exception in RaccoonDetector constructor", exc_info=True)
            print("Caught an exception in RaccoonDetector constructor", e)

    def detect_objects(self):
        
        try:
            self.timer.start("Load Image")
            npimg = numpy.fromstring(self.image, numpy.uint8)
            orig_image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            load_end_time = self.timer.end("Load Image")
            print("Load Image: {:4f} seconds.".format(load_end_time))
            logging.info(f"Load Image seconds {load_end_time}")

            self.timer.start("Predict")
            boxes, labels, probs = self.predictor.predict(orig_image, 10, self.nms_threshold)
            predict_end_time = self.timer.end("Predict")
            print("Prediction: {:4f} seconds.".format(predict_end_time))
            logging.info(f"Prediction seconds {predict_end_time}")
            print("---------------------------------------------------------------------")
            predictions = []
            pred_label_count = {}
            
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                c1 = int(box[0].numpy())
                c2 = int(box[1].numpy())
                c3 = int(box[2].numpy())
                c4 = int(box[3].numpy())
                confidence = str(float(probs[i])*100)
                pred_label = str(self.class_names [labels[i]])
                
                if pred_label in pred_label_count:
                    prev_pred_label_count = pred_label_count[pred_label]
                    pred_label_count[pred_label] = prev_pred_label_count + 1
                else:
                    pred_label_count[pred_label] = 1
                   
                cv2.rectangle(orig_image, (c1, c2), (c3, c4), (255, 255, 0), 4)
                label = f"{self.class_names [labels[i]]}: {probs[i]:.2f}"
                cv2.putText(orig_image, label, (c1+30, c2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                predictions.append({'confidence':confidence, 'x1':c1, 'y1':c2, 'x2':c3, 'y2':c4, 'label': pred_label})
            
            filename = str(uuid.uuid1()) + '.jpg'
            file_save_path = 'results\\' + filename
            cv2.imwrite(file_save_path, orig_image)
           
            result = {'status':'ok', 'error': False, 'output_image': filename, 'predictions': predictions,\
                    'prediction_count': pred_label_count, 'inference_time': round(predict_end_time, 2)}
            return result
        except Exception as e:
            logging.error("Caught an exception in detect_objects", exc_info=True)
            print("Caught an exception in detect_objects", e)