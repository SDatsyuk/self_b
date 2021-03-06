import cv2
import os
import tensorflow as tf
import time
import argparse
import uuid
import glob
import numpy as np
import shutil

import barcode_reader
from transfLearning import transflearn
from transfLearning import cluster_vectors
from transfLearning.model_builder import build_model

from config import Config as conf
from utils.extract_background import extract_background

products = {"4823063105439": "sadochok",
            "4820048190138": "bread",
            "4823077616167": "englishClub",
            "4823077616150": "englishClubHoney",
            "4820104250554": "ananas",
            "4823012232124": "konti"}
vector_ext = ".npz"
ness_folders = ['transfLearning/image_vectors', "transfLearning/images", "transfLearning/NNs", "new_image"]

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--take_photos", action='store_true', help='create new product class and take a photos')
ap.add_argument("-c", "--convert_new_photos", action="store_true", help="convert photos from folder `new` to vector")
ap.add_argument("-t", "--test", action='store_true', help='start test for model')
ap.add_argument('-s', "--save_path", default="new_image", help="images store")
ap.add_argument('-m', "--model", help='Model type', default="InceptionV4")
ap.add_argument('-v', '--vector_path', help="vector store path", default="transfLearning/image_vectors")
ap.add_argument("-i", "--camera_id", default=0, help="set camera id to use")

args = vars(ap.parse_args())
print(args)

capture_zone = None #[100, 200, 200, 250]


def check_folders(path_list):
    if not isinstance(path_list, list):
        path_list = [path_list]
    for i in path_list:
        if not os.path.exists(i):
            os.makedirs(i)

def create_back_image():
    print('Press `f` to create background image')
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = frame[:,:440]


        cv2.imshow('background', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            back = frame
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def take_photos(back):
    print("Press `f` to take photo from camera. Photos stores in folder `new`. To finish press `q`.")

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    cap = cv2.VideoCapture(args['camera_id'])

    while True:
        category = input("Enter class name (enter `q` to stop process): ")
        if category == 'q':
            break
        while True:
            _, frame = cap.read()
            frame = frame[:,:440]
            obj = extract_background(back, frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            (x, y, w, h) = cv2.boundingRect(obj)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(30)

            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('f'):

                file_path = os.path.join(args['save_path'], "{}_{}.jpg".format(category, str(uuid.uuid4())))
                cv2.imwrite(file_path, frame[y:y+h, x:x+w])
                print("%s image saved to %s" % (category, file_path))

    cap.release()
    cv2.destroyAllWindows()

def convert_images_to_vector(model):

    image_list = glob.glob1(args['save_path'], '*.jpg')
    if len(image_list) == 0:
        print("No images for converting in folder. Use `-p` flag to create photos")
        return False

    for i in image_list:
        img = cv2.imread(os.path.join(args['save_path'], i))
        print("Converting %s" % i)
        feature_vector = model.run_inference_on_image(cv2.resize(img, (299, 299)))
        out_path = os.path.join(args['vector_path'], args['model'])
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.savetxt(os.path.join(out_path, os.path.splitext(i)[0] + vector_ext), feature_vector, delimiter=',')
        dest_path = os.path.join('transfLearning/images', i.split("_")[0])
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        shutil.copyfile(os.path.join(args['save_path'], i), os.path.join(dest_path, i))
        os.remove(os.path.join(args['save_path'], i))
    print('Images converted succesfully')
    # del model
    return True


def reco_process(model, back):
    
    camera = cv2.VideoCapture(args['camera_id'])
    camera.set(1, 60)
    camera.set(3, 640)
    camera.set(4, 480)

    
    ann_tree = cluster_vectors.build_ann_index(path=os.path.join("transfLearning/image_vectors", args['model'], '*.npz'), dims=model.layer_dims)
    print(ann_tree)

    while True:
        status, frame = camera.read()

        # if capture_zone:
            # frame = frame[capture_zone[0]: capture_zone[1], capture_zone[2]: capture_zone[3]]
        back = back[:, :400]
        frame = frame[:, :400]
        obj = extract_background(back, frame)
        (x, y, w, h) = cv2.boundingRect(obj)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # barcodes = barcode_reader.find_barcode(frame)
        # barcodes = [bar for bar in barcodes if bar.data.decode() in products]

        # viz_boxes = barcode_reader.viz_barcodes(frame.copy(), barcodes)
        # if barcodes:
        st_time = time.time()
        feature_vector = model.run_inference_on_image(frame[y:y+h, x:x+w])
        # print(feature_vector.shape)

        nearest_neighbors = cluster_vectors.nearest_neighbors(ann_tree, feature_vector)
        # print(nearest_neighbors)
        
        rec_prod = sorted(nearest_neighbors, key=lambda k: k['similarity'])[0]
        rec_class = rec_prod['filename'].split('\\')[-1].split('_')[0]

        cv2.putText(frame, '%s: %s' % (rec_class, rec_prod['similarity']), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("reco", frame)
        
    #     if rec_class != products[barcodes[0].data.decode()]:
    #         cv2.putText(viz_boxes, 'wrong product: %s' % rec_class, (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #     else:
    #         cv2.putText(viz_boxes, '%s' % rec_class, (20, 10), cv2.FONT_HERSHEY_SIMPLEX,
    # 0.5, (0, 255, 0), 2)


        fn_time = time.time()
        # print("Time spend: %s" % (fn_time - st_time))

        # cv2.imshow("ss", viz_boxes)
        
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


def main():
    model = build_model(args['model'])

    check_folders(ness_folders)
    background = create_back_image()

    while True:
        print("1 - take new photos")
        print("2 - make vectors from image")
        print("3 - test NN")
        ch = input("Enter (q - exit): ")
        if ch not in ['1', '2', '3']:
            if ch == 'q':
                break
            raise ValueError("Enter only 1, 2, 3")
        if ch == '1':
            take_photos(background)
        elif ch == '2':
            ret = convert_images_to_vector(model)
        else:
            reco_process(model, background)


    # cv2.destroyAllWindows()

    # if args["take_photos"]:
    #   take_photos()
    # if args['convert_new_photos']:
    #   ret = convert_images_to_vector(model)
    # if args['test']:
    #   reco_process(model)


if __name__ == "__main__":
    main()