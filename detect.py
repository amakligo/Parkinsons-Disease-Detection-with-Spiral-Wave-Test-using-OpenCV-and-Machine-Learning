# imports
import os
import cv2
import numpy as np
from imutils import build_montages
from imutils import paths
from skimage import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


class Program:
    def __init__(self):
        self.args = dict()
        self.args["trials"] = 10
        # for spiral use dataset/wave
        self.args["dataset"] = 'dataset/wave'
        self.initialized = True

    def __enter__(self):
        """
        :return:
        """
        try:
            if self.initialized:
                s, d = self.preflight()
                if d['status']:
                    print('pre-flight: ok')

                    s, d = self.flight(d["trials"], d["model"], d["le"])
                    if d['status']:
                        print('flight: ok')
                    else:
                        print('flight: failed')

                else:
                    print('pre-flight: failed')
            else:
                print('initialized: failed')
        except Exception as e:
            print(e)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def preflight(self):
        """
        :return:
        """
        status = False
        data = dict()
        data['status'] = False
        try:
            # define the path to the training and testing directories
            trainingPath = os.path.sep.join([self.args["dataset"], "training"])
            testingPath = os.path.sep.join([self.args["dataset"], "testing"])

            # loading the training and testing data
            print("[INFO] loading data...")
            (trainX, trainY) = self.load_split(trainingPath)
            (testX, testY) = self.load_split(testingPath)

            # encode the labels as integers
            data["le"] = LabelEncoder()
            trainY = data["le"].fit_transform(trainY)
            testY = data["le"].transform(testY)

            # initialize our trials dictionary
            data["trials"] = {}

            # loop over the number of trials to run
            for i in range(0, self.args["trials"]):
                # train the model
                print("[INFO] training model {} of {}...".format(i + 1,
                                                                 self.args["trials"]))
                data["model"] = RandomForestClassifier(n_estimators=100)
                data["model"].fit(trainX, trainY)

                # make predictions on the testing data and initialize a dictionary
                # to store our computed metrics
                predictions = data["model"].predict(testX)
                metrics = {}

                # compute the confusion matrix and and use it to derive the raw
                # accuracy, sensitivity, and specificity
                cm = confusion_matrix(testY, predictions).flatten()
                (tn, fp, fn, tp) = cm
                metrics["accuracy"] = (tp + tn) / float(cm.sum())
                metrics["true-positive"] = tp / float(tp + fn)
                metrics["true-negative"] = tn / float(tn + fp)

                # loop over the metrics
                for (k, v) in metrics.items():
                    # update the trials dictionary with the list of values for
                    # the current metric
                    lst = data["trials"].get(k, [])
                    lst.append(v)
                    data["trials"][k] = lst
                data["status"] = True
                status = True
        except Exception as e:
            print(e)
        return status, data

    def flight(self, trials, model, le):
        """
        :return:
        """
        status = False
        data = dict()
        data['status'] = False
        try:
            testingPath = os.path.sep.join([self.args["dataset"], "testing"])
            # loop over our metrics
            for metric in ("accuracy", "true-positive", "true-negative"):
                # grab the list of values for the current metric, then compute
                # the mean and standard deviation
                values = trials[metric]
                mean = np.mean(values)
                std = np.std(values)

                # show the computed metrics for the statistic
                print(metric)
                print("=" * len(metric))
                print("mean = {:.4f}, std = {:.4f}".format(mean, std))
                print("")

            # randomly select a few images and then initialize the output images
            # for the montage
            testingPaths = list(paths.list_images(testingPath))
            idxs = np.arange(0, len(testingPaths))
            idxs = np.random.choice(idxs, size=(25,), replace=False)

            images = []

            # loop over the testing samples
            for i in idxs:
                # load the testing image, clone it, and resize it
                image = cv2.imread(testingPaths[i])
                output = image.copy()
                output = cv2.resize(output, (128, 128))

                # pre-process the image in the same manner we did earlier
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (200, 200))
                image = cv2.threshold(image, 0, 255,
                                      cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                # quantify the image and make predictions based on the extracted
                # features using the last trained Random Forest
                features = self.quantify_image(image)
                preds = model.predict([features])
                label = le.inverse_transform(preds)[0]

                # draw the colored class label on the output image and add it to
                # the set of output images
                color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
                cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
                images.append(output)

            # create a montage using 128x128 "tiles" with 5 rows and 5 columns
            montage = build_montages(images, (128, 128), (5, 5))[0]

            # show the output montage
            cv2.imshow("Output", montage)
            cv2.waitKey(0)
            data["status"] = True
            status = True
        except Exception as e:
            print(e)
        return status, data

    @staticmethod
    def quantify_image(img):
        """
        This function will be used to extract features from each input image
        :param img:
        :return:

        [[credits::: Introduced by Dalal and Triggs in their CVPR 2005 paper,
        Histogram of Oriented Gradients for Human Detection]]
        """
        # compute the histogram of oriented gradients feature vector for
        # the input image
        feats = feature.hog(img, orientations=9,
                            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1")
        # return the feature vector
        return feats

    def load_split(self, path):
        """
         This function has a goal of accepting a dataset path and returning all feature data and associated class labels
        :param path:
        :return:
        """
        # grab the list of images in the input directory, then initialize
        # the list of data (i.e., images) and class labels
        imagePaths = list(paths.list_images(path))
        featdata = []
        featlabels = []
        # loop over the image paths
        for imagePath in imagePaths:
            # extract the class label from the filename
            lableid = imagePath.split(os.path.sep)[-2]

            # load the input image, convert it to grayscale, and resize
            # it to 200x200 pixels, ignoring aspect ratio
            imge = cv2.imread(imagePath)
            imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
            imge = cv2.resize(imge, (200, 200))

            # threshold the image such that the drawing appears as white
            # on a black background
            imge = cv2.threshold(imge, 0, 255,
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # quantify the image
            feats = self.quantify_image(imge)

            # update the data and labels lists, respectively
            featdata.append(feats)
            featlabels.append(lableid)

        # return the data and labels
        return np.array(featdata), np.array(featlabels)


if __name__ == "__main__":
    """
    """
    with Program():
        pass


