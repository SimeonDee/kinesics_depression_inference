### GENERAL LIBS  ###
# Data Analysis Libraries
from nltk import text
from keras.utils import load_img, img_to_array
from keras.models import load_model
import pydot
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, Adam
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPool2D, AveragePooling2D
from keras.layers import Conv2D
from keras.models import Sequential
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS  # Google colab
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat
import random
# %matplotlib inline

import os
import shutil
from os import listdir, path

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Models used
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score

# Saving Trained Models
import pickle
import joblib


### NLP LIBS ###
# Text Preprocessing Libs
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
# Importing NLP Text Preprocessing Libraries
#!pip install nltk
import re
import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
nltk.download('wordnet')

# Word Analysis Lib
#!pip install wordcloud

# An alternative library of English stop-words in "sklearn" library rather than 'nltk'
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


# CNN IMAGE LIBS
# CNN keras libs


# Visualizing Model Architecture

# loading saved cnn model


class KinesicsHybridModel:

    __INPUT_SIZE = (48, 48)
    __BATCH_SIZE = 128
    __N_EPOCHS = 40
    __METRICS = ["Accuracy", "Precision", "Recall", "ROC-AUC", "F-Measure"]

    def __init__(self, model_dir_or_url=None):
        """
          model_dir_or_url: the folder containing the trained models or url pointing to the location or endpoint of the model
        """
        self.default_dir = "/content/gdrive/My Drive/Datasets/Tolu_OAU_Depression_Datasets/Datasets/"
        self.model_dir = model_dir_or_url or f'{self.default_dir}results/models/'
        # Loading models
        self.trained_text_model, self.trained_image_model = self.load_models(
            text_model_name='SVC_Model_v2.sav', image_model_name='CNN_Model_v1.h5')

    def set_model_directory(self, model_uri):
        self.model_dir = model_uri

    def load_models(self, text_model_name, image_model_name):
        """
          Returns a tupple containing trained models: (textmodel, imagemodel)
        """
        try:
            text_model_path = path.join(self.model_dir, text_model_name)
            image_model_path = path.join(self.model_dir, image_model_name)

            text_model = self.__load_model_pickle(text_model_path)
            image_model = load_model(image_model_path)

            return text_model, image_model

        except Exception as err:
            print(f'Sorry, an error occured. Could not load model...\n{err}')
            return None, None

    # UTILITY FUNCTIONS HERE (Private)

    def __cleanPost(self, postText, to_lower=True, rmv_eng_stop_words=True, stem=True):
        # Removing all numbers, punctuations and special characters, and links. Extract only alphabets
        post = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', postText)

        if to_lower:
            # Converting all text to lowercase
            post = post.lower()
            post = post.split()

        if rmv_eng_stop_words:
            eng_stop_words = set(ENGLISH_STOP_WORDS)
            stop_word_exceptions = ['alone', 'most', 'everything', 'down', 'very', 'fire',
                                    'am', 'almost', 'mostly', 'enough', 'serious', 'nobody',
                                          'cry', 'against', 'more', 'much', 'empty', 'my', 'often',
                                          'always', 'nothing', 'me', 'nowhere']

            for word in stop_word_exceptions:
                eng_stop_words.remove(word)

            # Remove English stop-words
            post = [word for word in post if (word not in eng_stop_words)]

        if stem:
            # Stemming (Using only the root word of every polymorphic words. e.g. Loved, Loving = Love; Eat, Ate, Eaten = Eat; etc)
            # ps = PorterStemmer()
            lemmatizer = WordNetLemmatizer()

            # Removing all common words e.g. Preposition, article, conjunction, etc.
            # post = [ps.stem(word) for word in post if word not in set(stopwords.words('english'))]
            # post = [ps.stem(word) for word in post if word not in set(ENGLISH_STOP_WORDS)]
            post = [lemmatizer.lemmatize(word, pos='n') for word in post]
            post = [lemmatizer.lemmatize(word, pos='v') for word in post]
            post = [word for word in post if len(word) > 2]

        post = ' '.join(post)

        return post

    def __encode_text(self, text_data, col_name=None):
        """
          text_data: {str / list of string values / Series of string values}
        """
        # Generating Corpus
        if (type(text_data) == str or type(text_data) == list):
            corpus_text = pd.Series(data=text_data, dtype='U')

        elif type(text_data) == pd.Series:
            corpus_text = text_data.values.astype('U')

        elif type(text_data) == pd.DataFrame and col_name is not None:
            if col_name in text_data.columns:
                corpus_text = text_data[text_data[col_name]].values.astype('U')
            else:
                raise Exception(f'{col_name} cannot be found in \'text_data\'')
                return None
        else:
            raise Exception(
                f'error occured ecoding text. Text data of type str or pandas Series or pandas DataFreame expected, Unrecognized data format receiced.')
            return None

        # loading trained encoder
        tfidf = self.__load_model_pickle(
            path.join(self.model_dir, 'tfidf_encoding_model.sav'))

        tfidf_encoded = tfidf.transform(corpus_text)
        print(f'TfIdf Encoded shape: {tfidf_encoded.shape}')

        return tfidf_encoded

    def __predict_text_post(self, textPost):
        classes = ['non-depressed', 'depressed']
        if type(textPost) == str:
            cleaned_post = self.__cleanPost(textPost)
            encoded_post = self.__encode_text(text_data=cleaned_post)

            # pred_lbl = self.trained_text_model.predict(encoded_post.toarray())
            probability = self.trained_text_model.predict_proba(
                encoded_post.toarray())

            pred_val = classes[probability.argmax()]
            confidence = round(probability.max() * 100, 2)
            hybrid_pred_prob = {'probability': probability, 'classes': ['non-depressed', 'depressed'],
                                'prediction': pred_val, 'confidence': confidence}
            return hybrid_pred_prob

        elif type(textPost) == list:
            cleaned_posts = list(map(self.__cleanPost, textPost))
            encoded_posts = self.__encode_text(text_data=cleaned_posts)
            hybrid_pred_probs = []

            # pred_lbls = self.trained_text_model.predict(encoded_posts.toarray())
            probabilities = self.trained_text_model.predict_proba(
                encoded_posts.toarray())

            for prob in probabilities:  # for every predicted value
                pred_val = classes[prob.argmax()]
                confidence = round(prob.max() * 100, 2)

                hybrid_pred_probs.append(
                    {'probability': prob, 'prediction': pred_val, 'confidence': confidence})

            return hybrid_pred_probs

        else:
            cleaned_posts = textPost.apply(self.__cleanPost)
            encoded_posts = self.__encode_text(text_data=cleaned_posts)
            hybrid_pred_probs = []

            # pred_lbls = self.trained_text_model.predict(encoded_posts.toarray())
            probabilities = self.trained_text_model.predict_proba(
                encoded_posts.toarray())

            for prob in probabilities:  # for every predicted value
                pred_val = classes[prob.argmax()]
                confidence = round(prob.max() * 100, 2)

                hybrid_pred_probs.append(
                    {'probability': prob, 'prediction': pred_val, 'confidence': confidence})

            return hybrid_pred_probs

    def plot_conf_normalize_percentage(self, conf_mat):
        norm_cf = np.zeros(shape=(2, 2), dtype=np.float64)
        for row in range(2):
            for col in range(2):
                norm_cf[row][col] = conf_mat[row][col] / conf_mat[row].sum()

        return norm_cf

    def plot_confusion_matrix(self, true_val, predicted_val, title='Title of Plot', perc=False, model_label=None):
        """
          model_label can be a string. Possible values are 'CNN': for CNN, 'RF': Random Forest, 'SVC': for SVC
        """
        cfm = confusion_matrix(true_val, predicted_val)
        xyticks = list(np.array(list(range(cfm.shape[0]))) + .5)

        if not perc:
            print()
            plt.figure(figsize=(6, 5))
            sns.heatmap(cfm, annot=True, fmt='d', cmap=plt.cm.Blues)

        else:
            print()
            cf_perc = self.plot_conf_normalize_percentage(cfm) * 100

            plt.figure(figsize=(6, 5))
            sns.heatmap(cf_perc, annot=True, fmt='.1f', cmap=plt.cm.Blues)

        if model_label in ['CNN', 'cnn']:
            plt.xticks(rotation=45, ticks=xyticks, labels=[
                       'depressed', 'non-depressed'])
            plt.yticks(rotation=0, ticks=xyticks, labels=[
                       'depressed', 'non-depressed'])
        else:
            plt.xticks(rotation=45, ticks=xyticks, labels=[
                       'non-depressed', 'depressed'])
            plt.yticks(rotation=0, ticks=xyticks, labels=[
                       'non-depressed', 'depressed'])

        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

    def evaluate_performance(self, true_val, predicted_val, label='Score', model_label=None, show_conf_mat=False):
        evaluations = [round(accuracy_score(true_val, predicted_val), 2),
                       round(precision_score(true_val, predicted_val), 2),
                       round(recall_score(true_val, predicted_val), 2),
                       #  round(r2_score(true_val, predicted_val),2),
                       round(roc_auc_score(true_val, predicted_val), 2),
                       round(f1_score(true_val, predicted_val), 2)
                       ]

        eval_result = pd.DataFrame(
            evaluations, index=self.__METRICS, columns=[label])

        if show_conf_mat:
            self.plot_confusion_matrix(
                true_val, predicted_val, model_label=model_label)

            print()
            self.plot_confusion_matrix(
                true_val, predicted_val, model_label=model_label, perc=True)

        return eval_result

    def plot_bar(self, df, title):
        df.T.sort_values(by='Recall', ascending=False).T.plot.bar(figsize=(15, 8),
                                                                  title=title,
                                                                  fontsize=12,
                                                                  rot=45,
                                                                  grid=True
                                                                  )

        plt.ylim(ymax=1.1)
        plt.legend(loc='upper center')
        plt.grid(which='major', axis='x')
        plt.xlabel('METRICS')
        plt.ylabel('SCORES')

    # def __save_model_pickle(self, model, filename):
    #   try:
    #     pickle.dump(model, open(f'{MODEL_PATH}{filename}', 'wb'))
    #     print('Saved')
    #   except Exception as err:
    #     print(err)

    def __load_model_pickle(self, filename):
        try:
            print(f'Model path: .....................{filename}')
            model = pickle.load(open(filename, 'rb'))
            return model
        except Exception as err:
            print(err)
            return None

    # =============== Utilities for Image with CNN =======================
    def __load_image(self, img_path, show=False):
        img = load_img(img_path, target_size=(48, 48), color_mode='rgb')

        # (height,width,channels)
        img_tensor = img_to_array(img)

        # img_tensor = np.vstack([img_tensor])

        # (1,height,width,channels), adds a dim coz model expects shape: (batch_size,height,width,channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # imshow expects values in range[0,1]
        img_tensor = img_tensor / 255.

        if show:
            plt.imshow(img_tensor[0])
            plt.axis('off')
            plt.show()

        return img_tensor

    def get_labels_codes_from_filepath(paths):
        if type(paths) == list:
            labels = []
            for path in paths:
                classes = path.split("/")[-2]
                if classes == 'depressed':
                    labels.append(0)
                elif classes == 'non-depressed':
                    labels.append(1)

            return labels
        else:
            classes = paths.split("/")[-2]
            if classes == 'depressed':
                return 0
            elif classes == 'non-depressed':
                return 1

    def __get_label_string_from_code(self, x):
        if type(x) == list:
            classes = []
            for label in x:
                if label == 0:
                    classes.append('depressed')
                elif label == 1:
                    classes.append('non-depressed')

            return classes

        else:
            if x == 0:
                return 'depressed'
            elif x == 1:
                return 'non-depressed'

    def get_label_code_from_string(self, string_values):
        if type(string_values) == list or type(string_values) == np.ndarray:
            labels = []
            for code in string_values:
                if code == 'depressed':
                    labels.append(0)
                elif code == 'non-depressed':
                    labels.append(1)

            return labels

        elif type(string_values) == str:
            if '.' not in string_values and '/' not in string_values:  # Ensuring it is not a filepath/filename
                if string_values == 'depressed':
                    return 0
                elif string_values == 'non-depressed':
                    return 1

    def get_preprocessed_aug_images(self, image_directory_path):
        # Loading Testing Set data
        datagen = ImageDataGenerator(rescale=1./255)

        aug_images = datagen.flow_from_directory(image_directory_path,
                                                 target_size=self.__INPUT_SIZE,
                                                 batch_size=self.__BATCH_SIZE,
                                                 class_mode='categorical')
        return aug_images

    def __predict_cnn(self, data, y_true=None, kind='single', limit=0, show=True, max_no_of_plots=28):
        """
        Predicts the class(es) of image(s) in 'data' using 'model' and returns a dictionary object.
        Arguments:
        ----------
        model (cnn model): model predict with
        data (str or augmented_image_obj): A file path or an augmented image object:
        kind (str):  Specifies the type of 'data' passed.
        limit(int): The number of images to predict from given list of paths (< size of data)
        show (bool): Determines whether the predictions should be visualized. (will always show if kind='single')
          Value:
            * Can be any of {'augmented' / 'single' / 'multiple'}, defaults to 'single'.
            * Set to 'single' if data is an image path,
            * set to 'multiple' if data is a list of image paths

        """

        # Initialize predictions
        predictions = None

        if kind == 'augmented':
            predictions = self.__predict_from_aug_obj(
                data, y_true, limit, show=show, max_no_of_plots=max_no_of_plots)
        elif kind == 'single':
            # predictions = self.__predict_single_cnn(data)
            predictions = self.predict_single_image_with_details(data)
        elif kind == 'multiple':
            predictions = self.__predict_multiple_cnn(
                data, y_true, limit, show=show, max_no_of_plots=max_no_of_plots)
        else:
            raise Exception('Unsupported value for \'kind\' parameter')
            return None

        # print(predictions)
        return predictions

    def __predict_from_aug_obj(self, aug_img_set_obj, y_true=None, limit=0, show=False, max_no_of_plots=28):
        # result template
        # result = {'pred_labels':[], 'True_labels':true_lbls, 'pred_values':pred_vals, 'True_values':true_vals}
        pred_lbls = []
        pred_vals = []
        true_lbls = []
        true_vals = []
        pred_hyb_probs = []

        # if y_true is available, check if it is string or encoded, and use it for evaluation appropriately
        if y_true is not None and type(y_true) == list:
            if len(y_true) == len(aug_img_set_obj.filepaths):
                if type(y_true[0]) != str:  # then it is number
                    true_vals = self.__get_label_string_from_code(y_true)
                    true_lbls = y_true
                else:  # it is string
                    true_vals = y_true
                    true_lbls = self.get_label_code_from_string(y_true)

        plot_size = max_no_of_plots if (limit > max_no_of_plots or
                                        (limit == 0 and len(aug_img_set_obj.filepaths) > max_no_of_plots)) else \
            len(aug_img_set_obj.filepaths)

        if limit <= 0 or limit > len(aug_img_set_obj.filepaths):
            file_paths = aug_img_set_obj.filepaths
        else:
            file_paths = aug_img_set_obj.filepaths[:limit]

        if show:
            rows = plot_size // 4
            if plot_size % 4 != 0:
                rows += 1

            plt.figure(num=plot_size, figsize=(20, 4 * rows))
            if len(aug_img_set_obj.filepaths) > plot_size:
                print(
                    f"\nPlotting only the first {plot_size} predictions out of {len(aug_img_set_obj.filepaths)}")
                print(
                    "----------------------------------------------------------------\n\n")

        for i, img_path in enumerate(file_paths):
            # Converting the Image to array and predict
            new_image = self.__load_image(img_path)
            predicted = self.trained_image_model.predict(
                new_image, batch_size=self.__BATCH_SIZE)

            # Showing Prediction details
            classes = ["depressed", "non-depressed"]
            pred_lbl = predicted[0].argmax()
            predicted_value = self.__get_label_string_from_code(pred_lbl)
            confidence = round(predicted[0].max() * 100, 2)
            actual_value = true_vals[i] if len(true_vals) > 0 else ''

            if (show and i < plot_size):
                img_data = plt.imread(img_path)
                plt.subplot(rows, 4, i+1)
                # ax[row, col].imshow(img_data)
                plt.imshow(img_data)
                # plt.title("Pred. :{} True :{}".format(predicted[0], aug_img_set_obj.labels[i]))
                plt.axis('off')
                plt.title(
                    f"Predicted : {pred_lbl} ({predicted_value}) ({confidence}%) \nActual : {aug_img_set_obj.labels[i]} ({actual_value})")

            # Add predicted label to predictions
            pred_hyb_probs.append(
                {'probability': predicted, 'prediction': predicted_value, 'confidence': confidence})
            pred_lbls.append(pred_lbl)
            pred_vals.append(predicted_value)

        if len(true_lbls) > 0:
            # self.show_eval_metrics(true_val=true_lbls, predicted_val=pred_lbls, title=f'Showing Plot of only {plot_size} predictions')
            print(
                f'\n==> Showing Plot of only {plot_size} of the total predictions made...\n\n')

        return pred_hyb_probs

    def __predict_multiple_cnn(self, img_paths, y_true=None, limit=0, show=False, max_no_of_plots=28):
        pred_lbls = []
        pred_vals = []
        pred_hyb_probs = []
        is_y_true_avail = False
        true_lbls = y_true

        plot_size = max_no_of_plots if (limit > max_no_of_plots or (
            limit == 0 and len(img_paths) > max_no_of_plots)) else len(img_paths)

        if y_true is not None and type(y_true) == list:
            if len(img_paths) == len(y_true):
                is_y_true_avail = True

        if limit <= 0 or limit > len(img_paths):
            file_paths = img_paths
        else:
            file_paths = img_paths[:limit]

        if show:
            rows = plot_size // 4
            if plot_size % 4 != 0:
                rows += 1
            # Initialize plot to 20 width and (4 x rows) height (4 for each pix)
            plt.figure(num=plot_size, figsize=(20, 4 * rows))
            if len(img_paths) > plot_size:
                print(
                    f"\nPlotting only the first {plot_size} predictions out of {len(img_paths)}")
                print(
                    "------------------------------------------------------------------\n\n")

        for i, path in enumerate(file_paths):
            new_image = self.__load_image(path)
            predicted = self.trained_image_model.predict(
                new_image, batch_size=self.__BATCH_SIZE)
            pred_lbl = predicted[0].argmax()
            confidence = round(predicted[0].max() * 100, 2)
            pred_val = self.__get_label_string_from_code(pred_lbl)

            pred_lbls.append(pred_lbl)
            pred_vals.append(pred_val)
            pred_hyb_probs.append(
                {'probability': predicted, 'prediction': pred_val, 'confidence': confidence})

            if (show and i < plot_size):
                # Show Image
                img_data = plt.imread(path)
                plt.subplot(rows, 4, i+1)
                plt.imshow(img_data)
                # plt.title("Pred. :{} True :{}".format(predicted[0], aug_img_set_obj.labels[i]))
                plt.axis('off')
                if is_y_true_avail == True:
                    true_lbl = y_true[i] if type(
                        y_true[i]) != str else self.get_label_code_from_string(y_true[i])
                    true_val = self.__get_label_string_from_code(
                        y_true[i]) if type(y_true[i]) != str else y_true[i]

                    plt.title(
                        f"Predicted : {pred_lbl} ({pred_val}) ({confidence}%) \nActual : {true_lbl} ({true_val})")
                else:
                    plt.title('Predicted: {} ({})\nConfidence: ({}%)'.format(
                        pred_lbl, pred_val, confidence))

        true_lbls = y_true or []
        true_vals = [x for x in list(
            map(self.__get_label_string_from_code, true_lbls))]

        if is_y_true_avail == True:
            # self.show_eval_metrics(true_val=true_lbls, predicted_val=pred_lbls, title=f'Showing Plot of only {plot_size} predictions')
            print(
                f'\n==> Showing Plot of only {plot_size} of the total predictions made...\n\n')

        return pred_hyb_probs

    def __predict_single_cnn(self, img_path):
        new_image = self.__load_image(img_path)
        predicted = self.trained_image_model.predict(
            new_image, batch_size=self.__BATCH_SIZE)
        pred_lbl = predicted[0].argmax()
        confidence = round(predicted[0].max() * 100, 2)
        pred_val = self.__get_label_string_from_code(pred_lbl)
        pred_hyb_prob = {'probability': predicted,
                         'prediction': pred_val, 'confidence': confidence}

        # Show Image
        img_data = plt.imread(img_path)
        plt.imshow(img_data)
        plt.title(f'Pred.: {pred_lbl} ({pred_val})')
        plt.axis('off')

        return pred_hyb_prob

    def predict_single_image_with_details(self, img_path):
        # if supported image format
        if img_path.split('.')[-1] in ['jgp', 'png', 'jpeg', 'bmp']:
            # Preprocessing Image
            print(img_path)
            new_image = self.__load_image(img_path, show=True)

            # Making Prediction
            predicted = self.trained_image_model.predict(
                new_image, batch_size=self.__BATCH_SIZE)

            # Showing Prediction details
            classes = ['depressed', 'non-depressed']
            pred_lbl = predicted[0].argmax()
            pred_val = self.__get_label_string_from_code(pred_lbl)
            confidence = round(predicted[0].max() * 100, 2)

            predicted = np.array([[predicted[0][1], predicted[0][0]]])
            hybrid_pred = {'probability': predicted, 'classes': ['non-depressed', 'depressed'],
                           'prediction': pred_val, 'confidence': confidence}
            # print(predicted[0].argmin())

            print()
            print(f"Prediction: \t{predicted}")
            print(f"Encoded Classes: {classes}")
            print(f"Predicted Value: {pred_val} ")
            print(f"Confidence:: {confidence}%")
            return hybrid_pred

        else:
            raise Exception(
                "Unsupported file or image format. Supported image formats are ('.jgp', '.png', 'jpeg', '.bmp')")
            return None

    def show_eval_metrics(self, true_val, predicted_val, title='Prediction Reports of the Hybrid Kinesics Depression Model', label='Score'):
        res = self.evaluate_performance(
            true_val=true_val, predicted_val=predicted_val, label=label)
        print("\n")
        print(f"\t\t\t\t{title}")
        print(f"\t\t\t\t{'*' * len(title)}\n")
        print('-'*150)
        print('\tEVALUATION: ', end='\t')
        for i, m in enumerate(res.index):
            print(f'{m}: {round(res.values[i][0] * 100, 2)}%\t', end='\t')
        print()
        print('-'*150)

    # Hybrid Specific Functions

    def predict_hybrid(self, data=None, text_col_name=None, image_col_name=None, img_path=None, post=None, mode='single', y_true=None, kind='single', limit=0, show=True, max_no_of_plots=28):
        """
          mode: {'single', 'multiple', 'dataframe'}
          text_col_name, image_col_name: should only be set if 'data' is a pandas dataframe or Series
          kind: {'single', 'multiple', 'augmented'} set the size of or nature of input data
          img_path: needed to be set to the directory containing images. NOTE: A must use for mode='augmented'
        """
        if data is not None:
            if type(data) == pd.DataFrame:  # if dataframe

                if text_col_name and image_col_name:  # if both columns for text and images are supplied together

                    # must contain equal number of values, texts must equal corresponding images
                    if len(data[text_col_name]) != len(data[image_col_name]):
                        raise Exception(
                            "Sorry, the text_posts and images_post columns must be of same length")
                        return (None, None)

                    else:  # Same length
                        img_pred = self.predict_image(
                            data=data[image_col_name], y_true=y_true, kind=kind, limit=limit, show=show, max_no_of_plots=max_no_of_plots)
                        text_pred = self.predict_text_post(
                            text_post=data[text_col_name])

                        if y_true:  # Show evaluation metrics, if y_true is available
                            if len(y_true) == len(img_pred) and len(y_true) == len(text_pred):
                                y_pred = self.get_best_prediction(
                                    text_pred, img_pred)
                                y_pred = self.get_label_code_from_string(
                                    y_pred)

                                self.show_eval_metrics(
                                    y_true, y_pred)  # Display Report

                        return (text_pred, img_pred)

                elif image_col_name and not text_col_name:
                    img_pred = self.predict_image(
                        data=data[image_col_name], y_true=y_true, kind=kind, limit=limit, show=show, max_no_of_plots=max_no_of_plots)

                    if y_true:  # Show evaluation metrics, if y_true is available
                        if len(y_true) == len(img_pred):
                            y_pred = self.get_prediction_image_only(img_pred)
                            y_pred = self.get_label_code_from_string(y_pred)

                            self.show_eval_metrics(
                                y_true, y_pred)  # Display Report
                    return (None, img_pred)

                elif text_col_name and not image_col_name:
                    text_pred = self.predict_text_post(
                        text_post=data[text_col_name])

                    if y_true:  # Show evaluation metrics, if y_true is available
                        if len(y_true) == len(text_pred):
                            y_pred = self.get_prediction_text_only(text_pred)
                            y_pred = self.get_label_code_from_string(y_pred)

                            self.show_eval_metrics(
                                y_true, y_pred)  # Display Report
                    return (text_pred, None)

                else:
                    raise Exception(
                        "'text_col_name' and/or 'image_col_name' parameters must be set for a DataFrame object data'")
                    return (None, None)

            elif type(data) == list:  # if list
                # if list of dictionaries of the form [{'tag':'text1', 'image':'img_url1'}, {'tag':'text2', 'image':'img_url2'},...]
                if type(data[0]) == dict:
                    keys = data[0].keys()
                    if 'tag' in keys and 'image' in keys:
                        temp = pd.DataFrame(data)
                        img_pred = self.predict_image(
                            data=temp['image'], y_true=y_true, kind=kind, limit=limit, show=show, max_no_of_plots=max_no_of_plots)
                        text_pred = self.predict_text_post(
                            text_post=temp['tag'])

                        if y_true:  # Show evaluation metrics, if y_true is available
                            if len(y_true) == len(img_pred) and len(y_true) == len(text_pred):
                                y_pred = self.get_best_prediction(
                                    text_pred, img_pred)
                                y_pred = self.get_label_code_from_string(
                                    y_pred)

                                self.show_eval_metrics(
                                    y_true, y_pred)  # Display Report
                        return (text_pred, img_pred)

                    elif 'text' in keys and 'image' in keys:
                        temp = pd.DataFrame(data)
                        img_pred = self.predict_image(
                            temp['image'], y_true, kind, limit, show, max_no_of_plots)
                        text_pred = self.predict_text_post(
                            text_post=temp['text'])

                        if y_true:  # Show evaluation metrics, if y_true is available
                            if len(y_true) == len(img_pred) and len(y_true) == len(text_pred):
                                y_pred = self.get_best_prediction(
                                    text_pred, img_pred)
                                y_pred = self.get_label_code_from_string(
                                    y_pred)

                                self.show_eval_metrics(
                                    y_true, y_pred)  # Display Report
                        return (text_pred, img_pred)

                    elif 'tag' in keys and 'image' not in keys:  # tag present, no image
                        temp = pd.DataFrame(data)
                        text_pred = self.predict_text_post(
                            text_post=temp['tag'])

                        if y_true:  # Show evaluation metrics, if y_true is available
                            if len(y_true) == len(text_pred):
                                y_pred = self.get_prediction_text_only(
                                    text_pred)
                                y_pred = self.get_label_code_from_string(
                                    y_pred)

                                self.show_eval_metrics(
                                    y_true, y_pred)  # Display Report
                        return (text_pred, None)

                    elif 'text' in keys and 'image' not in keys:
                        temp = pd.DataFrame(data)
                        text_pred = self.predict_text_post(
                            text_post=temp['text'])
                        return (text_pred, None)

                    elif 'tag' not in keys and 'text' not in keys and 'image' in keys:
                        temp = pd.DataFrame(data)
                        img_pred = self.predict_image(
                            temp['image'], y_true, kind, limit, show, max_no_of_plots)

                        if y_true:  # Show evaluation metrics, if y_true is available
                            if len(y_true) == len(img_pred):
                                y_pred = self.get_prediction_image_only(
                                    img_pred)
                                y_pred = self.get_label_code_from_string(
                                    y_pred)

                                self.show_eval_metrics(
                                    y_true, y_pred)  # Display Report
                        return (None, img_pred)

                    else:
                        raise Exception(
                            "Invalid data format supplied. Dictionary data must have \n 'tag' or 'text' keys holding post-text and/or 'image' key holding post image-url")
                        return None, None

                # if list of lists of the form [['text1', 'text2', ...], ['img_url1', 'img_url2', ...]]
                elif type(data[0]) == list:
                    if len(data) >= 2:
                        if len(data[0]) == len(data[1]):
                            texts, imgs = data[0], data[1]
                            img_pred = self.predict_image(
                                imgs, y_true, kind, limit, show, max_no_of_plots)
                            text_pred = self.predict_text_post(text_post=texts)

                            if y_true:  # Show evaluation metrics, if y_true is available
                                if len(y_true) == len(img_pred) and len(y_true) == len(text_pred):
                                    y_pred = self.get_best_prediction(
                                        text_pred, img_pred)
                                    y_pred = self.get_label_code_from_string(
                                        y_pred)

                                    self.show_eval_metrics(
                                        y_true, y_pred)  # Display Report
                            return (text_pred, img_pred)

                        else:
                            raise Exception(
                                'Error: both data for post texts and images must be of the same length')
                            return None, None
                    else:
                        raise Exception(
                            'Confusion: Cannot infer text data or image data from the data supplied in the list')
                        return None, None

        else:
            if post is not None and img_path is not None:
                text_pred = self.predict_text_post(text_post=post)
                img_pred = self.predict_single_image_with_details(
                    img_path=img_path)

                return text_pred, img_pred

    # Hybrid Model Public Methods

    def predict_image(self, data, y_true=None, kind='single', limit=0, show=True, max_no_of_plots=28):
        return self.__predict_cnn(data, y_true=y_true, kind=kind, limit=limit, show=show, max_no_of_plots=max_no_of_plots)

    def predict_text_post(self, text_post):
        return self.__predict_text_post(textPost=text_post)

    def get_best_prediction(self, text_pred, img_pred):
        best_res = []
        if len(text_pred) == len(img_pred):
            for i in range(len(text_pred)):
                if text_pred[i]['prediction'] == img_pred[i]['prediction']:
                    best_res.append(text_pred[i]['prediction'])
                elif text_pred[i]['confidence'] > img_pred[i]['confidence']:
                    best_res.append(text_pred[i]['prediction'])
                else:
                    best_res.append(img_pred[i]['prediction'])
        else:
            raise Exception('size of both dataset must be the same')
            return None

        best_res['classes'] = ['non-depressed', 'depressed']
        return best_res

    def get_prediction_image_only(self, img_pred):
        best_res = []
        for i in range(len(img_pred)):
            best_res.append(img_pred[i]['prediction'])

        return best_res

    def get_prediction_text_only(self, text_pred):
        best_res = []
        for i in range(len(text_pred)):
            best_res.append(text_pred[i]['prediction'])

        return best_res

    def get_best_prediction_with_details(self, text_pred, img_pred):
        best_res = []
        if len(text_pred) == len(img_pred):
            for i in range(len(text_pred)):
                if text_pred[i]['prediction'] == img_pred[i]['prediction']:
                    if text_pred[i]['confidence'] > img_pred[i]['confidence']:
                        best_res.append(text_pred[i])
                    else:
                        t = img_pred[i]['probability'][0]
                        img_pred[i]['probability'] = np.array(
                            [t[1], t[0]], dtype=np.float32)

                        best_res.append(img_pred[i])

                elif text_pred[i]['confidence'] > img_pred[i]['confidence']:
                    best_res.append(text_pred[i])
                else:
                    t = img_pred[i]['probability'][0]
                    img_pred[i]['probability'] = np.array(
                        [t[1], t[0]], dtype=np.float32)

                    best_res.append(img_pred[i])
        else:
            raise Exception('size of both dataset must be the same')
            return None

        return best_res

    def get_avg_prediction_with_details(self, text_pred, img_pred):
        avg_pred = []
        if type(text_pred) == list and type(img_pred) == list:
            if len(text_pred) == len(img_pred):
                for i in range(len(text_pred)):
                    if text_pred[i]['prediction'] == img_pred[i]['prediction']:
                        if text_pred[i]['confidence'] > img_pred[i]['confidence']:
                            avg_pred.append(text_pred[i])
                        else:
                            t = img_pred[i]['probability'][0]
                            img_pred[i]['probability'] = np.array(
                                [t[1], t[0]], dtype=np.float32)

                            avg_pred.append(img_pred[i])

                    elif text_pred[i]['confidence'] > img_pred[i]['confidence']:
                        avg_pred.append(text_pred[i])
                    else:
                        t = img_pred[i]['probability'][0]
                        img_pred[i]['probability'] = np.array(
                            [t[1], t[0]], dtype=np.float32)

                        avg_pred.append(img_pred[i])
            else:
                raise Exception('size of both dataset must be the same')
                return None
        elif type(text_pred) == dict and type(img_pred) == dict:
            avg_pred = {}
            txt = text_pred
            img = img_pred

            classes = ['non-depressed', 'depressed']

            txt['probability'] = txt['probability'] if len(
                txt['probability']) > 1 else txt['probability'][0]
            img['probability'] = img['probability'] if len(
                img['probability']) > 1 else img['probability'][0]

            avg_prob = (txt['probability'] + img['probability']) / 2
            pred = classes[np.argmax(avg_prob)]
            conf = np.round(avg_prob.max() * 100, 2)

            avg_pred['probability'] = avg_prob
            avg_pred['classes'] = classes
            avg_pred['prediction'] = pred
            avg_pred['confidence'] = conf

            return avg_pred

        return avg_pred

    def get_best_prediction_with_confidence_as_df(self, text_pred, img_pred):
        best_res = {'prediction': [], 'confidence': []}
        if len(text_pred) == len(img_pred):
            for i in range(len(text_pred)):
                if text_pred[i]['prediction'] == img_pred[i]['prediction']:
                    best_res['prediction'].append(text_pred[i]['prediction'])
                    best_res['confidence'].append(
                        max([text_pred[i]['confidence'], img_pred[i]['confidence']]))

                elif text_pred[i]['confidence'] > img_pred[i]['confidence']:
                    best_res['prediction'].append(text_pred[i]['prediction'])
                    best_res['confidence'].append(text_pred[i]['confidence'])

                else:
                    best_res['prediction'].append(img_pred[i]['prediction'])
                    best_res['confidence'].append(img_pred[i]['confidence'])

        else:
            raise Exception('size of both dataset must be the same')
            return None

        return pd.DataFrame(best_res)

    def visualize_predictions_in_bars(self, text_pred, img_pred, n_cols=4, n_predictions_to_plot=40):
        # try:
        # Plotting first five predictions
        y_pred_viz_data = self.get_best_prediction_with_details(
            text_pred, img_pred)
        y_pred_viz_data_prob = [x['probability'] for x in y_pred_viz_data]
        outcome = [x['prediction'] for x in y_pred_viz_data]

        data = y_pred_viz_data_prob[:n_predictions_to_plot]
        cols = n_cols
        rows = int(len(data) // cols)
        rows = (rows + 1) if len(data) % cols != 0 else rows

        plt.figure(figsize=(20, 17))
        for i in range(len(data)):
            ax = plt.subplot(rows, cols, i+1)
            sns.barplot(x=data[i], y=["non-depressed",
                        'depressed'], palette='deep', ax=ax)
            plt.title(f'[Predicion {i+1}: {outcome[i]}]')
            plt.tight_layout()

        # except Exception as err:
        #   print(err)
