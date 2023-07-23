from pathlib import Path
import torchxrayvision.datasets as xrv_data
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import torch, random

def return_root_path():
    """ Return the root path for data saving given the current OS """
    if Path.cwd().as_posix().startswith('/'):
        return Path('/nfs/turbo/med-kayvan-lab/')
    return Path('Z:/')


def apply_transforms(sample, transform, seed=None, transform_seg=False):
    """
    Customized function from apply_transforms in xrv_data.py
    
    Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or 
    other tasks which use masks information.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

        if "pathology_masks" in sample:
            for i in sample["pathology_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])

        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])
        
        if "seg" in sample and transform_seg:
            random.seed(seed)
            torch.random.manual_seed(seed)
            sample["seg"] = transform(sample["seg"])

    return sample

class ARDSchestXrayDataset(xrv_data.Dataset):
    """
    Dataset for ARDS CXR Images. It's an updated version of the original, which
    can return the uncertainty labels of all instances. Be sure to notice the 
    changes on the default values of the parameters.
    Inputs:
        csvpath <Pathlib object>: path to the csv file containing the labels.
        uncertainty_file <Pathlib object>: path to the csv file containing 
            the uncertainty labels.
        uncertainty_measure_function <Python function>: 
            function to apply to the uncertainty labels.
        uncertainty_encoding <Python str>: how to encode the uncertainty labels.
            could be one of ['onehot', 'combined', 'separate'].
        uncertainty_threshold <Python float>: threshold to apply to the
            output of uncertainty_measure_function. Use 0 to disable threshold.
                The encoding would be substituted to all zeros when the case has 
                a lower uncertainty than the threshold.
        transforms <PyTorch transform>: transforms to apply to the images.
        xrv_normalize <Python int>: 
            1: xrv normalize function, normalize to [-1024, 1024].
            0: normalize to [0, 1].
            2: normalize to [0, 255].
        confusion_encode <Python str>: how to encode the confusion labels.
            could be one of ['binary', 'grade'].
        seed <Python int>: seed for the random number generator.
    """
    def __init__(self,
                 csvpath,
                 uncertainty_file,
                 uncertainty_measure_function,
                 uncertainty_encoding='onehot',
                 uncertainty_threshold=0,
                 transforms=None,
                 xrv_normalize=1,
                 confusion_encode='binary',
                 return_segmentation=False,
                 seed=0):

        super(ARDSchestXrayDataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.pathologies = ["nonARDS", "ARDS"]
        self.confusion_encode = confusion_encode
        self.return_segmentation = return_segmentation
        
        self.transforms = transforms
        self.xrv_normalize = xrv_normalize
        # Load csv data
        self.csv = pd.read_csv(csvpath)
        self.csv["PatientID"] = self.csv["PatientID"].astype(str)
        # load the uncertainty data
        self.uncertainty_csv = pd.read_csv(uncertainty_file)

        # retrived the mean uncertainty for each patient and use that as label
        labels = []
        # also record the grade
        grades = []
        # retrived the uncertainty level defined by the uncertainty function
        certainty_level = []
        for id in self.csv["PatientName"]:
            image_rows = self.uncertainty_csv[
                self.uncertainty_csv["image_id"] == id]
            grades.append(round(image_rows["cxr_scale"].mean()))
            labels.append(int(image_rows["cxr_scale"].mean() > 4.5))
            certainty_level.append(uncertainty_measure_function(
                    image_rows['cxr_scale'].values))
            
        # convert the uncertain level and grades to numpy array
        self.certainty_level = np.array(certainty_level)
        self.raw_grades = np.array(grades)
        self.raw_labels = np.array(labels)

        self.label_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.label_encoder.fit(np.array(range(len(self.pathologies)))[:, np.newaxis])
        labels_encoded = self.label_encoder.transform(
            np.array(labels)[:, np.newaxis])
        # convert the labels to encoded numpy array
        self.labels = labels_encoded.astype(np.float32)

        # Handling the uncertainty labels
        self.uncertainty_encoding = uncertainty_encoding
        self.uncertainty_threshold = uncertainty_threshold
        # construct and fit the reviewer and uncertainty encoders
        self.reviewer_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.reviewer_enc.fit(self.uncertainty_csv[
            'reviewer_id'].values[:, np.newaxis])
        self.uncertainty_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.uncertainty_enc.fit(self.uncertainty_csv[
            'cxr_scale'].values[:, np.newaxis])
        
    @property
    def csv_file(self):
        return self.csv
    
    @property
    def certainty(self):
        return self.certainty_level
    
    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)
    
    def get_additional_dim(self):
        sample = self[0]
        if "add" in sample:
            return sample["add"].shape[0]
        return 0
    
    def get_num_reviewer(self):
        return len(self.reviewer_enc.categories_[0])
    
    def get_num_label(self):
        if self.confusion_encode == 'binary':
            return len(self.label_encoder.categories_[0])
        else:
            return len(self.uncertainty_enc.categories_[0])
    
    def get_reviewer_label_matrix(self, image_rows):
        """ Return the one-hot representation of labels from multiple annotators. """
        onehot_encode_id = self.reviewer_enc.transform(image_rows[
                'reviewer_id'].values[:, np.newaxis])
        # find the idx of the reviewer's id
        non_zero_idx = np.nonzero(onehot_encode_id)[1]
        reviewer_label = np.zeros((len(self.reviewer_enc.categories_[0]), 
                               self.get_num_label()))
        if self.confusion_encode == 'binary':
            encoded_label = self.label_encoder.transform(
                (image_rows['cxr_scale'].values[:, np.newaxis] > 4.5).astype(int))
        else:
            encoded_label = self.uncertainty_enc.transform(
                image_rows['cxr_scale'].values[:, np.newaxis])

        for i, pos in enumerate(non_zero_idx):
            reviewer_label[pos, :] = encoded_label[i,:]

        return reviewer_label

    def __getitem__(self, idx):
        sample = {}

        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        sample["raw_grades"] = self.raw_grades[idx]

        image_id = self.csv['PatientName'].iloc[idx]
        image_rows = self.uncertainty_csv[
            self.uncertainty_csv['image_id'] == image_id]
        sample["image_id"] = image_id

        # construct the additional information vector
        if self.uncertainty_encoding == 'combined':
            sample["add"] = (self.reviewer_enc.transform(image_rows[
                'reviewer_id'].values[:, np.newaxis])*image_rows[
                'cxr_scale'].values[:, np.newaxis]).sum(axis=0).astype(np.float32)
        elif self.uncertainty_encoding == 'onehot':
            sample["add"] = self.uncertainty_enc.transform(
                    image_rows['cxr_scale'].values[
                    :, np.newaxis]).sum(axis=0).astype(np.float32)
        elif self.uncertainty_encoding == 'separate':
            sample["add"] = np.concatenate((
                self.reviewer_enc.transform(image_rows['reviewer_id'].values[
                    :, np.newaxis]).sum(axis=0),
                self.uncertainty_enc.transform(image_rows['cxr_scale'].values[
                    :, np.newaxis]).sum(axis=0)), axis=0).astype(np.float32)
        else:
            raise ValueError("Unknown uncertainty encoding method.")
       
        sample["uncertainty"] = self.certainty_level[idx]
        if self.certainty_level[idx] < self.uncertainty_threshold:
            # substitute the addtional information to array of zeros
            # when the uncertainty is lower than the threshold
            sample["add"] = np.zeros((sample["add"].shape[0])).astype(np.float32)
        
        sample['reviewer_label'] = self.get_reviewer_label_matrix(image_rows)

        path = Path(self.csv['ImageFile'].iloc[idx])
        img_path = return_root_path() / path.relative_to(
            '/nfs/turbo/med-kayvan-lab/')
        img = imread(img_path.as_posix())
        sample['img_path'] = img_path.as_posix()
        
        if self.xrv_normalize == 1:
            img = xrv_data.normalize(img, maxval=255, reshape=True)
        elif self.xrv_normalize == 2:
            img = ((img - img.min()) / (img.max() - img.min())) * 255.
            img = img.astype('uint8')
        else:
            img = img.astype(np.float32) / 255.
        sample["img"] = img

        sample["seg"] = -1

        if self.return_segmentation:
            seg = imread(img_path.as_posix().replace('.', '_lung_seg.'))
            # make the segmentation binary
            seg[seg > 0] = 1
            sample["seg"] = seg.astype(np.float32)

        if self.transforms is not None:
            sample = apply_transforms(sample, self.transforms, self.return_segmentation)

        return sample


class CheXpertDataset(xrv_data.Dataset):
    """
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
              https://arxiv.org/abs/1901.07031

    The train_uncertainty.csv file contains labels obtained by running the CheXbert labeler 
        on reports in the CheXpert dataset. These labels include positive (1), negative (0), 
        uncertain (-1), and unmentioned (blank) classes.
    
    Input:
        csvpath <python str>: path to the csv file
        views <python list>: list of views to use
        seed <python int>: seed for randomization
        unique_patients <python bool>: if True, only use one image per patient
        encoding <python str>: the encoding to use for the labels.
            One of ['onehot', 'label']
        transforms <torchvision.transforms>: transforms to apply to the image
        xrv_normalize <python bool>: if True, normalize the image using the xrv normalize function
    """

    def __init__(self,
                 csvpath,
                 views=["PA", "AP"],
                 seed=0,
                 unique_patients=False,
                 encoding='onehot',
                 transforms=None,
                 xrv_normalize=0
                 ):

        super(CheXpertDataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.transforms = transforms
        self.xrv_normalize = xrv_normalize

        self.csvpath = csvpath
        # Read in the csv, fill the empty value with 2 in read_csv
        self.csv = pd.read_csv(self.csvpath)
        self.csv.loc[:, 
            "Enlarged Cardiomediastinum":"No Finding"] = self.csv.loc[:, 
            "Enlarged Cardiomediastinum":"No Finding"].fillna(2)

        # Rename to have a csv that is consistent with the views column 
        self.views = views
        self.csv["view"] = self.csv["Frontal/Lateral"] 
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"]  
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"}) 
        self.limit_to_selected_views(views)

        # Settings on unqiue patients
        self.csv["PatientID"] = self.csv["Path"].str.extract(pat=r'patient(\d+)')
        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()
        # Refine demographic information
        self.csv['Age'][(self.csv['Age'] == 0)] = np.mean(self.csv['Age'])
        self.csv['Sex'][(self.csv['Sex'] == 'Male')] = 0
        self.csv['Sex'][(self.csv['Sex'] == 'Female')] = 0

        # reset the index of the csv
        self.csv = self.csv.reset_index(drop=True)
        # Get the class label encoder
        self.to_encode = self.csv.loc[:, "Enlarged Cardiomediastinum":"No Finding"]
        if encoding == 'onehot':
            self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.enc.fit(self.to_encode)
        else:
            self.enc = LabelEncoder()
            self.enc.fit([-1, 0, 1, 2])
        
        # set a dummy label attribute to conform to the xrv_data.Dataset class
        self.labels = np.zeros((len(self), 1))
        self.pathologies = ["None"]

    @property
    def csv_file(self):
        return self.csv

    def string(self):
        return "Contains num_samples={} views={}".format(len(self), self.views)

    def get_additional_dim(self):
        sample = self[0]
        if "lab" in sample:
            return sample["lab"].shape[0]
        return 0
    
    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        
        img_path = Path(self.csvpath).parent / self.csv['Path'].iloc[idx]

        if self.xrv_normalize == 1:
            # single channel output renormliazed to [-1024, 1024]
            sample["img"] = xrv_data.normalize(imread(str(img_path)), maxval=255, reshape=True)
        elif self.xrv_normalize == 0:
            # single channel output renormliazed to [0, 1]
            sample["img"] = imread(str(img_path), as_gray=True).astype(np.float32) / 255.
        elif self.xrv_normalize == 2:
            # single channel output renormliazed to [0, 255]
            img = imread(str(img_path), as_gray=True)
            img = ((img - img.min()) / (img.max() - img.min())) * 255.
            img = img.astype('uint8')
            sample["img"] = img

        if self.transforms is not None:
            sample = apply_transforms(sample, self.transforms)

        return sample

