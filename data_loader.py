from pathlib import Path
import numpy as np
import torch.utils.data as module_data
import torch.utils.data.dataloader as module_default_dataloader
import torch.utils.data.sampler as module_sampler
from torchvision import transforms
import dataset as module_dataset
import torchxrayvision.datasets as module_xrv_data
import torch

############################################################################################
########################### Dataloader for the ARDSchestXrayDataset ########################
############################################################################################

class DualDataLoader(module_data.DataLoader):
    """
    Dual data loader
    Input:
        train_dataset <torch.utils.data.Dataset>: the training dataset
        val_dataset <torch.utils.data.Dataset>: the validation dataset
        batch_size <python int>: the batch size
        num_workers <python int>: the number of workers for data loader
        shuffle <python bool>: whether to shuffle the data
        collate_fn <python function>: the collate function for the data loader
    """
    def __init__(self, train_dataset, val_dataset, 
                 batch_size, val_indices=None, num_workers=0, shuffle=True, 
                 collate_fn=module_default_dataloader.default_collate):
        self.val_indices = val_indices
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
        }
        self.val_dataset = val_dataset
        
        super().__init__(dataset=train_dataset, **self.init_kwargs)

    def get_val_loader(self):
        sampler = None
        if self.val_indices is not None:
            self.init_kwargs['shuffle'] = False
            sampler = module_sampler.SubsetRandomSampler(self.val_indices)
        return module_data.DataLoader(dataset=self.val_dataset, sampler=sampler, **self.init_kwargs)
    
    def get_test_loader(self, test_dataset, test_indices=None):
        sampler = None
        if test_indices is not None:
            self.init_kwargs['shuffle'] = False
            sampler = module_sampler.SubsetRandomSampler(test_indices)
        return module_data.DataLoader(dataset=test_dataset, sampler=sampler, **self.init_kwargs)


def uncertainty_func(x_array):
    total = 0
    for x in x_array:
        if x < 4.5:
            total += x - 1
        else:
            total += -x + 8
    return total / x_array.shape[0]

def decison_function(x_array):
    return uncertainty_func(x_array) + x_array.std()

class ARDSDataLoader(DualDataLoader):
    """
    ARDS data loader for the Chest X-ray images
    Load both the training and validation dataset with two csv files.
    Input:
        csv_dir <python string>: the directory of the csv files
        batch_size <python int>: the batch size
        image_size <python int>: the image size
        uncertainty_csv <python string>: the csv file containing the uncertainty infor  
        uncertainty_encoding <python string>: the encoding of the uncertainty level, 
            can be 'onehot', 'combined' or 'separate'
        use_uncertainty_func <python bool>: whether to use a decison_function to decide
            if the uncertainty encoding should be used
        uncertainty_threshold <python int>: the threshold for the decsion function
            if the uncertainty is larger than the threshold, the uncertainty encoding
            will be used. Otherwise, an all zero encoding will be returned.
        xrv_normalize <python int>: define which normalization method to use
            1: xrv normalize function, normalize to [-1024, 1024], 
               returned as PIL image format in grayscale.
            0: self-defined, normalize to [0, 1].
            2: self-defined, normalize to [0, 255].
        clean_data_only <python bool>: whether use clean validation data, which means
            only those whoes uncertainty is smaller than the certain threshold will be used.
    """
    def __init__(self, csv_dir, batch_size, image_size, 
                 uncertainty_csv="/nfs/turbo/med-kayvan-lab/Projects/ARDS/Data/Processed/ARDS/CXR_Representation_Learning/Infor/CXRReview.csv",
                 uncertainty_encoding='onehot', use_uncertainty_func=False,
                 uncertainty_threshold=2, xrv_normalize=True, 
                 clean_data_only=False, clean_level=2, segmentation=False):
        
        if xrv_normalize == 1:
            trsfm = transforms.Compose([
                module_xrv_data.XRayCenterCrop(),
                module_xrv_data.XRayResizer(image_size)
            ])
        elif xrv_normalize == 2:
            trsfm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Turn grayscale to RGB
                ])
        elif xrv_normalize == 0:
            trsfm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Turn grayscale to RGB
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError("xrv_normalize can only be 0, 1 or 2")

        self.kwag_dict = {"transforms":trsfm,
                    "xrv_normalize":xrv_normalize,
                    "uncertainty_encoding":uncertainty_encoding,
                    "return_segmentation":segmentation}
        
        if use_uncertainty_func:
            self.kwag_dict["uncertainty_threshold"] = uncertainty_threshold
        
        self.csv_dir = csv_dir
        self.uncertainty_csv = uncertainty_csv
        train_dataset = module_dataset.ARDSchestXrayDataset(
            Path(self.csv_dir) / 'train.csv', uncertainty_csv, decison_function, **self.kwag_dict)
        val_dataset = module_dataset.ARDSchestXrayDataset(
            Path(self.csv_dir) / 'valid.csv', uncertainty_csv, decison_function, **self.kwag_dict)
        
        self.clean_level = clean_level
        val_indices = None
        if clean_data_only:
            cetainties = val_dataset.certainty
            # find the indices when the value of cetainties is bigger than cetainty_level
            val_indices = np.where(cetainties > self.clean_level)[0]

        super().__init__(train_dataset, val_dataset, batch_size, val_indices=val_indices)
    
    @property
    def train_dataset(self):
        return self.dataset
    
    @property
    def validation_dataset(self):
        return self.val_dataset
    
    def get_additional_dim(self):
        return self.dataset.get_additional_dim()
    
    def get_test_loader(self, datatype='all', clean_level=None):

        test_dataset = module_dataset.ARDSchestXrayDataset(
            Path(self.csv_dir) / 'test.csv', self.uncertainty_csv, decison_function, **self.kwag_dict)
        clean_level = self.clean_level if clean_level is None else clean_level
        if datatype == 'all':
            test_indices = list(range(len(test_dataset)))
        elif datatype == 'clean':
            cetainties = test_dataset.certainty
            # find the indices when the value of cetainties is smaller than cetainty_level
            test_indices = np.where(cetainties <= clean_level)[0]
        elif datatype == 'noisy':
            cetainties = test_dataset.certainty
            # find the indices when the value of cetainties is bigger than cetainty_level
            test_indices = np.where(cetainties > clean_level)[0]
        else:
            raise ValueError(f"datatype {datatype} is not supported")
            
        return super().get_test_loader(test_dataset, test_indices=test_indices)


class KFoldsDataLodaer():
    """
    Loading data from a dataset with k-fold cross validation. The split is defined to 
    be patient-wise. The dataloader can be set up in a way that only the 'less uncertain' 
    instances are loaded in validation.
    Input:
        dataset <torch.utils.data.Dataset>: the dataset
        batch_size <python int>: the batch size
        total_folds <python int>: the total number of folds
        split_infor <python str>: the information for the split, in the format of
            'csv;id_column_name'. For example, 'csv;PatientID;' means the split is performed on
            the csv attribute of the dataset and is patient-wise for the column name 'PatientID'.
        data_certainties <python list or None>: the uncertainty level correpsonding to each
            instance, which is used to return a low uncertainty dataset in the validation.
        clean_level <python int>: the threshold for the uncertainty level 
            as a metric to determine whether the instance is low uncertainty or not.
    """
    def __init__(self, dataset, batch_size, total_folds=5, 
                 split_infor='csv;PatientID', 
                 data_certainties=None, clean_level=2,
                 weighted_training=False,
                 num_workers=0, shuffle=False, 
                 collate_fn=module_default_dataloader.default_collate):

        self.dataset = dataset
        self.shuffle = shuffle
        self.weighted_training = weighted_training
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        
        if weighted_training:
            self.init_kwargs['drop_last'] = True
        
        self.split_dict = self._split_sampler_kfolds(split_infor, total_folds, 
                                data_certainties=data_certainties, 
                                clean_level=clean_level)

    def get_data_loaders(self, fold=0):
        train_idx, valid_idx = self.split_dict[fold]
        if self.weighted_training:
            # obtain the raw labels
            labels = self.dataset.raw_labels
            # sort the unique labels from small to large
            unique_labels = np.sort(np.unique(labels))
            # obtain the weights
            weights = 1. / torch.tensor([len(np.where(labels == t)[0]) for t in unique_labels])
            # put more weights on the 
            # weights[-1] = weights[-1] * 3
            # obtain the weights for each sample
            samples_weights = torch.tensor([weights[t] for t in labels])
            # create a sampler with the weights
            self._train_sampler = module_sampler.WeightedRandomSampler(samples_weights, len(samples_weights))
        else:
            self._train_sampler = module_sampler.SubsetRandomSampler(train_idx)

        self._valid_sampler = module_sampler.SubsetRandomSampler(valid_idx)

        train_loader = module_data.DataLoader(dataset=self.dataset, 
                                              sampler=self._train_sampler, **self.init_kwargs)
        self.init_kwargs['drop_last'] = False
        val_loader = module_data.DataLoader(dataset=self.dataset,
                                            sampler=self._valid_sampler, **self.init_kwargs)

        return train_loader, val_loader
    
    def get_test_loader(self, test_dataset, test_indices=None):
        sampler = None
        if test_indices is not None:
            self.init_kwargs['shuffle'] = False
            sampler = module_sampler.SubsetRandomSampler(test_indices)
        self.init_kwargs['drop_last'] = False
        return module_data.DataLoader(dataset=test_dataset, sampler=sampler, **self.init_kwargs)
       
    def _split_sampler_kfolds(self, split_infor, total_folds, 
                              data_certainties=None, clean_level=2):
        """
        Split the dataset into k folds. Only return the "certain" instances in the validation
        if need be.

        Input:
            split_infor <python str>: the information for the split, in the format of
                'csv;id_column_name;stratified_name'. For example, 'csv;PatientID' means 
                the split is patient-wise based on 'PatientID'.
            total_folds <python int>: the total number of folds
            data_certainties <python list or None>: the uncertainty level correpsonding to each
                instance, which is used to return a low uncertainty dataset in the validation.
            clean_level <python int>: the threshold for the uncertainty level 
                as a metric to determine whether the instance is low uncertainty or not.
        """
        csv_attribute, id_column_name = split_infor.split(';')
        if not hasattr(self.dataset, csv_attribute):
            raise ValueError("The dataset class does"+ \
                             " not have the attribute {}".format(csv_attribute))
        else:
            patient_df = getattr(self.dataset, csv_attribute)

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=total_folds, random_state=42, shuffle=True)
        patients = patient_df[id_column_name].unique()
        split_dict = {}
        for i, (train_index, test_index) in enumerate(kf.split(patients)):
            train_patients = patients[train_index]
            valid_patients = patients[test_index]
            train_idx = patient_df[patient_df[id_column_name].isin(train_patients)].index
            valid_idx = patient_df[patient_df[id_column_name].isin(valid_patients)].index
            split_dict[i] = (train_idx, valid_idx)
        
        if data_certainties is not None:
            # return only the low uncertainty instances in the validation idxes
            for i in range(total_folds):
                train_idx, valid_idx = split_dict[i]
                valid_idx = valid_idx[data_certainties[valid_idx] <= clean_level]
                split_dict[i] = (train_idx, valid_idx)

        return split_dict
    

class ARDSKFoldDataLoader(KFoldsDataLodaer):
    """
    ARDS data loader for the Chest X-ray images, loading datas from the ARDS dataset with k-fold cross validation.
    
    Input:
        csv_dir <python string>: the directory of the csv files
        batch_size <python int>: the batch size
        image_size <python int>: the image size
        uncertainty_csv <python string>: the csv file containing the uncertainty infor  
        uncertainty_encoding <python string>: the encoding of the uncertainty level, 
            can be 'onehot', 'combined' or 'separate'
        use_uncertainty_func <python bool>: whether to use a decison_function to decide
            if the uncertainty encoding should be used
        uncertainty_threshold <python int>: the threshold for the decsion function
            if the uncertainty is larger than the threshold, the uncertainty encoding
            will be used. Otherwise, an all zero encoding will be returned.
        xrv_normalize <python int>: define which normalization method to use
            1: xrv normalize function, normalize to [-1024, 1024], 
               returned as PIL image format in grayscale.
            0: self-defined, normalize to [0, 1].
            2: self-defined, normalize to [0, 255].
    """
    def __init__(self, csv_dir, batch_size, image_size, 
                 uncertainty_csv="/nfs/turbo/med-kayvan-lab/Projects/ARDS/Data/Processed/ARDS/CXR_Representation_Learning/Infor/CXRReview.csv",
                 uncertainty_encoding='onehot', use_uncertainty_func=False,
                 uncertainty_threshold=2, xrv_normalize=True, 
                 clean_data_only=False, clean_level=2, folds=5, weighted_training=False,
                 segmentation=False):
        
        if xrv_normalize == 1:
            trsfm = transforms.Compose([
                module_xrv_data.XRayCenterCrop(),
                module_xrv_data.XRayResizer(image_size)
            ])
        elif xrv_normalize == 2:
            trsfm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Turn grayscale to RGB
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        elif xrv_normalize == 0:
            trsfm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Turn grayscale to RGB
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError("xrv_normalize can only be 0, 1 or 2")
            
        self.kwag_dict = {"transforms":trsfm,
                    "xrv_normalize":xrv_normalize,
                    "uncertainty_encoding":uncertainty_encoding,
                    "return_segmentation":segmentation}
        
        if use_uncertainty_func:
            # the use_uncertainty_func is the controller attribute that determines
            # if the uncertainty encoding should be returned in the dataset
            self.kwag_dict["uncertainty_threshold"] = uncertainty_threshold
        
        self.csv_dir = csv_dir
        self.uncertainty_csv = uncertainty_csv
        all_train_dataset = module_dataset.ARDSchestXrayDataset(
            Path(self.csv_dir) / 'train_data.csv', uncertainty_csv, 
            decison_function, **self.kwag_dict)
        
        self.clean_level = clean_level
        cetainties = all_train_dataset.certainty if clean_data_only else None
        # data_certainties is the controller attribute for cleaning the validation set
        super().__init__(all_train_dataset, batch_size, total_folds=folds, 
                         data_certainties=cetainties, 
                         clean_level=clean_level, weighted_training=weighted_training)

    def get_additional_dim(self):
        return self.dataset.get_additional_dim()
    
    def get_test_loader(self, datatype='all', clean_level=None):

        test_dataset = module_dataset.ARDSchestXrayDataset(
            Path(self.csv_dir) / 'test.csv', self.uncertainty_csv, 
            decison_function, **self.kwag_dict)
        clean_level = self.clean_level if clean_level is None else clean_level
        if datatype == 'all':
            test_indices = list(range(len(test_dataset)))
        elif datatype == 'clean':
            cetainties = test_dataset.certainty
            # find the indices when the value of cetainties is smaller than cetainty_level
            test_indices = np.where(cetainties <= clean_level)[0]
        elif datatype == 'noisy':
            cetainties = test_dataset.certainty
            # find the indices when the value of cetainties is bigger than cetainty_level
            test_indices = np.where(cetainties > clean_level)[0]
        else:
            raise ValueError(f"datatype {datatype} is not supported")
            
        return super().get_test_loader(test_dataset, test_indices=test_indices)


############################################################################################
########################### Dataloader for the CheXpertDataset #############################
############################################################################################

class SplitDataLoader(module_data.DataLoader):
    """
    Loading datas from a dataset with a split ratio for validation set.
    The split can be done either patient-wisely or randomly.

    Input:
        dataset <orch.utils.data.Dataset>: the name of the dataset class
        batch_size <python int>: the batch size
        shuffle <python bool>: whether to shuffle the data
        validation_split <python float or int>: the ratio of the validation set
        patient_split <python str>: the name of class attribute that correspond to  
            the csv file and the patient ID column name. The format is "csv;PatientID"
        num_workers <python int>: the number of workers for data loader

    """
    def __init__(self, dataset, batch_size, validation_split, shuffle=True,
                 patient_split="csv;PatientID", num_workers=0, 
                 collate_fn=module_default_dataloader.default_collate):
        
        self.dataset = dataset

        self.validation_split = validation_split
        self.shuffle = shuffle

        self.n_samples = len(dataset)

        if patient_split is not None and self.validation_split > 0:
            self._sampler, self._valid_sampler = self._split_sampler_patient(
                self.validation_split, patient_split)
        else: # the validation split is done randomly
            self._sampler, self._valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        
        super().__init__(sampler=self._sampler, **self.init_kwargs)

    def _split_sampler_patient(self, split, patient_split):

        csv_attribute, patient_col = patient_split.split(';')
        if not hasattr(self.dataset, csv_attribute):
            raise ValueError("The dataset class does"+ \
                             " not have the attribute {}".format(csv_attribute))
        else:
            patient_df = getattr(self.dataset, csv_attribute)
        patients = patient_df[patient_col].unique()

        np.random.seed(0)
        np.random.shuffle(patients)

        if isinstance(split, int):
            assert split > 0
            assert split < len(patients), "patient in validation is configured "+\
                                    "to be larger than total number of patients."
            len_valid = split
        else:
            len_valid = int(len(patients) * split)

        valid_patients = patients[0:len_valid]
        train_patients = np.delete(patients, np.arange(0, len_valid))

        train_idx = patient_df[patient_df[patient_col].isin(train_patients)].index
        valid_idx = patient_df[patient_df[patient_col].isin(valid_patients)].index

        train_sampler = module_sampler.SubsetRandomSampler(train_idx)
        valid_sampler = module_sampler.SubsetRandomSampler(valid_idx)

        self.shuffle = False
        self.n_samples = len(train_patients)

        return train_sampler, valid_sampler
    
    def _split_sampler(self, split):

        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured "+ \
                                            "to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = module_sampler.SubsetRandomSampler(train_idx)
        valid_sampler = module_sampler.SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def get_val_loader(self):
        if self._valid_sampler is None:
            return None
        else:
            return module_data.DataLoader(sampler=self._valid_sampler, **self.init_kwargs)


class CheXpertDataLoader(SplitDataLoader):
    """
    CheXpert data loading base on the SplitDataLoader

    Input:
        image_size <python int>: the shape of the resized image

    - The images are normalized to [0, 1] in dataset, 
    - In this dataloader, images are resized to (image_size, image_size), 
        turned to RGB, and normalized based on ImageNet statistics.
    - The dimession of the output tensor is (batch_size, 3, image_size, image_size)
    - Could be resized to 512 to retain the most information before augmentation 
      (e.g. random crop) in the model part
    """
    def __init__(self, data_dir, batch_size, image_size, encoding='onehot', validation_split=0.0, 
                 patient_split=None, num_workers=0, xrv_normalize=False):
        
        if xrv_normalize:
            # input channel = 1
            trsfm = transforms.Compose([
                module_xrv_data.XRayCenterCrop(), # just to make it square
                module_xrv_data.XRayResizer(image_size)
            ])
        else:
            # input channel = 3
            trsfm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Turn grayscale to RGB
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
        self.dataset = module_dataset.CheXpertDataset(data_dir, encoding=encoding,
                                                      transforms=trsfm, xrv_normalize=xrv_normalize)

        super().__init__(self.dataset, batch_size, validation_split, 
                         num_workers=num_workers, patient_split=patient_split)
        
    def get_additional_dim(self):
        return self.dataset.get_additional_dim()
    
    def get_dataset(self):
        return self.dataset

    
