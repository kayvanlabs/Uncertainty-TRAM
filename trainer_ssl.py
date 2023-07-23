import torch
from abc import abstractmethod
from numpy import inf

import metric as module_metric
import logger_utils as module_log
import torch.nn.functional as torch_functional
import math
import main_utils as util_module


# Class Inheritance of trainer_ssl.py
# |--SSL_BaseTrainer
#      |--SSL_Trainer
#          |--BYOL_Trainer

class SSL_BaseTrainer:
    """
    Base class for all self-supervised learning (SSL) trainers.
    Setup the training logic with early stopping, model saving, logging, etc.
    Inputs:
        logger: logger for logging information
        num_epochs: number of epochs to train
        save_period: number of epochs to save model
        monitor_mode: mode to monitor model performance, 'min' or 'max' with the target metric
        early_stop: number of epochs to wait before early stopping
        start_epoch: the epoch number to start training
        lr_scheduler: learning rate scheduler
    """
    def __init__(self, save_dir, logger, num_epochs, save_period, 
                 monitor_mode, start_epoch=0, early_stop=0):

        self.epochs = int(num_epochs)
        self.start_epoch = int(start_epoch)
        
        self.monitor_on = int(early_stop) > 0
        # configuration to monitor model performance and save best
        if self.monitor_on:
            self.mnt_mode, self.mnt_metric = monitor_mode.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = int(early_stop)
        else:
            self.mnt_mode = 'off'
            self.mnt_best = 0

        self.logger = logger
        self.metric_tracker = module_metric.FullMetricTracker()

        self.save_dir = save_dir
        self.save_period = int(save_period)

    
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic with early stopping and model saving
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # The result is a dictory containing training loss and metrics, 
            # validation loss and metrics
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            self.metric_tracker.update(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, 
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not based on mnt_metric
                    improved = (self.mnt_mode == 'min' and log[
                        self.mnt_metric] <= self.mnt_best) or (self.mnt_mode == 'max' and log[
                        self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is " + \
                                        "disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            self._save_checkpoint(epoch, save_best=best)
            self._save_metric_tracker()

    @abstractmethod
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        """
        raise NotImplementedError
    
    def _save_metric_tracker(self):
        """
        Saving metric tracker
        """
        filename = str(self.save_dir / 'metric_tracker.csv')
        if util_module.is_main_process():
            self.metric_tracker.get_data().to_csv(filename, index=False)
        self.logger.info(f"Saving metric tracker ...")

class SSL_Trainer(SSL_BaseTrainer):
    """
    Trainer class for the SSL model
    Inputs:
        config <ConfigParser instance>: containing all the configurations
        learner <torch.nn.Module>: the ssl learner model
        device <torch.device>: device to run the model
        train_loader <torch.utils.data.DataLoader>: training data loader
        val_loader <torch.utils.data.DataLoader>: validation data loader
        feature_func <python string>: name of the feature function to 
            extract features from the encoder
        start_epoch <python int>: the epoch number to start training
        lr_scheduler <torch.optim.lr_scheduler>: learning rate scheduler
        distributed <python bool>: if True, use distributed training
    """
    def __init__(self, config, learner, optimizer, device, train_loader, val_loader, logger,
                 feature_func='features', start_epoch=1, lr_scheduler=None, distributed=False):

        self.cfg_trainer = config['trainer']

        # This super().__init__() will call the __init__() of the SSL_BaseTrainer class, 
        # which will initialize the following variables used in the BYOL_Trainer class:
        # self.logger, self.learner, self.optimizer, self.save_dir and self.scheduler,
        # among the rest mentioned in self.cfg_trainer.
        super().__init__(config.save_dir, logger, self.cfg_trainer['epochs'], 
                         self.cfg_trainer['save_period'], self.cfg_trainer['monitor'], 
                         start_epoch, self.cfg_trainer['early_stop'])
        
        self.learner = learner
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # obtain the log directory from the config instance
        self.log_dir = config.log_dir

        # Setup the device which to run the model
        self.device = device

        # Setup the distributed training
        self.distributed = distributed

        # Setup the training and validation data loader
        self.data_loader = train_loader
        self.val_data_loader = val_loader

        # Setup the feature function to extract features from the encoder
        self.feature_func = feature_func
    
        # Initilzate the tensorboard writer 
        self.writer = module_log.get_tensorboard_writer(
            self.log_dir, logger, self.cfg_trainer['tensorboard'], distributed)

        # Initialize the metrics tracker
        self.self_define_metric = ['average_std', 'reference_std']
        self.train_metrics = module_metric.MetricTracker('loss', *[m for m in self.self_define_metric], 
                                                         writer=self.writer)
        self.valid_metrics = module_metric.MetricTracker('loss', writer=self.writer)
    
    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        Input:
            epoch <python int>: current training epoch.
        Output:
            log <python dict>: a log that contains information about validation
        """
        if self.distributed:
            self.val_data_loader.sampler.set_epoch(epoch)

        self.learner.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for _, data in enumerate(self.val_data_loader):
                image = data['img'].to(self.device)

                ssl_loss = self.learner(image)
                loss = ssl_loss.mean()
                self.valid_metrics.update('loss', loss.item())

        return self.valid_metrics.result()
    
    # @abstractmethod
    # def _save_checkpoint(self, epoch, save_best=False):
    #     raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        Inputs:
            epoch <python int>: current epoch number
            save_best <python bool>: if True, rename the saved checkpoint to 
                'checkpoint_best.pth' and save to the desired location

        The self.learner, self.optimizer, and self.lr_scheduler are saved in the checkpoint
        The self.learner should have the net attribute, which is the backbone network
            function get_embedding_dimension(), get_drop_layer() and get_image_size() 
            should be implemented in the self.learner
        """
        # prepare the checkpoint dictionary for resuming training
        model_state_dict = self.learner.module.state_dict() if hasattr(
            self.learner, 'module') else self.learner.state_dict()
        checkpoint = {'epoch': epoch,
            'learner': model_state_dict,
            'optimizer': self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        # prepare the backbone dictionary for down-stream tasks
        save_target = self.learner.module if hasattr(self.learner, 'module') else self.learner
        backbone = save_target.net
        backbone_state_dict = backbone.module.state_dict() if hasattr(
            backbone, 'module') else backbone.state_dict()
        backbone_checkpoint = {'backbone': backbone_state_dict,
                               'feature_dim': save_target.get_embedding_dimension(),
                               'layer_drop':save_target.get_drop_layer(),
                               'image_size': save_target.get_image_size()}
        
        # save the checkpoint
        filename = str(self.save_dir / f'checkpoint-epoch{epoch}.pth')
        backbone_filename = str(self.save_dir / f'backbone-epoch{epoch}.pth')
        if epoch % self.save_period == 0:
            util_module.save_on_master(checkpoint, filename)
            util_module.save_on_master(backbone_checkpoint, backbone_filename)
            self.logger.info(f"Saving checkpoint on epoch {epoch} ...")

        # save the best checkpoint
        if save_best:
            best_path = str(self.save_dir / 'checkpoint_best.pth')
            best_backbone_path = str(self.save_dir / 'backbone_best.pth')
            util_module.save_on_master(checkpoint, best_path)
            util_module.save_on_master(backbone_checkpoint, best_backbone_path)
            self.logger.info("Saving current best: checkpoint_best.pth ...")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.data_loader)
        return base.format(current, total, 100.0 * current / total)
    

class BYOL_Trainer(SSL_Trainer):

    def __init__(self, *kwags, **kwargs):
        super().__init__(*kwags, **kwargs)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        Input:
            epoch <python int>: current training epoch.
        Output:
            log <python dict>: a log that contains average loss and metric in this epoch.
        """
        if self.distributed:
            self.data_loader.sampler.set_epoch(epoch)

        self.learner.train(True)
        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):

            image = data['img'].to(self.device)

            self.optimizer.zero_grad()
            ssl_loss = self.learner(image)
            loss = ssl_loss.mean()
            loss.backward()

            self.optimizer.step()
            if hasattr(self.learner, 'module'):
                self.learner.module.update_moving_average()
            else:
                self.learner.update_moving_average()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.train_metrics.update('loss', loss.item())

            with torch.no_grad():
                # size of embedding is (batch_size, embedding_dim=2048)
                if hasattr(self.learner, 'module'):
                    embedding = getattr(self.learner.module, self.feature_func)(image)
                else:
                    embedding = getattr(self.learner, self.feature_func)(image)
                l2_normalized = torch_functional.normalize(embedding, p=2, dim=-1)
                avg_std = torch.mean(torch.std(l2_normalized, dim=-1))

            self.train_metrics.update('average_std', avg_std.item())
            self.train_metrics.update('reference_std', 1 / math.sqrt(embedding.shape[1]))

            if len(self.data_loader) > 20:
                if batch_idx % (int(len(self.data_loader)/20)) == 0: #
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))

        log = self.train_metrics.result()

        if self.val_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log
