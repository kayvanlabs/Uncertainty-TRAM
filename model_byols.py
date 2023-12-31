import torch
import torch.nn as nn
import copy
from functools import wraps
import torch.nn.functional as F
import torchvision.models as vision_models
import augmentations as module_augment


def singleton(cache_key):
    """Decorator to make a property a singleton."""
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA():
    """Exponential moving average."""
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(),
                                         ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class MLP(nn.Module):
    """
    MLP class for projector and predictor, with 2 layers and ReLU activation.
    """
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_size, projection_size))

    def forward(self, x):
        return self.net(x)


class NetWrapper(nn.Module):
    """
    Wrapper class for the backbone and the projection network
    """
    def __init__(self, net, projection_size, projection_hidden_size, layer_drop=-1):
        super().__init__()
        if layer_drop is None:
            self.net = net
            print('Use the backbone as is')
        else:
            modules = list(net.children())[:layer_drop]
            modules.append(torch.nn.Flatten())
            self.net = torch.nn.Sequential(*modules)
            print(
                'Create backbone network as Sequential'+\
                    f' module by slicing it from layers [0..{layer_drop}]'
            )

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def forward(self, x, return_embedding=False):
        representation = self.net(x)
        if return_embedding:
            return representation
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class BYOL(nn.Module):
    """
    BYOL model class. The forward() method directly returns the loss.
    
    Args:
        backbone: the backbone network
        image_size: the size of the input image
        device: the device to run the model
        pretrained: whether to use the pretrained backbone network
        drop_layer: the layer two drop from the backbone network before 
            feature extraction
        projection_size: the size of the projection
        augmentation: the augmentation function
        projection_hidden_size: the size of the hidden layer in the projection network
        moving_average_decay: the decay rate for the moving average
        use_momentum: whether to use momentum for the target network
    
    """
    def __init__(self,
                 backbone,
                 image_size,
                 augmentation,
                 pretrained: bool = True,
                 drop_layer: int = -2,
                 projection_size: int = 256,
                 projection_hidden_size: int = 4096,
                 moving_average_decay: float = 0.996,
                 use_momentum: bool = True,
                 in_channels: int = 3):
        
        super().__init__()

        # The backbone network
        if isinstance(backbone, str) and hasattr(vision_models, backbone):
            backbone = getattr(vision_models, backbone)(pretrained=pretrained)
    
        self.drop_layer = drop_layer
        self.image_size = image_size
        self.pretrained = pretrained
        self.in_channels = in_channels

        if in_channels == 1 and backbone.__class__.__name__ == 'ResNet':
            updated_conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            torch.nn.init.normal_(updated_conv1.weight, std=0.01)  
            backbone.conv1 = updated_conv1

        self._net = backbone

        # The online encoder include the backbone network and the projection network
        self.online_encoder = NetWrapper(self._net, projection_size, 
                                        projection_hidden_size,
                                        layer_drop=drop_layer)
        # The online predictor network 
        self.online_predictor = MLP(projection_size, projection_size,
                                    projection_hidden_size)
        
        # Initialize the target encoder
        self.target_encoder = None

        # Initialize the augmentation function
        if isinstance(augmentation, nn.Module):
            self.augmentation = augmentation
        elif isinstance(augmentation, str):
            self.augmentation = getattr(
                module_augment, augmentation)(image_size=self.image_size)

        # Set the EMA updater
        self.use_momentum = use_momentum
        self.target_ema_updater = EMA(moving_average_decay)

        self.monk_embedding = None

    @torch.no_grad()
    def init_network(self, device):
        # send a mock image tensor to instantiate singleton parameters
        monk_image_tensor = torch.randn(2, self.in_channels, self.image_size, 
                                        self.image_size, device=device)
        _ = self.forward(monk_image_tensor)
        # send a mock image tensor to get the dimension of the embedding
        self.monk_embedding = self.online_encoder(monk_image_tensor, 
                                                  return_embedding=True)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    @property
    def net(self):
        return self._net
    
    def get_drop_layer(self):
        return self.drop_layer
    
    def get_image_size(self):
        return self.image_size
    
    def reset_augmentation(self, augmentation):
        if isinstance(augmentation, nn.Module):
            self.augmentation = augmentation
        elif isinstance(augmentation, str) and hasattr(module_augment, augmentation):
            self.augmentation = getattr(
                module_augment, augmentation)(size=self.image_size)

    def get_embedding_dimension(self):
        if self.monk_embedding is None:
            raise Exception("Embedding dimension not set yet. "+\
                            "Call init_target_encoder() first")
        return self.monk_embedding.shape[1]

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'momentum for the target encoder turned off'
        assert self.target_encoder is not None, 'target encoder '+\
            'has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder,
                              self.online_encoder)
    
    def features(self, x):
        # Return the embedding of the online encoder
        return self.online_encoder(x, return_embedding=True)

    def forward(self, x):

        with torch.no_grad():
            # Augment the image with the defined augmentation function
            aug = self.augmentation(x.repeat(2, 1, 1, 1))
            image_one = aug[:aug.size(0) // 2, :]
            image_two = aug[aug.size(0) // 2:, :]

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder(
            ) if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = byol_loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = byol_loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss

class DetachNetWrapper(nn.Module):
    """
    Wrapper class for the backbone and the projection network, 
    where the embedding can be detached
    from the computational graph before sending it to the projector.
    """
    def __init__(self, net, projection_size, projection_hidden_size, layer_drop=-1):
        super().__init__()
        if layer_drop is None:
            self.net = net
            print('Use the backbone as is')
        else:
            modules = list(net.children())[:layer_drop]
            modules.append(torch.nn.Flatten())
            self.net = torch.nn.Sequential(*modules)
            print(
                'Create backbone network as Sequential '+ \
                    f'module by slicing it from layers [0..{layer_drop}]'
            )

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def forward(self, x, return_embedding=False, detach=False):
        representation = self.net(x)
        if return_embedding:
            return representation
        projector = self._get_projector(representation)
        projection = projector(representation.detach()) if detach \
            else projector(representation)
        return projection, representation
    

class TRAM(nn.Module):
    """
    TRAM model class. The forward() method directly returns the tram loss.
    Inputs:
        pi_input_dim <python int>: dimension of the privileged 
            information input to the pi encoder
        pi_encode_embed_dim <python int>: dimension of the 
            embedding layer in the pi encoder
        base_input_dim <python int>: dimension of the base 
            input to the pi and base coencoder
        pi_encode_hid_dim <python int>: dimension of the 
            hidden layer in the pi encoder
        pi_encode_project_dim <python int>: dimension of 
            the projection layer in the pi encoder
    Not used:
        loss_factor <python float>: factor to scale the loss by
        loss_type <python str>: type of loss to use. Options are 'byol'
    """
    def __init__(self, 
                 pi_input_dim:int,
                 base_input_dim:int,
                 pi_encode_embed_dim:int=64,
                 pi_encode_hid_dim:int=1024,
                 pi_encode_project_dim:int=256,
                 projection_size:int=256,
                 projection_hidden_size:int=4096):
        
        super().__init__()
        self.pi_encode = nn.Sequential(nn.Linear(pi_input_dim, pi_encode_embed_dim),
                                 nn.BatchNorm1d(pi_encode_embed_dim),
                                 nn.ReLU(inplace=True))
        
        self.pi_co_encoder = MLP(dim=pi_encode_embed_dim+base_input_dim,
                                 projection_size=pi_encode_project_dim,
                                 hidden_size=pi_encode_hid_dim,)
        
        self.pi_projector = MLP(dim=pi_encode_project_dim,
                                projection_size=projection_size,
                                hidden_size=projection_hidden_size,)
    
        
    def forward(self, base_x_embed, pi_x):
        pi_embed = self.pi_encode(pi_x)
        pi_and_base_embed = torch.cat((pi_embed, base_x_embed), dim=1)
        pi_projection = self.pi_projector(self.pi_co_encoder(pi_and_base_embed))
        return pi_projection


class BYOL_TRAM(nn.Module):
    """
    BYOL TRAM model class. The forward() method directly returns 
    the byol loss plus the tram branch loss.
    
    Args:
        backbone: the backbone network
        image_size: the size of the input image
        device: the device to run the model
        pretrained: whether to use the pretrained backbone network
        drop_layer: the layer two drop from the backbone 
            network before feature extraction
        projection_size: the size of the projection
        augmentation: the augmentation function
        projection_hidden_size: the size of the hidden layer in the projection network
        moving_average_decay: the decay rate for the moving average
        use_momentum: whether to use momentum for the target network
    
    """
    def __init__(self,
                 backbone,
                 image_size,
                 augmentation,
                 pi_input_dim:int,
                 loss_factor:float=1.0,
                 loss_type:str='byol',
                 pi_encode_embed_dim:int=64,
                 pi_encode_hid_dim:int=1024,
                 pi_encode_project_dim:int=256,
                 projection_size: int = 256,
                 projection_hidden_size: int = 4096,
                 pretrained: bool = True,
                 drop_layer: int = -1,
                 moving_average_decay: float = 0.996,
                 use_momentum: bool = True,
                 in_channels: int = 3):
        
        super().__init__()

        # The backbone network
        if isinstance(backbone, str) and hasattr(vision_models, backbone):
            backbone = getattr(vision_models, backbone)(pretrained=pretrained)
    
        self.drop_layer = drop_layer
        self.image_size = image_size
        self.pretrained = pretrained
        self.in_channels = in_channels

        if in_channels == 1 and backbone.__class__.__name__ == 'ResNet':
            updated_conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            torch.nn.init.normal_(updated_conv1.weight, std=0.01)  
            backbone.conv1 = updated_conv1

        self._net = backbone
        self.tram_args = {"pi_input_dim":pi_input_dim,
                            "pi_encode_embed_dim":pi_encode_embed_dim,
                            "pi_encode_hid_dim":pi_encode_hid_dim,
                            "pi_encode_project_dim":pi_encode_project_dim,
                            "projection_size":projection_size,
                            "projection_hidden_size":projection_hidden_size}
        
        self.tram_network = None
        self.loss_factor = loss_factor
        self.loss_type = loss_type

        # The online encoder include the backbone network and the projection network
        self.online_encoder = DetachNetWrapper(self._net, projection_size, 
                                        projection_hidden_size,
                                        layer_drop=drop_layer)
        # The online predictor network 
        self.online_predictor = MLP(projection_size, projection_size,
                                    projection_hidden_size)
        
        # Initialize the target encoder
        self.target_encoder = None

        # Initialize the augmentation function
        if isinstance(augmentation, nn.Module):
            self.augmentation = augmentation
        elif isinstance(augmentation, str):
            self.augmentation = getattr(
                module_augment, augmentation)(image_size=self.image_size)

        # Set the EMA updater
        self.use_momentum = use_momentum
        self.target_ema_updater = EMA(moving_average_decay)

        self.monk_embedding = None

    @torch.no_grad()
    def init_network(self, device):
        # send a mock image tensor to instantiate singleton parameters
        monk_image_tensor = torch.randn(2, self.in_channels, self.image_size, 
                                        self.image_size, device=device)
        _ = self.forward(monk_image_tensor)
        # send a mock image tensor to get the dimension of the embedding
        self.monk_embedding = self.online_encoder(monk_image_tensor, 
                                                  return_embedding=True)
        self.tram_args["base_input_dim"] = self.monk_embedding.shape[1]
        self.tram_network = TRAM(**self.tram_args).to(device)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    @property
    def net(self):
        return self._net
    
    def get_drop_layer(self):
        return self.drop_layer
    
    def get_image_size(self):
        return self.image_size
    
    def reset_augmentation(self, augmentation):
        if isinstance(augmentation, nn.Module):
            self.augmentation = augmentation
        elif isinstance(augmentation, str) and hasattr(module_augment, augmentation):
            self.augmentation = getattr(
                module_augment, augmentation)(size=self.image_size)

    def get_embedding_dimension(self):
        if self.monk_embedding is None:
            raise Exception("Embedding dimension not set yet. "+\
                            "Call init_target_encoder() first")
        return self.monk_embedding.shape[1]

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'momentum for the target encoder turned off'
        assert self.target_encoder is not None, 'target encoder has '+\
            'not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder,
                              self.online_encoder)
    
    def features(self, x):
        # Return the embedding of the online encoder
        return self.online_encoder(x, return_embedding=True)

    def forward(self, x, pi_x=None, train=False, only_tram=False, only_byol=False):
        """
        Input:
            x: the input image
            pi_x: the input privliged information
            train: whether or not the network is in training loop, if True, 
                the training mode will be decided by only_tram and only_byol flags
            only_tram: whether or not to only train the TRAM network in this step
            only_byol: whether or not to only train the BYOL network in this step
        
            """
        if (only_tram is False) or (train is False):
            
            with torch.no_grad():
                aug = self.augmentation(x.repeat(2, 1, 1, 1))
                image_one = aug[:aug.size(0) // 2, :]
                image_two = aug[aug.size(0) // 2:, :]

            online_proj_one, _ = self.online_encoder(image_one)
            online_proj_two, _ = self.online_encoder(image_two)

            online_pred_one = self.online_predictor(online_proj_one)
            online_pred_two = self.online_predictor(online_proj_two)

            with torch.no_grad():
                target_encoder = self._get_target_encoder(
                ) if self.use_momentum else self.online_encoder
                target_proj_one, _ = target_encoder(image_one)
                target_proj_two, _ = target_encoder(image_two)
                target_proj_one.detach_()
                target_proj_two.detach_()

            loss_one = byol_loss_fn(online_pred_one, target_proj_two.detach())
            loss_two = byol_loss_fn(online_pred_two, target_proj_one.detach())

            byol_loss = loss_one + loss_two

            if (only_byol is True) or (train is False):
                return byol_loss 

        if (train is True) and (pi_x is not None) and (only_byol is False):
            x_proj, x_embed = self.online_encoder(x, detach=True)
            tram_proj = self.tram_network(x_embed, pi_x)

            tram_loss = self.loss_factor * byol_loss_fn(x_proj, tram_proj)
            return tram_loss if only_tram else (tram_loss, byol_loss)

