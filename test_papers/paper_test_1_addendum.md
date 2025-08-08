## Useful details

### Focus on overall results

The paper does not go into depth about the architecture of the agent used. The
main focus is on optimizations and the learning algorithms employed. The
specific architecture is therefore omitted from the reproduction -- the results
should hold with any generic RL agent policy compatible with the environments
used. Both the explanation method (as well as StateMask) and the refinement
method (as well as StateMask-R) are based on the black-box assumption. The
performance of explanation and refinement methods should therefore be
independent of the internal network structure of the target agent.

As such, the rubric does not check for exact numbers in the reproduction, but
instead verifies that the general trends and takeaways for each individual
result in the paper are reproduced.

### Architectures

It nevertheless is perhaps useful to know that the authors relied on default
policy architectures from Stable Baselines3. When asked, the authors provided
the following information:

> We utilized the default setting ('MlpPolicy') in stable baselines 3 library
> for the policy network and SAC networks in dense/sparse MuJoCo tasks.
> (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#policy-networks)
Policy Networks
Stable Baselines3 provides policy networks for images (CnnPolicies), other type of input features (MlpPolicies) and multiple different inputs (MultiInputPolicies).

Warning

For A2C and PPO, continuous actions are clipped during training and testing (to avoid out of bound error). SAC, DDPG and TD3 squash the action, using a tanh() transformation, which handles bounds more correctly.

SB3 Policy
SB3 networks are separated into two main parts (see figure below):

A features extractor (usually shared between actor and critic when applicable, to save computation) whose role is to extract features (i.e. convert to a feature vector) from high-dimensional observations, for instance, a CNN that extracts features from images. This is the features_extractor_class parameter. You can change the default parameters of that features extractor by passing a features_extractor_kwargs parameter.

A (fully-connected) network that maps the features to actions/value. Its architecture is controlled by the net_arch parameter.

Note

All observations are first pre-processed (e.g. images are normalized, discrete obs are converted to one-hot vectors, …) before being fed to the features extractor. In the case of vector observations, the features extractor is just a Flatten layer.

../_images/net_arch.png
SB3 policies are usually composed of several networks (actor/critic networks + target networks when applicable) together with the associated optimizers.

Each of these network have a features extractor followed by a fully-connected network.

Note

When we refer to “policy” in Stable-Baselines3, this is usually an abuse of language compared to RL terminology. In SB3, “policy” refers to the class that handles all the networks useful for training, so not only the network used to predict actions (the “learned controller”).

../_images/sb3_policy.png
Default Network Architecture
The default network architecture used by SB3 depends on the algorithm and the observation space. You can visualize the architecture by printing model.policy (see issue #329).

For 1D observation space, a 2 layers fully connected net is used with:

64 units (per layer) for PPO/A2C/DQN

256 units for SAC

[400, 300] units for TD3/DDPG (values are taken from the original TD3 paper)

For image observation spaces, the “Nature CNN” (see code for more details) is used for feature extraction, and SAC/TD3 also keeps the same fully connected network after it. The other algorithms only have a linear layer after the CNN. The CNN is shared between actor and critic for A2C/PPO (on-policy algorithms) to reduce computation. Off-policy algorithms (TD3, DDPG, SAC, …) have separate feature extractors: one for the actor and one for the critic, since the best performance is obtained with this configuration.

For mixed observations (dictionary observations), the two architectures from above are used, i.e., CNN for images and then two layers fully-connected network (with a smaller output size for the CNN).

Custom Network Architecture
One way of customising the policy network architecture is to pass arguments when creating the model, using policy_kwargs parameter:

Note

An extra linear layer will be added on top of the layers specified in net_arch, in order to have the right output dimensions and activation functions (e.g. Softmax for discrete actions).

In the following example, as CartPole’s action space has a dimension of 2, the final dimensions of the net_arch’s layers will be:

        obs
        <4>
   /            \
 <32>          <32>
  |              |
 <32>          <32>
  |              |
 <2>            <1>
action         value
import gymnasium as gym
import torch as th

from stable_baselines3 import PPO

# Custom actor (pi) and value function (vf) networks
# of two layers of size 32 each with Relu activation function
# Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[32, 32], vf=[32, 32]))
# Create the agent
model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
# Retrieve the environment
env = model.get_env()
# Train the agent
model.learn(total_timesteps=20_000)
# Save the agent
model.save("ppo_cartpole")

del model
# the policy_kwargs are automatically loaded
model = PPO.load("ppo_cartpole", env=env)
Custom Feature Extractor
If you want to have a custom features extractor (e.g. custom CNN when using images), you can define class that derives from BaseFeaturesExtractor and then pass it to the model when training.

Note

For on-policy algorithms, the features extractor is shared by default between the actor and the critic to save computation (when applicable). However, this can be changed setting share_features_extractor=False in the policy_kwargs (both for on-policy and off-policy algorithms).

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
model.learn(1000)
Multiple Inputs and Dictionary Observations
Stable Baselines3 supports handling of multiple inputs by using Dict Gym space. This can be done using MultiInputPolicy, which by default uses the CombinedExtractor features extractor to turn multiple inputs into a single vector, handled by the net_arch network.

By default, CombinedExtractor processes multiple inputs as follows:

If input is an image (automatically detected, see common.preprocessing.is_image_space), process image with Nature Atari CNN network and output a latent vector of size 256.

If input is not an image, flatten it (no layers).

Concatenate all previous vectors into one long vector and pass it to policy.

Much like above, you can define custom features extractors. The following example assumes the environment has two keys in the observation space dictionary: “image” is a (1,H,W) image (channel first), and “vector” is a (D,) dimensional vector. We process “image” with a simple downsampling and “vector” with a single linear layer.

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
On-Policy Algorithms
Custom Networks
If you need a network architecture that is different for the actor and the critic when using PPO, A2C or TRPO, you can pass a dictionary of the following structure: dict(pi=[<actor network architecture>], vf=[<critic network architecture>]).

For example, if you want a different architecture for the actor (aka pi) and the critic (value-function aka vf) networks, then you can specify net_arch=dict(pi=[32, 32], vf=[64, 64]).

Otherwise, to have actor and critic that share the same network architecture, you only need to specify net_arch=[128, 128] (here, two hidden layers of 128 units each, this is equivalent to net_arch=dict(pi=[128, 128], vf=[128, 128])).

If shared layers are needed, you need to implement a custom policy network (see advanced example below).

Examples
Same architecture for actor and critic with two layers of size 128: net_arch=[128, 128]

        obs
   /            \
 <128>          <128>
  |              |
 <128>          <128>
  |              |
action         value
Different architectures for actor and critic: net_arch=dict(pi=[32, 32], vf=[64, 64])

        obs
   /            \
 <32>          <64>
  |              |
 <32>          <64>
  |              |
action         value
Advanced Example
If your task requires even more granular control over the policy/value architecture, you can redefine the policy directly:

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
model.learn(5000)
Off-Policy Algorithms
If you need a network architecture that is different for the actor and the critic when using SAC, DDPG, TQC or TD3, you can pass a dictionary of the following structure: dict(pi=[<actor network architecture>], qf=[<critic network architecture>]).

For example, if you want a different architecture for the actor (aka pi) and the critic (Q-function aka qf) networks, then you can specify net_arch=dict(pi=[64, 64], qf=[400, 300]).

Otherwise, to have actor and critic that share the same network architecture, you only need to specify net_arch=[256, 256] (here, two hidden layers of 256 units each).

Note

For advanced customization of off-policy algorithms policies, please take a look at the code. A good understanding of the algorithm used is required, see discussion in issue #425

from stable_baselines3 import SAC

# Custom actor architecture with two layers of 64 units each
# Custom critic architecture with two layers of 400 and 300 units
policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
# Create the agent
model = SAC("MlpPolicy", "Pendulum-v1", policy_kwargs=policy_kwargs, verbose=1)
model.learn(5000)

> We utilized an MLP for the policy network in the selfish mining task. The
> hidden sizes are [128, 128, 128, 128].

> We utilized an MLP for the policy network in the cage challenge task. The
> hidden sizes are [64,64,64].

> We utilized the default network structure (see
> https://di-engine-docs.readthedocs.io/en/latest/_modules/ding/model/template/vac.html#VAC)
Source code for ding.model.template.vac
from typing import Union, Dict, Optional
from easydict import EasyDict
import torch
import torch.nn as nn
from copy import deepcopy
from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder, IMPALAConvEncoder
from ding.torch_utils.network.dreamer import ActionHead, DenseHead


[docs]@MODEL_REGISTRY.register('vac')
class VAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of algorithms related to (state) Value Actor-Critic (VAC), such as \
        A2C/PPO/IMPALA. This model now supports discrete, continuous and hybrid action space. The VAC is composed of \
        four parts: ``actor_encoder``, ``critic_encoder``, ``actor_head`` and ``critic_head``. Encoders are used to \
        extract the feature from various observation. Heads are used to predict corresponding value or action logit. \
        In high-dimensional observation space like 2D image, we often use a shared encoder for both ``actor_encoder`` \
        and ``critic_encoder``. In low-dimensional observation space like 1D vector, we often use different encoders.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``, ``compute_actor_critic``.
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

[docs]    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType, EasyDict],
        action_space: str = 'discrete',
        share_encoder: bool = True,
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        fixed_sigma_value: Optional[int] = 0.3,
        bound_type: Optional[str] = None,
        encoder: Optional[torch.nn.Module] = None,
        impala_cnn_encoder: bool = False,
    ) -> None:
        """
        Overview:
            Initialize the VAC model according to corresponding input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - action_space (:obj:`str`): The type of different action spaces, including ['discrete', 'continuous', \
                'hybrid'], then will instantiate corresponding head, including ``DiscreteHead``, \
                ``ReparameterizationHead``, and hybrid heads.
            - share_encoder (:obj:`bool`): Whether to share observation encoders between actor and decoder.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element is used as the input size of ``actor_head`` and ``critic_head``.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of ``actor_head`` network, defaults \
                to 64, it is the hidden size of the last layer of the ``actor_head`` network.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the ``actor_head`` network to compute action.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of ``critic_head`` network, defaults \
                to 64, it is the hidden size of the last layer of the ``critic_head`` network.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the ``critic_head`` network.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details. you can choose one of ['BN', 'IN', 'SyncBN', 'LN']
            - sigma_type (:obj:`Optional[str]`): The type of sigma in continuous action space, see \
                ``ding.torch_utils.network.dreamer.ReparameterizationHead`` for more details, in A2C/PPO, it defaults \
                to ``independent``, which means state-independent sigma parameters.
            - fixed_sigma_value (:obj:`Optional[int]`): If ``sigma_type`` is ``fixed``, then use this value as sigma.
            - bound_type (:obj:`Optional[str]`): The type of action bound methods in continuous action space, defaults \
                to ``None``, which means no bound.
            - encoder (:obj:`Optional[torch.nn.Module]`): The encoder module, defaults to ``None``, you can define \
                your own encoder module and pass it into VAC to deal with different observation space.
            - impala_cnn_encoder (:obj:`bool`): Whether to use IMPALA CNN encoder, defaults to ``False``.
        """
        super(VAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.obs_shape, self.action_shape = obs_shape, action_shape
        self.impala_cnn_encoder = impala_cnn_encoder
        self.share_encoder = share_encoder

        # Encoder Type
        def new_encoder(outsize, activation):
            if impala_cnn_encoder:
                return IMPALAConvEncoder(obs_shape=obs_shape, channels=encoder_hidden_size_list, outsize=outsize)
            else:
                if isinstance(obs_shape, int) or len(obs_shape) == 1:
                    return FCEncoder(
                        obs_shape=obs_shape,
                        hidden_size_list=encoder_hidden_size_list,
                        activation=activation,
                        norm_type=norm_type
                    )
                elif len(obs_shape) == 3:
                    return ConvEncoder(
                        obs_shape=obs_shape,
                        hidden_size_list=encoder_hidden_size_list,
                        activation=activation,
                        norm_type=norm_type
                    )
                else:
                    raise RuntimeError(
                        "not support obs_shape for pre-defined encoder: {}, please customize your own encoder".
                        format(obs_shape)
                    )

        if self.share_encoder:
            if encoder:
                if isinstance(encoder, torch.nn.Module):
                    self.encoder = encoder
                else:
                    raise ValueError("illegal encoder instance.")
            else:
                self.encoder = new_encoder(encoder_hidden_size_list[-1], activation)
        else:
            if encoder:
                if isinstance(encoder, torch.nn.Module):
                    self.actor_encoder = encoder
                    self.critic_encoder = deepcopy(encoder)
                else:
                    raise ValueError("illegal encoder instance.")
            else:
                self.actor_encoder = new_encoder(encoder_hidden_size_list[-1], activation)
                self.critic_encoder = new_encoder(encoder_hidden_size_list[-1], activation)

        # Head Type
        self.critic_head = RegressionHead(
            encoder_hidden_size_list[-1],
            1,
            critic_head_layer_num,
            activation=activation,
            norm_type=norm_type,
            hidden_size=critic_head_hidden_size
        )
        self.action_space = action_space
        assert self.action_space in ['discrete', 'continuous', 'hybrid'], self.action_space
        if self.action_space == 'continuous':
            self.multi_head = False
            self.actor_head = ReparameterizationHead(
                encoder_hidden_size_list[-1],
                action_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type,
                hidden_size=actor_head_hidden_size,
            )
        elif self.action_space == 'discrete':
            actor_head_cls = DiscreteHead
            multi_head = not isinstance(action_shape, int)
            self.multi_head = multi_head
            if multi_head:
                self.actor_head = MultiHead(
                    actor_head_cls,
                    actor_head_hidden_size,
                    action_shape,
                    layer_num=actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            else:
                self.actor_head = actor_head_cls(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
        elif self.action_space == 'hybrid':  # HPPO
            # hybrid action space: action_type(discrete) + action_args(continuous),
            # such as {'action_type_shape': torch.LongTensor([0]), 'action_args_shape': torch.FloatTensor([0.1, -0.27])}
            action_shape.action_args_shape = squeeze(action_shape.action_args_shape)
            action_shape.action_type_shape = squeeze(action_shape.action_type_shape)
            actor_action_args = ReparameterizationHead(
                encoder_hidden_size_list[-1],
                action_shape.action_args_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                fixed_sigma_value=fixed_sigma_value,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type,
                hidden_size=actor_head_hidden_size,
            )
            actor_action_type = DiscreteHead(
                actor_head_hidden_size,
                action_shape.action_type_shape,
                actor_head_layer_num,
                activation=activation,
                norm_type=norm_type,
            )
            self.actor_head = nn.ModuleList([actor_action_type, actor_action_args])

        if self.share_encoder:
            self.actor = [self.encoder, self.actor_head]
            self.critic = [self.encoder, self.critic_head]
        else:
            self.actor = [self.actor_encoder, self.actor_head]
            self.critic = [self.critic_encoder, self.critic_head]
        # Convenient for calling some apis (e.g. self.critic.parameters()),
        # but may cause misunderstanding when `print(self)`
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)


[docs]    def forward(self, x: torch.Tensor, mode: str) -> Dict:
        """
        Overview:
            VAC forward computation graph, input observation tensor to predict state value or action logit. Different \
            ``mode`` will forward with different network modules to get different outputs and save computation.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
            - mode (:obj:`str`): The forward mode, all the modes are defined in the beginning of this class.
        Returns:
            - outputs (:obj:`Dict`): The output dict of VAC's forward computation graph, whose key-values vary from \
                different ``mode``.

        Examples (Actor):
            >>> model = VAC(64, 128)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 128])

        Examples (Critic):
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 64])

        Examples (Actor-Critic):
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> assert critic_outputs['value'].shape == torch.Size([4])
            >>> assert outputs['logit'].shape == torch.Size([4, 64])

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(x)


[docs]    def compute_actor(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            VAC forward computation graph for actor part, input observation tensor to predict action logit.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict`): The output dict of VAC's forward computation graph for actor, including ``logit``.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): The predicted action logit tensor, for discrete action space, it will be \
                the same dimension real-value ranged tensor of possible action choices, and for continuous action \
                space, it will be the mu and sigma of the Gaussian distribution, and the number of mu and sigma is the \
                same as the number of continuous actions. Hybrid action space is a kind of combination of discrete \
                and continuous action space, so the logit will be a dict with ``action_type`` and ``action_args``.
        Shapes:
            - logit (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``

        Examples:
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 64])
        """
        if self.share_encoder:
            x = self.encoder(x)
        else:
            x = self.actor_encoder(x)

        if self.action_space == 'discrete':
            return self.actor_head(x)
        elif self.action_space == 'continuous':
            x = self.actor_head(x)  # mu, sigma
            return {'logit': x}
        elif self.action_space == 'hybrid':
            action_type = self.actor_head[0](x)
            action_args = self.actor_head[1](x)
            return {'logit': {'action_type': action_type['logit'], 'action_args': action_args}}


[docs]    def compute_critic(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            VAC forward computation graph for critic part, input observation tensor to predict state value.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict`): The output dict of VAC's forward computation graph for critic, including ``value``.
        ReturnsKeys:
            - value (:obj:`torch.Tensor`): The predicted state value tensor.
        Shapes:
            - value (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch size, (B, 1) is squeezed to (B, ).

        Examples:
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> assert critic_outputs['value'].shape == torch.Size([4])
        """
        if self.share_encoder:
            x = self.encoder(x)
        else:
            x = self.critic_encoder(x)
        x = self.critic_head(x)
        return {'value': x['pred']}


[docs]    def compute_actor_critic(self, x: torch.Tensor) -> Dict:
        """
        Overview:
            VAC forward computation graph for both actor and critic part, input observation tensor to predict action \
            logit and state value.
        Arguments:
            - x (:obj:`torch.Tensor`): The input observation tensor data.
        Returns:
            - outputs (:obj:`Dict`): The output dict of VAC's forward computation graph for both actor and critic, \
                including ``logit`` and ``value``.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): The predicted action logit tensor, for discrete action space, it will be \
                the same dimension real-value ranged tensor of possible action choices, and for continuous action \
                space, it will be the mu and sigma of the Gaussian distribution, and the number of mu and sigma is the \
                same as the number of continuous actions. Hybrid action space is a kind of combination of discrete \
                and continuous action space, so the logit will be a dict with ``action_type`` and ``action_args``.
            - value (:obj:`torch.Tensor`): The predicted state value tensor.
        Shapes:
            - logit (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch size, (B, 1) is squeezed to (B, ).

        Examples:
            >>> model = VAC(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> assert critic_outputs['value'].shape == torch.Size([4])
            >>> assert outputs['logit'].shape == torch.Size([4, 64])


        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder and return the combination \
            dict output.
        """
        if self.share_encoder:
            actor_embedding = critic_embedding = self.encoder(x)
        else:
            actor_embedding = self.actor_encoder(x)
            critic_embedding = self.critic_encoder(x)

        value = self.critic_head(critic_embedding)['pred']

        if self.action_space == 'discrete':
            logit = self.actor_head(actor_embedding)['logit']
            return {'logit': logit, 'value': value}
        elif self.action_space == 'continuous':
            x = self.actor_head(actor_embedding)
            return {'logit': x, 'value': value}
        elif self.action_space == 'hybrid':
            action_type = self.actor_head[0](actor_embedding)
            action_args = self.actor_head[1](actor_embedding)
            return {'logit': {'action_type': action_type['logit'], 'action_args': action_args}, 'value': value}



[docs]@MODEL_REGISTRY.register('dreamervac')
class DREAMERVAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of DreamerV3 (state) Value Actor-Critic (VAC).
        This model now supports discrete, continuous action space.
    Interfaces:
        ``__init__``, ``forward``.
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

[docs]    def __init__(
            self,
            action_shape: Union[int, SequenceType, EasyDict],
            dyn_stoch=32,
            dyn_deter=512,
            dyn_discrete=32,
            actor_layers=2,
            value_layers=2,
            units=512,
            act='SiLU',
            norm='LayerNorm',
            actor_dist='normal',
            actor_init_std=1.0,
            actor_min_std=0.1,
            actor_max_std=1.0,
            actor_temp=0.1,
            action_unimix_ratio=0.01,
    ) -> None:
        """
        Overview:
            Initialize the ``DREAMERVAC`` model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
        """
        super(DREAMERVAC, self).__init__()
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape

        if dyn_discrete:
            feat_size = dyn_stoch * dyn_discrete + dyn_deter
        else:
            feat_size = dyn_stoch + dyn_deter
        self.actor = ActionHead(
            feat_size,  # pytorch version
            action_shape,
            actor_layers,
            units,
            act,
            norm,
            actor_dist,
            actor_init_std,
            actor_min_std,
            actor_max_std,
            actor_temp,
            outscale=1.0,
            unimix_ratio=action_unimix_ratio,
        )
        self.critic = DenseHead(
            feat_size,  # pytorch version
            (255, ),
            value_layers,
            units,
            'SiLU',  # act
            'LN',  # norm
            'twohot_symlog',
            outscale=0.0,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

> for the policy network in the autonomous driving task.

### Clarifying the Fidelity Score and Top-K Critical Steps

The fidelity score pipeline is as follows:

- The **explanation method** for which we wish to measure fidelity score (e.g.,
  StateMask) generates step-level importance scores for the trajectory,
  identifying how critical each step is to the agent's final reward.
- A **sliding window** of size $l = L \times K$ (where $L$ is the total
  trajectory length and $K$ is a fraction) is applied to find the **most
  critical segment** of consecutive steps. The segment with the highest average
  importance score is selected.
- The agent is **fast-forwarded** to the start of the identified critical
  segment, and **random actions** are taken for the duration of the segment
  (i.e., for $l$ steps).
- After the random actions, the agent's **policy is allowed to continue**
  executing normally until the episode ends.
- The change in reward is measured:
  - $R$ is the original episode reward (with no randomization).
  - $R'$ is the reward after randomizing actions in the critical segment.
  - The reward change is computed as $d = |R' - R|$.
- The **maximum possible reward change** ($d_{\text{max}}$) is defined as the
  largest reward value that the environment allows within a single episode,
  based on the environment's specifications.
- The **fidelity score** is calculated using the formula:
  $$
  \text{Fidelity Score} = \log \left( \frac{d}{d_{\text{max}}} \right) - \log \left( \frac{l}{L} \right).
  $$
- A higher fidelity score indicates that the explanation method has identified
  steps that are truly critical to the agent's performance.
- The finishing of the episode is governed by the environment end conditions,
  e.g. in some MuJoCo games, the episode end condition is that either the game
  reaches 1000 steps or the agent is unhealthy.

### Hyperparameter figures

Figures 7 and 8 show slightly different information than what is outlined in the
surrounding text in the paper. Specifically what they are truly plotting is:

- Figure 7 is plotting $\lambda$ for $\lambda \in \{0, 0.1, 0.01, 0.001\}$
- Figure 8 is plotting $p$ for $p \in \{0, 0.25, 0.5, 0.75, 1\}$

The legends for each figure are correct.

## Out of scope for reproduction

Section 3.4 is out of scope for the reproduction of this paper.

The refining results of the Sparse-Walker2d game and the hyper-parameter
sensitivity results of all sparse MuJoCo games are out of scope for the
reproduction of this paper.

The qualitative analysis of the autonomous driving case is out of scope for the
reproduction of this paper.

In "Refining based on Different Explanations.", the paper claims "using the
explanation generated by our mask network, the refining achieves the best
outcome across all applications". This claimed result is not significant however
-- as shown in Table 1, under the "Fix Refine; Vary Explanation Methods", the
refining performance achieved when using the "Ours" explanation method is
comparable to the refining performance achieved when using the "StateMask"
explanation method.

This particular claimed result is therefore ignored when judging a reproduction
attempt.

Experiments directly concerned with the Malware Mutation environment are out of
scope for the reproduction of this paper and can therefore be ignored. Note that
if the same experiment is run for other environments, those parts are still in
scope.
