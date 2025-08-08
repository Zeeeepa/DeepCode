#!/usr/bin/env python3
"""
Automated Environment Setup Script for StateMask Paper Reproduction

This script automatically sets up all required environments and dependencies
for reproducing the StateMask paper experiments, including:
- MuJoCo environments with proper normalization
- Special environments (Selfish Mining, CAGE Challenge 2, MetaDrive, Malware)
- Dependency management and version control
- Environment validation and configuration
"""

import os
import sys
import subprocess
import importlib
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Automated environment setup for StateMask paper reproduction"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize environment setup
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.setup_dir = Path.cwd() / "statemask_reproduction"
        self.venv_path = self.setup_dir / "venv"
        
        # Required package versions based on paper requirements
        self.required_packages = {
            "torch": ">=1.8.0",
            "stable-baselines3": ">=1.5.0",
            "tianshou": ">=0.4.8",
            "gymnasium": ">=0.26.0",
            "mujoco": ">=2.3.0",
            "numpy": ">=1.21.0",
            "scipy": ">=1.7.0",
            "matplotlib": ">=3.5.0",
            "seaborn": ">=0.11.0",
            "pandas": ">=1.3.0",
            "tqdm": ">=4.62.0",
            "tensorboard": ">=2.8.0",
            "opencv-python": ">=4.5.0",
            "imageio": ">=2.9.0",
            "pillow": ">=8.3.0"
        }
        
        # MuJoCo environments configuration
        self.mujoco_envs = {
            "Hopper-v3": {"normalize_obs": False, "normalize_reward": False},
            "Walker2d-v3": {"normalize_obs": True, "normalize_reward": False},
            "Reacher-v2": {"normalize_obs": False, "normalize_reward": False},
            "HalfCheetah-v3": {"normalize_obs": True, "normalize_reward": False}
        }
        
        # Special environments configuration
        self.special_envs = {
            "selfish_mining": {
                "repo": "https://github.com/roibarzur/pto-selfish-mining.git",
                "branch": "main",
                "install_cmd": "pip install -e ."
            },
            "cage_challenge": {
                "repo": "https://github.com/cage-challenge/cage-challenge-2.git",
                "branch": "main",
                "install_cmd": "pip install -e ."
            },
            "metadrive": {
                "package": "metadrive-simulator",
                "version": ">=0.2.5"
            },
            "malware": {
                "custom_setup": True,
                "description": "Custom malware mutation environment"
            }
        }
        
    def setup_directory_structure(self):
        """Create necessary directory structure"""
        logger.info("Setting up directory structure...")
        
        directories = [
            self.setup_dir,
            self.setup_dir / "environments",
            self.setup_dir / "configs",
            self.setup_dir / "data",
            self.setup_dir / "results",
            self.setup_dir / "logs",
            self.setup_dir / "models"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def check_system_requirements(self) -> bool:
        """Check system requirements and dependencies"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        # Check for CUDA availability (optional but recommended)
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available, using CPU")
        except ImportError:
            logger.info("PyTorch not installed yet")
        
        # Check for MuJoCo license (if needed)
        mujoco_key_path = Path.home() / ".mujoco" / "mjkey.txt"
        if not mujoco_key_path.exists():
            logger.warning("MuJoCo license key not found, using free version")
        
        return True
    
    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        logger.info("Creating virtual environment...")
        
        if self.venv_path.exists():
            logger.info("Virtual environment already exists")
            return
        
        try:
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], check=True)
            logger.info(f"Virtual environment created at {self.venv_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            raise
    
    def get_pip_command(self) -> str:
        """Get pip command for the virtual environment"""
        if sys.platform == "win32":
            return str(self.venv_path / "Scripts" / "pip")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def install_base_packages(self):
        """Install base packages with correct versions"""
        logger.info("Installing base packages...")
        
        pip_cmd = self.get_pip_command()
        
        # Upgrade pip first
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install packages in order of dependencies
        install_order = [
            "numpy", "scipy", "torch", "gymnasium", "stable-baselines3",
            "tianshou", "mujoco", "matplotlib", "seaborn", "pandas",
            "tqdm", "tensorboard", "opencv-python", "imageio", "pillow"
        ]
        
        for package in install_order:
            if package in self.required_packages:
                version_spec = self.required_packages[package]
                install_spec = f"{package}{version_spec}"
                
                try:
                    logger.info(f"Installing {install_spec}...")
                    subprocess.run([pip_cmd, "install", install_spec], check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {install_spec}: {e}")
                    raise
    
    def setup_mujoco_environments(self):
        """Setup and validate MuJoCo environments"""
        logger.info("Setting up MuJoCo environments...")
        
        # Test MuJoCo installation
        try:
            import mujoco
            import gymnasium as gym
            logger.info(f"MuJoCo version: {mujoco.__version__}")
        except ImportError as e:
            logger.error(f"Failed to import MuJoCo: {e}")
            raise
        
        # Test each environment
        for env_name, config in self.mujoco_envs.items():
            try:
                logger.info(f"Testing environment: {env_name}")
                env = gym.make(env_name)
                
                # Test basic functionality
                obs, _ = env.reset()
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Test state save/load functionality (required for StateMask)
                if hasattr(env.unwrapped, 'sim'):
                    # MuJoCo environments support state save/load
                    state = env.unwrapped.sim.get_state()
                    env.unwrapped.sim.set_state(state)
                    logger.info(f"✓ {env_name}: State save/load supported")
                else:
                    logger.warning(f"⚠ {env_name}: State save/load not available")
                
                env.close()
                logger.info(f"✓ {env_name}: Environment test passed")
                
            except Exception as e:
                logger.error(f"✗ {env_name}: Environment test failed - {e}")
                raise
    
    def setup_sparse_reward_environments(self):
        """Setup sparse reward versions of MuJoCo environments"""
        logger.info("Setting up sparse reward environments...")
        
        sparse_env_config = """
import gymnasium as gym
from gymnasium.wrappers import TransformReward
import numpy as np

class SparseRewardWrapper(TransformReward):
    '''Wrapper to create sparse reward versions of environments'''
    
    def __init__(self, env, threshold=0.1, reward_scale=1.0):
        def sparse_reward_fn(reward):
            # Only give reward if above threshold
            return reward_scale if abs(reward) > threshold else 0.0
        
        super().__init__(env, sparse_reward_fn)

# Register sparse reward environments
for env_name in ['Hopper-v3', 'Walker2d-v3', 'Reacher-v2', 'HalfCheetah-v3']:
    sparse_name = env_name.replace('-v3', '-sparse-v3').replace('-v2', '-sparse-v2')
    gym.register(
        id=sparse_name,
        entry_point=lambda env_name=env_name: SparseRewardWrapper(gym.make(env_name))
    )
"""
        
        # Save sparse environment configuration
        config_file = self.setup_dir / "configs" / "sparse_environments.py"
        with open(config_file, 'w') as f:
            f.write(sparse_env_config)
        
        logger.info(f"Sparse environment configuration saved to {config_file}")
    
    def setup_special_environments(self):
        """Setup special environments (Selfish Mining, CAGE, MetaDrive, Malware)"""
        logger.info("Setting up special environments...")
        
        pip_cmd = self.get_pip_command()
        envs_dir = self.setup_dir / "environments"
        
        for env_name, config in self.special_envs.items():
            logger.info(f"Setting up {env_name}...")
            
            if env_name == "selfish_mining":
                # Clone and install Selfish Mining environment
                repo_dir = envs_dir / "pto-selfish-mining"
                if not repo_dir.exists():
                    subprocess.run([
                        "git", "clone", config["repo"], str(repo_dir)
                    ], check=True)
                
                # Install the package
                subprocess.run([
                    pip_cmd, "install", "-e", str(repo_dir)
                ], check=True)
                
            elif env_name == "cage_challenge":
                # Clone and install CAGE Challenge 2
                repo_dir = envs_dir / "cage-challenge-2"
                if not repo_dir.exists():
                    subprocess.run([
                        "git", "clone", config["repo"], str(repo_dir)
                    ], check=True)
                
                # Install the package
                subprocess.run([
                    pip_cmd, "install", "-e", str(repo_dir)
                ], check=True)
                
            elif env_name == "metadrive":
                # Install MetaDrive
                subprocess.run([
                    pip_cmd, "install", f"{config['package']}{config['version']}"
                ], check=True)
                
            elif env_name == "malware":
                # Setup custom malware mutation environment
                self._setup_malware_environment(envs_dir)
            
            logger.info(f"✓ {env_name}: Setup completed")
    
    def _setup_malware_environment(self, envs_dir: Path):
        """Setup custom malware mutation environment"""
        malware_dir = envs_dir / "malware_mutation"
        malware_dir.mkdir(exist_ok=True)
        
        # Create basic malware environment structure
        malware_env_code = '''
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

class MalwareMutationEnv(gym.Env):
    """
    Custom Malware Mutation Environment for StateMask experiments
    
    This environment simulates malware mutation scenarios where an agent
    learns to modify malware features to evade detection.
    """
    
    def __init__(self, feature_dim: int = 100, max_mutations: int = 10):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_mutations = max_mutations
        self.current_mutations = 0
        
        # Action space: mutation operations
        self.action_space = spaces.Discrete(feature_dim)
        
        # Observation space: malware features + mutation count
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(feature_dim + 1,), dtype=np.float32
        )
        
        # Initialize malware features
        self.original_features = None
        self.current_features = None
        self.detection_threshold = 0.5
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate random malware features
        self.original_features = self.np_random.random(self.feature_dim)
        self.current_features = self.original_features.copy()
        self.current_mutations = 0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Apply mutation
        if action < self.feature_dim:
            self.current_features[action] = self.np_random.random()
            self.current_mutations += 1
        
        # Calculate detection probability
        detection_prob = self._calculate_detection_probability()
        
        # Calculate reward
        reward = self._calculate_reward(detection_prob)
        
        # Check termination conditions
        terminated = (detection_prob < self.detection_threshold or 
                     self.current_mutations >= self.max_mutations)
        truncated = False
        
        obs = self._get_observation()
        info = {
            "detection_probability": detection_prob,
            "mutations_used": self.current_mutations,
            "evasion_success": detection_prob < self.detection_threshold
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        mutation_ratio = self.current_mutations / self.max_mutations
        return np.concatenate([self.current_features, [mutation_ratio]])
    
    def _calculate_detection_probability(self) -> float:
        # Simple detection model based on feature similarity
        similarity = np.mean(np.abs(self.current_features - self.original_features))
        return max(0.1, 1.0 - similarity)
    
    def _calculate_reward(self, detection_prob: float) -> float:
        # Reward for low detection probability, penalty for mutations
        evasion_reward = (1.0 - detection_prob) * 10.0
        mutation_penalty = self.current_mutations * 0.1
        return evasion_reward - mutation_penalty

# Register the environment
gym.register(
    id='MalwareMutation-v0',
    entry_point='malware_mutation.env:MalwareMutationEnv',
    max_episode_steps=100
)
'''
        
        # Save malware environment
        env_file = malware_dir / "__init__.py"
        with open(env_file, 'w') as f:
            f.write("")
        
        env_file = malware_dir / "env.py"
        with open(env_file, 'w') as f:
            f.write(malware_env_code)
        
        # Install as editable package
        setup_py = malware_dir / "setup.py"
        setup_content = '''
from setuptools import setup, find_packages

setup(
    name="malware_mutation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy"]
)
'''
        with open(setup_py, 'w') as f:
            f.write(setup_content)
        
        pip_cmd = self.get_pip_command()
        subprocess.run([pip_cmd, "install", "-e", str(malware_dir)], check=True)
    
    def setup_normalization_wrappers(self):
        """Setup VecNormalize and other normalization wrappers"""
        logger.info("Setting up normalization wrappers...")
        
        normalization_config = '''
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np

class StateMaskNormalizationWrapper:
    """
    Normalization wrapper compatible with StateMask requirements
    
    Provides observation and reward normalization while maintaining
    state save/load functionality required for StateMask.
    """
    
    @staticmethod
    def create_normalized_env(env_id: str, normalize_obs: bool = True, 
                            normalize_reward: bool = False, **kwargs):
        """
        Create normalized environment with StateMask compatibility
        
        Args:
            env_id: Environment identifier
            normalize_obs: Whether to normalize observations
            normalize_reward: Whether to normalize rewards
            **kwargs: Additional environment arguments
        """
        
        def make_env():
            env = gym.make(env_id, **kwargs)
            env = Monitor(env)
            return env
        
        # Create vectorized environment
        vec_env = DummyVecEnv([make_env])
        
        if normalize_obs or normalize_reward:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=normalize_obs,
                norm_reward=normalize_reward,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.99,
                epsilon=1e-8
            )
        
        return vec_env
    
    @staticmethod
    def get_mujoco_env_config():
        """Get MuJoCo environment normalization configuration"""
        return {
            "Hopper-v3": {"normalize_obs": False, "normalize_reward": False},
            "Walker2d-v3": {"normalize_obs": True, "normalize_reward": False},
            "Reacher-v2": {"normalize_obs": False, "normalize_reward": False},
            "HalfCheetah-v3": {"normalize_obs": True, "normalize_reward": False}
        }

class StatePreservingWrapper(gym.Wrapper):
    """
    Wrapper that preserves state save/load functionality
    
    Required for StateMask algorithm which needs to reset environment
    to specific states during explanation generation.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._saved_states = {}
    
    def save_state(self, state_id: str = "default"):
        """Save current environment state"""
        if hasattr(self.unwrapped, 'sim'):
            # MuJoCo environments
            self._saved_states[state_id] = {
                'sim_state': self.unwrapped.sim.get_state(),
                'np_random_state': self.np_random.get_state()
            }
        else:
            # Other environments - save what we can
            self._saved_states[state_id] = {
                'np_random_state': self.np_random.get_state()
            }
        return True
    
    def load_state(self, state_id: str = "default"):
        """Load previously saved environment state"""
        if state_id not in self._saved_states:
            return False
        
        state_data = self._saved_states[state_id]
        
        if hasattr(self.unwrapped, 'sim') and 'sim_state' in state_data:
            # MuJoCo environments
            self.unwrapped.sim.set_state(state_data['sim_state'])
            self.unwrapped.sim.forward()
        
        # Restore random state
        self.np_random.set_state(state_data['np_random_state'])
        return True
    
    def has_state_save_load(self):
        """Check if environment supports state save/load"""
        return hasattr(self.unwrapped, 'sim') or len(self._saved_states) > 0
'''
        
        # Save normalization configuration
        config_file = self.setup_dir / "configs" / "normalization.py"
        with open(config_file, 'w') as f:
            f.write(normalization_config)
        
        logger.info(f"Normalization configuration saved to {config_file}")
    
    def validate_environments(self) -> Dict[str, bool]:
        """Validate all environments and their required functionality"""
        logger.info("Validating environments...")
        
        validation_results = {}
        
        # Test MuJoCo environments
        for env_name in self.mujoco_envs.keys():
            try:
                import gymnasium as gym
                env = gym.make(env_name)
                
                # Basic functionality test
                obs, _ = env.reset()
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # State save/load test
                state_save_load = False
                if hasattr(env.unwrapped, 'sim'):
                    try:
                        state = env.unwrapped.sim.get_state()
                        env.unwrapped.sim.set_state(state)
                        state_save_load = True
                    except:
                        pass
                
                env.close()
                validation_results[env_name] = {
                    'basic_functionality': True,
                    'state_save_load': state_save_load
                }
                logger.info(f"✓ {env_name}: Validation passed")
                
            except Exception as e:
                validation_results[env_name] = {
                    'basic_functionality': False,
                    'state_save_load': False,
                    'error': str(e)
                }
                logger.error(f"✗ {env_name}: Validation failed - {e}")
        
        # Test special environments
        special_env_tests = {
            'selfish_mining': 'selfish_mining',
            'cage_challenge': 'CageChallenge2-v0',
            'metadrive': 'MetaDrive-Macro-v1',
            'malware': 'MalwareMutation-v0'
        }
        
        for env_key, env_id in special_env_tests.items():
            try:
                import gymnasium as gym
                env = gym.make(env_id)
                obs, _ = env.reset()
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                env.close()
                
                validation_results[env_key] = {'basic_functionality': True}
                logger.info(f"✓ {env_key}: Validation passed")
                
            except Exception as e:
                validation_results[env_key] = {
                    'basic_functionality': False,
                    'error': str(e)
                }
                logger.warning(f"⚠ {env_key}: Validation failed - {e}")
        
        return validation_results
    
    def create_configuration_files(self):
        """Create configuration files for experiments"""
        logger.info("Creating configuration files...")
        
        # Main experiment configuration
        experiment_config = {
            'environments': {
                'mujoco': {
                    'dense_reward': list(self.mujoco_envs.keys()),
                    'sparse_reward': [
                        env.replace('-v3', '-sparse-v3').replace('-v2', '-sparse-v2')
                        for env in self.mujoco_envs.keys()
                    ],
                    'normalization': self.mujoco_envs
                },
                'special': {
                    'selfish_mining': 'selfish_mining',
                    'cage_challenge': 'CageChallenge2-v0',
                    'metadrive': 'MetaDrive-Macro-v1',
                    'malware': 'MalwareMutation-v0'
                }
            },
            'algorithms': {
                'base_agent': 'SAC',  # Default from Stable-Baselines3
                'policy_type': 'MlpPolicy',  # Default architecture
                'explanation_methods': ['StateMask', 'LIME', 'SHAP'],
                'refinement_methods': ['StateMask-R']
            },
            'training': {
                'total_timesteps': 1000000,
                'eval_freq': 10000,
                'n_eval_episodes': 10,
                'save_freq': 50000
            },
            'explanation': {
                'n_explanations': 100,
                'explanation_length': 10,
                'perturbation_std': 0.1
            }
        }
        
        # Save experiment configuration
        config_file = self.setup_dir / "configs" / "experiment_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(experiment_config, f, default_flow_style=False)
        
        # Create environment test script
        test_script = '''#!/usr/bin/env python3
"""
Environment Test Script for StateMask Reproduction

This script tests all environments and their required functionality
for StateMask paper reproduction.
"""

import sys
import gymnasium as gym
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_environment(env_id, test_state_save_load=True):
    """Test individual environment"""
    print(f"Testing {env_id}...")
    
    try:
        env = gym.make(env_id)
        
        # Basic functionality test
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        
        # Step test
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step test: OK")
        
        # State save/load test (if applicable)
        if test_state_save_load and hasattr(env.unwrapped, 'sim'):
            state = env.unwrapped.sim.get_state()
            env.unwrapped.sim.set_state(state)
            print(f"  State save/load: OK")
        
        env.close()
        print(f"  ✓ {env_id}: All tests passed")
        return True
        
    except Exception as e:
        print(f"  ✗ {env_id}: Test failed - {e}")
        return False

def main():
    """Run all environment tests"""
    print("StateMask Environment Test Suite")
    print("=" * 50)
    
    # MuJoCo environments
    mujoco_envs = [
        "Hopper-v3", "Walker2d-v3", "Reacher-v2", "HalfCheetah-v3"
    ]
    
    print("\\nTesting MuJoCo environments:")
    mujoco_results = []
    for env_id in mujoco_envs:
        result = test_environment(env_id, test_state_save_load=True)
        mujoco_results.append(result)
    
    # Special environments
    special_envs = [
        "MalwareMutation-v0"
    ]
    
    print("\\nTesting special environments:")
    special_results = []
    for env_id in special_envs:
        result = test_environment(env_id, test_state_save_load=False)
        special_results.append(result)
    
    # Summary
    print("\\n" + "=" * 50)
    print("Test Summary:")
    print(f"MuJoCo environments: {sum(mujoco_results)}/{len(mujoco_results)} passed")
    print(f"Special environments: {sum(special_results)}/{len(special_results)} passed")
    
    total_passed = sum(mujoco_results) + sum(special_results)
    total_tests = len(mujoco_results) + len(special_results)
    print(f"Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\\n✓ All environment tests passed! Ready for StateMask reproduction.")
        return 0
    else:
        print("\\n✗ Some environment tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        test_file = self.setup_dir / "test_environments.py"
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        # Make test script executable
        test_file.chmod(0o755)
        
        logger.info(f"Configuration files created in {self.setup_dir / 'configs'}")
        logger.info(f"Environment test script created: {test_file}")
    
    def generate_setup_report(self, validation_results: Dict[str, bool]):
        """Generate setup report"""
        logger.info("Generating setup report...")
        
        report = f"""
StateMask Paper Reproduction - Environment Setup Report
======================================================

Setup Directory: {self.setup_dir}
Virtual Environment: {self.venv_path}

Environment Validation Results:
------------------------------
"""
        
        for env_name, result in validation_results.items():
            if isinstance(result, dict):
                status = "✓ PASS" if result.get('basic_functionality', False) else "✗ FAIL"
                state_support = "✓" if result.get('state_save_load', False) else "✗"
                report += f"{env_name:20} | {status:8} | State Save/Load: {state_support}\n"
                if 'error' in result:
                    report += f"                     Error: {result['error']}\n"
            else:
                status = "✓ PASS" if result else "✗ FAIL"
                report += f"{env_name:20} | {status:8}\n"
        
        report += f"""
Package Versions:
----------------
"""
        
        # Check installed package versions
        for package in self.required_packages.keys():
            try:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'Unknown')
                report += f"{package:20} | {version}\n"
            except ImportError:
                report += f"{package:20} | Not installed\n"
        
        report += f"""
Next Steps:
----------
1. Run environment tests: python {self.setup_dir}/test_environments.py
2. Check configuration files in: {self.setup_dir}/configs/
3. Start StateMask reproduction experiments

For issues, check the logs in: {self.setup_dir}/logs/
"""
        
        # Save report
        report_file = self.setup_dir / "setup_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        logger.info(f"Setup report saved to {report_file}")
    
    def run_setup(self):
        """Run complete environment setup"""
        logger.info("Starting StateMask environment setup...")
        
        try:
            # Check system requirements
            if not self.check_system_requirements():
                raise RuntimeError("System requirements not met")
            
            # Setup directory structure
            self.setup_directory_structure()
            
            # Create virtual environment
            self.create_virtual_environment()
            
            # Install base packages
            self.install_base_packages()
            
            # Setup MuJoCo environments
            self.setup_mujoco_environments()
            
            # Setup sparse reward environments
            self.setup_sparse_reward_environments()
            
            # Setup special environments
            self.setup_special_environments()
            
            # Setup normalization wrappers
            self.setup_normalization_wrappers()
            
            # Create configuration files
            self.create_configuration_files()
            
            # Validate environments
            validation_results = self.validate_environments()
            
            # Generate setup report
            self.generate_setup_report(validation_results)
            
            logger.info("✓ StateMask environment setup completed successfully!")
            
        except Exception as e:
            logger.error(f"✗ Setup failed: {e}")
            raise

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated Environment Setup for StateMask Paper Reproduction"
    )
    parser.add_argument(
        "--config", type=str, help="Path to configuration file"
    )
    parser.add_argument(
        "--setup-dir", type=str, default="statemask_reproduction",
        help="Setup directory name"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip environment validation"
    )
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = EnvironmentSetup(config_path=args.config)
    if args.setup_dir:
        setup.setup_dir = Path.cwd() / args.setup_dir
        setup.venv_path = setup.setup_dir / "venv"
    
    # Run setup
    try:
        setup.run_setup()
        print("\n🎉 Environment setup completed successfully!")
        print(f"📁 Setup directory: {setup.setup_dir}")
        print(f"🧪 Run tests: python {setup.setup_dir}/test_environments.py")
        print(f"📋 Check report: {setup.setup_dir}/setup_report.txt")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Check the logs for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()