```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from huggingface_hub import HfApi
```

    d:\apps\anaconda\envs\rl\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

make_vec_env automatically use gym.make inside


```python
env_id = 'LunarLander-v3'
env = make_vec_env(env_id, n_envs=16) # Vectorized 16 environments

model = PPO(
    policy='MlpPolicy',
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.98,
    ent_coef=0.01,
    learning_rate=0.0003,
    verbose=1
)

print('Training section below: ------------------------')
model.learn(total_timesteps=1000000)

model.save('ppo-LunarLander-v3')
```

    d:\apps\anaconda\envs\rl\Lib\site-packages\pygame\pkgdata.py:25: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
      from pkg_resources import resource_stream, resource_exists
    

    Using cpu device
    Training section below: ------------------------
    ---------------------------------
    | rollout/           |          |
    |    ep_len_mean     | 91.1     |
    |    ep_rew_mean     | -196     |
    | time/              |          |
    |    fps             | 2843     |
    |    iterations      | 1        |
    |    time_elapsed    | 5        |
    |    total_timesteps | 16384    |
    ---------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 98.2         |
    |    ep_rew_mean          | -173         |
    | time/                   |              |
    |    fps                  | 1658         |
    |    iterations           | 2            |
    |    time_elapsed         | 19           |
    |    total_timesteps      | 32768        |
    | train/                  |              |
    |    approx_kl            | 0.0071233264 |
    |    clip_fraction        | 0.0573       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.38        |
    |    explained_variance   | 0.00306      |
    |    learning_rate        | 0.0003       |
    |    loss                 | 1.14e+03     |
    |    n_updates            | 4            |
    |    policy_gradient_loss | -0.00516     |
    |    value_loss           | 3.15e+03     |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 95.9         |
    |    ep_rew_mean          | -144         |
    | time/                   |              |
    |    fps                  | 1705         |
    |    iterations           | 3            |
    |    time_elapsed         | 28           |
    |    total_timesteps      | 49152        |
    | train/                  |              |
    |    approx_kl            | 0.0068540582 |
    |    clip_fraction        | 0.0489       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.37        |
    |    explained_variance   | -0.00594     |
    |    learning_rate        | 0.0003       |
    |    loss                 | 538          |
    |    n_updates            | 8            |
    |    policy_gradient_loss | -0.00531     |
    |    value_loss           | 1.51e+03     |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 92.1        |
    |    ep_rew_mean          | -116        |
    | time/                   |             |
    |    fps                  | 1674        |
    |    iterations           | 4           |
    |    time_elapsed         | 39          |
    |    total_timesteps      | 65536       |
    | train/                  |             |
    |    approx_kl            | 0.006750495 |
    |    clip_fraction        | 0.0831      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.35       |
    |    explained_variance   | -0.00179    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 298         |
    |    n_updates            | 12          |
    |    policy_gradient_loss | -0.0049     |
    |    value_loss           | 651         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 103          |
    |    ep_rew_mean          | -96.3        |
    | time/                   |              |
    |    fps                  | 1590         |
    |    iterations           | 5            |
    |    time_elapsed         | 51           |
    |    total_timesteps      | 81920        |
    | train/                  |              |
    |    approx_kl            | 0.0077616754 |
    |    clip_fraction        | 0.0864       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.35        |
    |    explained_variance   | -0.0103      |
    |    learning_rate        | 0.0003       |
    |    loss                 | 281          |
    |    n_updates            | 16           |
    |    policy_gradient_loss | -0.00681     |
    |    value_loss           | 483          |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 109         |
    |    ep_rew_mean          | -85.5       |
    | time/                   |             |
    |    fps                  | 1511        |
    |    iterations           | 6           |
    |    time_elapsed         | 65          |
    |    total_timesteps      | 98304       |
    | train/                  |             |
    |    approx_kl            | 0.008431777 |
    |    clip_fraction        | 0.0536      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.33       |
    |    explained_variance   | 4.95e-05    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 190         |
    |    n_updates            | 20          |
    |    policy_gradient_loss | -0.00696    |
    |    value_loss           | 536         |
    -----------------------------------------
    ----------------------------------------
    | rollout/                |            |
    |    ep_len_mean          | 123        |
    |    ep_rew_mean          | -75.6      |
    | time/                   |            |
    |    fps                  | 1423       |
    |    iterations           | 7          |
    |    time_elapsed         | 80         |
    |    total_timesteps      | 114688     |
    | train/                  |            |
    |    approx_kl            | 0.00886685 |
    |    clip_fraction        | 0.0473     |
    |    clip_range           | 0.2        |
    |    entropy_loss         | -1.28      |
    |    explained_variance   | 0.14       |
    |    learning_rate        | 0.0003     |
    |    loss                 | 157        |
    |    n_updates            | 24         |
    |    policy_gradient_loss | -0.00717   |
    |    value_loss           | 291        |
    ----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 150          |
    |    ep_rew_mean          | -49.5        |
    | time/                   |              |
    |    fps                  | 1365         |
    |    iterations           | 8            |
    |    time_elapsed         | 96           |
    |    total_timesteps      | 131072       |
    | train/                  |              |
    |    approx_kl            | 0.0109476345 |
    |    clip_fraction        | 0.0765       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.23        |
    |    explained_variance   | 0.216        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 152          |
    |    n_updates            | 28           |
    |    policy_gradient_loss | -0.00934     |
    |    value_loss           | 384          |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 147         |
    |    ep_rew_mean          | -30         |
    | time/                   |             |
    |    fps                  | 1334        |
    |    iterations           | 9           |
    |    time_elapsed         | 110         |
    |    total_timesteps      | 147456      |
    | train/                  |             |
    |    approx_kl            | 0.008443766 |
    |    clip_fraction        | 0.0555      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.19       |
    |    explained_variance   | 0.265       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 164         |
    |    n_updates            | 32          |
    |    policy_gradient_loss | -0.00417    |
    |    value_loss           | 369         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 168          |
    |    ep_rew_mean          | -18.3        |
    | time/                   |              |
    |    fps                  | 1264         |
    |    iterations           | 10           |
    |    time_elapsed         | 129          |
    |    total_timesteps      | 163840       |
    | train/                  |              |
    |    approx_kl            | 0.0076070107 |
    |    clip_fraction        | 0.0427       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.15        |
    |    explained_variance   | 0.413        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 280          |
    |    n_updates            | 36           |
    |    policy_gradient_loss | -0.00546     |
    |    value_loss           | 369          |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 238         |
    |    ep_rew_mean          | -15.6       |
    | time/                   |             |
    |    fps                  | 1187        |
    |    iterations           | 11          |
    |    time_elapsed         | 151         |
    |    total_timesteps      | 180224      |
    | train/                  |             |
    |    approx_kl            | 0.009257912 |
    |    clip_fraction        | 0.0847      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.13       |
    |    explained_variance   | 0.51        |
    |    learning_rate        | 0.0003      |
    |    loss                 | 130         |
    |    n_updates            | 40          |
    |    policy_gradient_loss | -0.00709    |
    |    value_loss           | 306         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 303         |
    |    ep_rew_mean          | -15         |
    | time/                   |             |
    |    fps                  | 1086        |
    |    iterations           | 12          |
    |    time_elapsed         | 180         |
    |    total_timesteps      | 196608      |
    | train/                  |             |
    |    approx_kl            | 0.005390588 |
    |    clip_fraction        | 0.0386      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.15       |
    |    explained_variance   | 0.515       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 142         |
    |    n_updates            | 44          |
    |    policy_gradient_loss | -0.00346    |
    |    value_loss           | 294         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 398          |
    |    ep_rew_mean          | -13          |
    | time/                   |              |
    |    fps                  | 1012         |
    |    iterations           | 13           |
    |    time_elapsed         | 210          |
    |    total_timesteps      | 212992       |
    | train/                  |              |
    |    approx_kl            | 0.0048512816 |
    |    clip_fraction        | 0.0382       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.18        |
    |    explained_variance   | 0.56         |
    |    learning_rate        | 0.0003       |
    |    loss                 | 67.9         |
    |    n_updates            | 48           |
    |    policy_gradient_loss | -0.0036      |
    |    value_loss           | 225          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 496          |
    |    ep_rew_mean          | -4.99        |
    | time/                   |              |
    |    fps                  | 951          |
    |    iterations           | 14           |
    |    time_elapsed         | 241          |
    |    total_timesteps      | 229376       |
    | train/                  |              |
    |    approx_kl            | 0.0054001934 |
    |    clip_fraction        | 0.0418       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.16        |
    |    explained_variance   | 0.467        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 162          |
    |    n_updates            | 52           |
    |    policy_gradient_loss | -0.00251     |
    |    value_loss           | 232          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 605          |
    |    ep_rew_mean          | 9.96         |
    | time/                   |              |
    |    fps                  | 903          |
    |    iterations           | 15           |
    |    time_elapsed         | 271          |
    |    total_timesteps      | 245760       |
    | train/                  |              |
    |    approx_kl            | 0.0053241067 |
    |    clip_fraction        | 0.0544       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.2         |
    |    explained_variance   | 0.544        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 111          |
    |    n_updates            | 56           |
    |    policy_gradient_loss | -0.00313     |
    |    value_loss           | 170          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 647          |
    |    ep_rew_mean          | 16.9         |
    | time/                   |              |
    |    fps                  | 869          |
    |    iterations           | 16           |
    |    time_elapsed         | 301          |
    |    total_timesteps      | 262144       |
    | train/                  |              |
    |    approx_kl            | 0.0064542172 |
    |    clip_fraction        | 0.0394       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.19        |
    |    explained_variance   | 0.619        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 67.2         |
    |    n_updates            | 60           |
    |    policy_gradient_loss | -0.00193     |
    |    value_loss           | 114          |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 701         |
    |    ep_rew_mean          | 23.4        |
    | time/                   |             |
    |    fps                  | 836         |
    |    iterations           | 17          |
    |    time_elapsed         | 333         |
    |    total_timesteps      | 278528      |
    | train/                  |             |
    |    approx_kl            | 0.007277415 |
    |    clip_fraction        | 0.0516      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.12       |
    |    explained_variance   | 0.413       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 99.2        |
    |    n_updates            | 64          |
    |    policy_gradient_loss | -0.00243    |
    |    value_loss           | 228         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 729         |
    |    ep_rew_mean          | 32.7        |
    | time/                   |             |
    |    fps                  | 808         |
    |    iterations           | 18          |
    |    time_elapsed         | 364         |
    |    total_timesteps      | 294912      |
    | train/                  |             |
    |    approx_kl            | 0.006417464 |
    |    clip_fraction        | 0.0518      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.14       |
    |    explained_variance   | 0.576       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 79.1        |
    |    n_updates            | 68          |
    |    policy_gradient_loss | -0.00323    |
    |    value_loss           | 131         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 739         |
    |    ep_rew_mean          | 40          |
    | time/                   |             |
    |    fps                  | 783         |
    |    iterations           | 19          |
    |    time_elapsed         | 397         |
    |    total_timesteps      | 311296      |
    | train/                  |             |
    |    approx_kl            | 0.006968148 |
    |    clip_fraction        | 0.0616      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.12       |
    |    explained_variance   | 0.625       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 49.4        |
    |    n_updates            | 72          |
    |    policy_gradient_loss | -0.00319    |
    |    value_loss           | 93.7        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 770         |
    |    ep_rew_mean          | 45.7        |
    | time/                   |             |
    |    fps                  | 756         |
    |    iterations           | 20          |
    |    time_elapsed         | 433         |
    |    total_timesteps      | 327680      |
    | train/                  |             |
    |    approx_kl            | 0.004737347 |
    |    clip_fraction        | 0.0372      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.09       |
    |    explained_variance   | 0.575       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 55.2        |
    |    n_updates            | 76          |
    |    policy_gradient_loss | -0.00178    |
    |    value_loss           | 110         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 830          |
    |    ep_rew_mean          | 59.1         |
    | time/                   |              |
    |    fps                  | 756          |
    |    iterations           | 21           |
    |    time_elapsed         | 454          |
    |    total_timesteps      | 344064       |
    | train/                  |              |
    |    approx_kl            | 0.0032763095 |
    |    clip_fraction        | 0.0356       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.09        |
    |    explained_variance   | 0.717        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 10.8         |
    |    n_updates            | 80           |
    |    policy_gradient_loss | 0.000253     |
    |    value_loss           | 50.5         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 841         |
    |    ep_rew_mean          | 67.9        |
    | time/                   |             |
    |    fps                  | 749         |
    |    iterations           | 22          |
    |    time_elapsed         | 480         |
    |    total_timesteps      | 360448      |
    | train/                  |             |
    |    approx_kl            | 0.004414697 |
    |    clip_fraction        | 0.041       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.08       |
    |    explained_variance   | 0.707       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 22          |
    |    n_updates            | 84          |
    |    policy_gradient_loss | -0.00175    |
    |    value_loss           | 64.3        |
    -----------------------------------------
    ----------------------------------------
    | rollout/                |            |
    |    ep_len_mean          | 854        |
    |    ep_rew_mean          | 76.5       |
    | time/                   |            |
    |    fps                  | 735        |
    |    iterations           | 23         |
    |    time_elapsed         | 512        |
    |    total_timesteps      | 376832     |
    | train/                  |            |
    |    approx_kl            | 0.00789126 |
    |    clip_fraction        | 0.0479     |
    |    clip_range           | 0.2        |
    |    entropy_loss         | -1.08      |
    |    explained_variance   | 0.597      |
    |    learning_rate        | 0.0003     |
    |    loss                 | 76.1       |
    |    n_updates            | 88         |
    |    policy_gradient_loss | -0.00154   |
    |    value_loss           | 112        |
    ----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 855         |
    |    ep_rew_mean          | 82.7        |
    | time/                   |             |
    |    fps                  | 721         |
    |    iterations           | 24          |
    |    time_elapsed         | 545         |
    |    total_timesteps      | 393216      |
    | train/                  |             |
    |    approx_kl            | 0.005422597 |
    |    clip_fraction        | 0.0493      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.02       |
    |    explained_variance   | 0.765       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 59.5        |
    |    n_updates            | 92          |
    |    policy_gradient_loss | -0.00107    |
    |    value_loss           | 54.6        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 879         |
    |    ep_rew_mean          | 89.3        |
    | time/                   |             |
    |    fps                  | 711         |
    |    iterations           | 25          |
    |    time_elapsed         | 575         |
    |    total_timesteps      | 409600      |
    | train/                  |             |
    |    approx_kl            | 0.005288357 |
    |    clip_fraction        | 0.0445      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.02       |
    |    explained_variance   | 0.722       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 103         |
    |    n_updates            | 96          |
    |    policy_gradient_loss | -0.00117    |
    |    value_loss           | 64.4        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 877          |
    |    ep_rew_mean          | 92           |
    | time/                   |              |
    |    fps                  | 702          |
    |    iterations           | 26           |
    |    time_elapsed         | 606          |
    |    total_timesteps      | 425984       |
    | train/                  |              |
    |    approx_kl            | 0.0056898072 |
    |    clip_fraction        | 0.0658       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.03        |
    |    explained_variance   | 0.778        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 7.86         |
    |    n_updates            | 100          |
    |    policy_gradient_loss | -0.00142     |
    |    value_loss           | 40.8         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 890          |
    |    ep_rew_mean          | 95.9         |
    | time/                   |              |
    |    fps                  | 692          |
    |    iterations           | 27           |
    |    time_elapsed         | 638          |
    |    total_timesteps      | 442368       |
    | train/                  |              |
    |    approx_kl            | 0.0044716364 |
    |    clip_fraction        | 0.0227       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.02        |
    |    explained_variance   | 0.779        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 28.2         |
    |    n_updates            | 104          |
    |    policy_gradient_loss | -0.000722    |
    |    value_loss           | 48.8         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 923          |
    |    ep_rew_mean          | 103          |
    | time/                   |              |
    |    fps                  | 683          |
    |    iterations           | 28           |
    |    time_elapsed         | 671          |
    |    total_timesteps      | 458752       |
    | train/                  |              |
    |    approx_kl            | 0.0054445686 |
    |    clip_fraction        | 0.0284       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.98        |
    |    explained_variance   | 0.815        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 4.54         |
    |    n_updates            | 108          |
    |    policy_gradient_loss | -0.00103     |
    |    value_loss           | 38.4         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 923         |
    |    ep_rew_mean          | 105         |
    | time/                   |             |
    |    fps                  | 676         |
    |    iterations           | 29          |
    |    time_elapsed         | 702         |
    |    total_timesteps      | 475136      |
    | train/                  |             |
    |    approx_kl            | 0.004945937 |
    |    clip_fraction        | 0.0561      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.975      |
    |    explained_variance   | 0.941       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 6.45        |
    |    n_updates            | 112         |
    |    policy_gradient_loss | -0.000548   |
    |    value_loss           | 11.4        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 898         |
    |    ep_rew_mean          | 104         |
    | time/                   |             |
    |    fps                  | 671         |
    |    iterations           | 30          |
    |    time_elapsed         | 731         |
    |    total_timesteps      | 491520      |
    | train/                  |             |
    |    approx_kl            | 0.003751075 |
    |    clip_fraction        | 0.031       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.98       |
    |    explained_variance   | 0.744       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 14.1        |
    |    n_updates            | 116         |
    |    policy_gradient_loss | -0.000255   |
    |    value_loss           | 58.2        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 880          |
    |    ep_rew_mean          | 103          |
    | time/                   |              |
    |    fps                  | 666          |
    |    iterations           | 31           |
    |    time_elapsed         | 762          |
    |    total_timesteps      | 507904       |
    | train/                  |              |
    |    approx_kl            | 0.0055841464 |
    |    clip_fraction        | 0.0356       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.95        |
    |    explained_variance   | 0.709        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 88.3         |
    |    n_updates            | 120          |
    |    policy_gradient_loss | -0.00108     |
    |    value_loss           | 95.1         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 877         |
    |    ep_rew_mean          | 107         |
    | time/                   |             |
    |    fps                  | 661         |
    |    iterations           | 32          |
    |    time_elapsed         | 792         |
    |    total_timesteps      | 524288      |
    | train/                  |             |
    |    approx_kl            | 0.004113046 |
    |    clip_fraction        | 0.035       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.936      |
    |    explained_variance   | 0.832       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 18          |
    |    n_updates            | 124         |
    |    policy_gradient_loss | -0.000434   |
    |    value_loss           | 51.7        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 834          |
    |    ep_rew_mean          | 105          |
    | time/                   |              |
    |    fps                  | 662          |
    |    iterations           | 33           |
    |    time_elapsed         | 816          |
    |    total_timesteps      | 540672       |
    | train/                  |              |
    |    approx_kl            | 0.0039353613 |
    |    clip_fraction        | 0.0312       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.979       |
    |    explained_variance   | 0.848        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 9.92         |
    |    n_updates            | 128          |
    |    policy_gradient_loss | -7.6e-05     |
    |    value_loss           | 42.8         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 833         |
    |    ep_rew_mean          | 106         |
    | time/                   |             |
    |    fps                  | 667         |
    |    iterations           | 34          |
    |    time_elapsed         | 833         |
    |    total_timesteps      | 557056      |
    | train/                  |             |
    |    approx_kl            | 0.004706623 |
    |    clip_fraction        | 0.0261      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.945      |
    |    explained_variance   | 0.697       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 34.6        |
    |    n_updates            | 132         |
    |    policy_gradient_loss | -0.000555   |
    |    value_loss           | 85.2        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 828          |
    |    ep_rew_mean          | 104          |
    | time/                   |              |
    |    fps                  | 674          |
    |    iterations           | 35           |
    |    time_elapsed         | 850          |
    |    total_timesteps      | 573440       |
    | train/                  |              |
    |    approx_kl            | 0.0027073384 |
    |    clip_fraction        | 0.0322       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.912       |
    |    explained_variance   | 0.798        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 16.8         |
    |    n_updates            | 136          |
    |    policy_gradient_loss | -0.000883    |
    |    value_loss           | 60.8         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 844         |
    |    ep_rew_mean          | 108         |
    | time/                   |             |
    |    fps                  | 681         |
    |    iterations           | 36          |
    |    time_elapsed         | 865         |
    |    total_timesteps      | 589824      |
    | train/                  |             |
    |    approx_kl            | 0.004292145 |
    |    clip_fraction        | 0.0332      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.916      |
    |    explained_variance   | 0.763       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 5.82        |
    |    n_updates            | 140         |
    |    policy_gradient_loss | -0.000732   |
    |    value_loss           | 67.8        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 796          |
    |    ep_rew_mean          | 100          |
    | time/                   |              |
    |    fps                  | 689          |
    |    iterations           | 37           |
    |    time_elapsed         | 878          |
    |    total_timesteps      | 606208       |
    | train/                  |              |
    |    approx_kl            | 0.0038603144 |
    |    clip_fraction        | 0.0398       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.872       |
    |    explained_variance   | 0.864        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 2.6          |
    |    n_updates            | 144          |
    |    policy_gradient_loss | -0.000371    |
    |    value_loss           | 36.1         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 775         |
    |    ep_rew_mean          | 99.5        |
    | time/                   |             |
    |    fps                  | 693         |
    |    iterations           | 38          |
    |    time_elapsed         | 897         |
    |    total_timesteps      | 622592      |
    | train/                  |             |
    |    approx_kl            | 0.005375206 |
    |    clip_fraction        | 0.0328      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.878      |
    |    explained_variance   | 0.724       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 38.7        |
    |    n_updates            | 148         |
    |    policy_gradient_loss | -0.000151   |
    |    value_loss           | 90          |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 773         |
    |    ep_rew_mean          | 104         |
    | time/                   |             |
    |    fps                  | 691         |
    |    iterations           | 39          |
    |    time_elapsed         | 923         |
    |    total_timesteps      | 638976      |
    | train/                  |             |
    |    approx_kl            | 0.004113937 |
    |    clip_fraction        | 0.0272      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.881      |
    |    explained_variance   | 0.742       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 14.9        |
    |    n_updates            | 152         |
    |    policy_gradient_loss | -0.00122    |
    |    value_loss           | 106         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 790          |
    |    ep_rew_mean          | 110          |
    | time/                   |              |
    |    fps                  | 689          |
    |    iterations           | 40           |
    |    time_elapsed         | 949          |
    |    total_timesteps      | 655360       |
    | train/                  |              |
    |    approx_kl            | 0.0034798284 |
    |    clip_fraction        | 0.0409       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.849       |
    |    explained_variance   | 0.808        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 35.1         |
    |    n_updates            | 156          |
    |    policy_gradient_loss | -0.000668    |
    |    value_loss           | 67           |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 770         |
    |    ep_rew_mean          | 108         |
    | time/                   |             |
    |    fps                  | 688         |
    |    iterations           | 41          |
    |    time_elapsed         | 975         |
    |    total_timesteps      | 671744      |
    | train/                  |             |
    |    approx_kl            | 0.007002934 |
    |    clip_fraction        | 0.0909      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.885      |
    |    explained_variance   | 0.898       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 5.02        |
    |    n_updates            | 160         |
    |    policy_gradient_loss | 0.000365    |
    |    value_loss           | 17.5        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 809          |
    |    ep_rew_mean          | 115          |
    | time/                   |              |
    |    fps                  | 686          |
    |    iterations           | 42           |
    |    time_elapsed         | 1001         |
    |    total_timesteps      | 688128       |
    | train/                  |              |
    |    approx_kl            | 0.0037452604 |
    |    clip_fraction        | 0.0291       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.881       |
    |    explained_variance   | 0.814        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 3.22         |
    |    n_updates            | 164          |
    |    policy_gradient_loss | -0.000722    |
    |    value_loss           | 63.2         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 821          |
    |    ep_rew_mean          | 112          |
    | time/                   |              |
    |    fps                  | 684          |
    |    iterations           | 43           |
    |    time_elapsed         | 1029         |
    |    total_timesteps      | 704512       |
    | train/                  |              |
    |    approx_kl            | 0.0035571326 |
    |    clip_fraction        | 0.0348       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.87        |
    |    explained_variance   | 0.852        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 2.72         |
    |    n_updates            | 168          |
    |    policy_gradient_loss | -0.000982    |
    |    value_loss           | 52.4         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 825          |
    |    ep_rew_mean          | 111          |
    | time/                   |              |
    |    fps                  | 683          |
    |    iterations           | 44           |
    |    time_elapsed         | 1054         |
    |    total_timesteps      | 720896       |
    | train/                  |              |
    |    approx_kl            | 0.0032166145 |
    |    clip_fraction        | 0.0233       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.877       |
    |    explained_variance   | 0.806        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 18.9         |
    |    n_updates            | 172          |
    |    policy_gradient_loss | 3.69e-05     |
    |    value_loss           | 64.7         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 778         |
    |    ep_rew_mean          | 103         |
    | time/                   |             |
    |    fps                  | 680         |
    |    iterations           | 45          |
    |    time_elapsed         | 1083        |
    |    total_timesteps      | 737280      |
    | train/                  |             |
    |    approx_kl            | 0.005312629 |
    |    clip_fraction        | 0.0509      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.92       |
    |    explained_variance   | 0.826       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 4.18        |
    |    n_updates            | 176         |
    |    policy_gradient_loss | -0.000475   |
    |    value_loss           | 64.4        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 763         |
    |    ep_rew_mean          | 103         |
    | time/                   |             |
    |    fps                  | 679         |
    |    iterations           | 46          |
    |    time_elapsed         | 1109        |
    |    total_timesteps      | 753664      |
    | train/                  |             |
    |    approx_kl            | 0.002292215 |
    |    clip_fraction        | 0.0182      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.928      |
    |    explained_variance   | 0.832       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 110         |
    |    n_updates            | 180         |
    |    policy_gradient_loss | -0.000206   |
    |    value_loss           | 64.6        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 771          |
    |    ep_rew_mean          | 107          |
    | time/                   |              |
    |    fps                  | 677          |
    |    iterations           | 47           |
    |    time_elapsed         | 1136         |
    |    total_timesteps      | 770048       |
    | train/                  |              |
    |    approx_kl            | 0.0037250437 |
    |    clip_fraction        | 0.0419       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.939       |
    |    explained_variance   | 0.822        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 21.8         |
    |    n_updates            | 184          |
    |    policy_gradient_loss | -0.000539    |
    |    value_loss           | 62.4         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 794          |
    |    ep_rew_mean          | 113          |
    | time/                   |              |
    |    fps                  | 676          |
    |    iterations           | 48           |
    |    time_elapsed         | 1163         |
    |    total_timesteps      | 786432       |
    | train/                  |              |
    |    approx_kl            | 0.0061068325 |
    |    clip_fraction        | 0.0628       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.929       |
    |    explained_variance   | 0.892        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 3.86         |
    |    n_updates            | 188          |
    |    policy_gradient_loss | -0.00025     |
    |    value_loss           | 30.6         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 818          |
    |    ep_rew_mean          | 118          |
    | time/                   |              |
    |    fps                  | 673          |
    |    iterations           | 49           |
    |    time_elapsed         | 1191         |
    |    total_timesteps      | 802816       |
    | train/                  |              |
    |    approx_kl            | 0.0040763365 |
    |    clip_fraction        | 0.0308       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.929       |
    |    explained_variance   | 0.829        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 2.61         |
    |    n_updates            | 192          |
    |    policy_gradient_loss | 0.000785     |
    |    value_loss           | 35.7         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 856         |
    |    ep_rew_mean          | 123         |
    | time/                   |             |
    |    fps                  | 671         |
    |    iterations           | 50          |
    |    time_elapsed         | 1220        |
    |    total_timesteps      | 819200      |
    | train/                  |             |
    |    approx_kl            | 0.005115967 |
    |    clip_fraction        | 0.0517      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.962      |
    |    explained_variance   | 0.899       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 0.986       |
    |    n_updates            | 196         |
    |    policy_gradient_loss | -0.000119   |
    |    value_loss           | 28          |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 902         |
    |    ep_rew_mean          | 128         |
    | time/                   |             |
    |    fps                  | 667         |
    |    iterations           | 51          |
    |    time_elapsed         | 1251        |
    |    total_timesteps      | 835584      |
    | train/                  |             |
    |    approx_kl            | 0.004327544 |
    |    clip_fraction        | 0.0567      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.99       |
    |    explained_variance   | 0.896       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 11.1        |
    |    n_updates            | 200         |
    |    policy_gradient_loss | -0.000458   |
    |    value_loss           | 30.2        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 931         |
    |    ep_rew_mean          | 131         |
    | time/                   |             |
    |    fps                  | 665         |
    |    iterations           | 52          |
    |    time_elapsed         | 1280        |
    |    total_timesteps      | 851968      |
    | train/                  |             |
    |    approx_kl            | 0.005437775 |
    |    clip_fraction        | 0.0558      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.958      |
    |    explained_variance   | 0.943       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 1.56        |
    |    n_updates            | 204         |
    |    policy_gradient_loss | 0.000251    |
    |    value_loss           | 16          |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 925          |
    |    ep_rew_mean          | 132          |
    | time/                   |              |
    |    fps                  | 663          |
    |    iterations           | 53           |
    |    time_elapsed         | 1307         |
    |    total_timesteps      | 868352       |
    | train/                  |              |
    |    approx_kl            | 0.0056799166 |
    |    clip_fraction        | 0.0472       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.991       |
    |    explained_variance   | 0.988        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 1.75         |
    |    n_updates            | 208          |
    |    policy_gradient_loss | -0.000492    |
    |    value_loss           | 2.71         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 925          |
    |    ep_rew_mean          | 131          |
    | time/                   |              |
    |    fps                  | 661          |
    |    iterations           | 54           |
    |    time_elapsed         | 1336         |
    |    total_timesteps      | 884736       |
    | train/                  |              |
    |    approx_kl            | 0.0030832891 |
    |    clip_fraction        | 0.0172       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.985       |
    |    explained_variance   | 0.874        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 93.7         |
    |    n_updates            | 212          |
    |    policy_gradient_loss | -0.00119     |
    |    value_loss           | 42.7         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 933          |
    |    ep_rew_mean          | 132          |
    | time/                   |              |
    |    fps                  | 659          |
    |    iterations           | 55           |
    |    time_elapsed         | 1365         |
    |    total_timesteps      | 901120       |
    | train/                  |              |
    |    approx_kl            | 0.0062792366 |
    |    clip_fraction        | 0.0471       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.981       |
    |    explained_variance   | 0.932        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 2.29         |
    |    n_updates            | 216          |
    |    policy_gradient_loss | -0.000186    |
    |    value_loss           | 20.5         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 891         |
    |    ep_rew_mean          | 129         |
    | time/                   |             |
    |    fps                  | 659         |
    |    iterations           | 56          |
    |    time_elapsed         | 1392        |
    |    total_timesteps      | 917504      |
    | train/                  |             |
    |    approx_kl            | 0.004786811 |
    |    clip_fraction        | 0.0351      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.948      |
    |    explained_variance   | 0.962       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 0.857       |
    |    n_updates            | 220         |
    |    policy_gradient_loss | -0.000166   |
    |    value_loss           | 10.2        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 868          |
    |    ep_rew_mean          | 126          |
    | time/                   |              |
    |    fps                  | 657          |
    |    iterations           | 57           |
    |    time_elapsed         | 1419         |
    |    total_timesteps      | 933888       |
    | train/                  |              |
    |    approx_kl            | 0.0035602464 |
    |    clip_fraction        | 0.0209       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.93        |
    |    explained_variance   | 0.788        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 8.13         |
    |    n_updates            | 224          |
    |    policy_gradient_loss | -0.00103     |
    |    value_loss           | 90.5         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 865         |
    |    ep_rew_mean          | 124         |
    | time/                   |             |
    |    fps                  | 657         |
    |    iterations           | 58          |
    |    time_elapsed         | 1445        |
    |    total_timesteps      | 950272      |
    | train/                  |             |
    |    approx_kl            | 0.004690166 |
    |    clip_fraction        | 0.048       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.929      |
    |    explained_variance   | 0.87        |
    |    learning_rate        | 0.0003      |
    |    loss                 | 154         |
    |    n_updates            | 228         |
    |    policy_gradient_loss | -0.000918   |
    |    value_loss           | 47.1        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 843          |
    |    ep_rew_mean          | 120          |
    | time/                   |              |
    |    fps                  | 656          |
    |    iterations           | 59           |
    |    time_elapsed         | 1471         |
    |    total_timesteps      | 966656       |
    | train/                  |              |
    |    approx_kl            | 0.0058763023 |
    |    clip_fraction        | 0.028        |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.929       |
    |    explained_variance   | 0.925        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 2.63         |
    |    n_updates            | 232          |
    |    policy_gradient_loss | 0.000212     |
    |    value_loss           | 26           |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 828          |
    |    ep_rew_mean          | 118          |
    | time/                   |              |
    |    fps                  | 656          |
    |    iterations           | 60           |
    |    time_elapsed         | 1497         |
    |    total_timesteps      | 983040       |
    | train/                  |              |
    |    approx_kl            | 0.0039461916 |
    |    clip_fraction        | 0.0378       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.947       |
    |    explained_variance   | 0.772        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 19.9         |
    |    n_updates            | 236          |
    |    policy_gradient_loss | -0.00083     |
    |    value_loss           | 87.1         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 866          |
    |    ep_rew_mean          | 126          |
    | time/                   |              |
    |    fps                  | 655          |
    |    iterations           | 61           |
    |    time_elapsed         | 1523         |
    |    total_timesteps      | 999424       |
    | train/                  |              |
    |    approx_kl            | 0.0059471997 |
    |    clip_fraction        | 0.0659       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.901       |
    |    explained_variance   | 0.873        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 2.87         |
    |    n_updates            | 240          |
    |    policy_gradient_loss | -0.00048     |
    |    value_loss           | 44.3         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 834          |
    |    ep_rew_mean          | 122          |
    | time/                   |              |
    |    fps                  | 654          |
    |    iterations           | 62           |
    |    time_elapsed         | 1551         |
    |    total_timesteps      | 1015808      |
    | train/                  |              |
    |    approx_kl            | 0.0050812555 |
    |    clip_fraction        | 0.0541       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.89        |
    |    explained_variance   | 0.941        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 22.3         |
    |    n_updates            | 244          |
    |    policy_gradient_loss | -6.81e-05    |
    |    value_loss           | 15.4         |
    ------------------------------------------
    


```python
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder=".", name_prefix="replay", episode_trigger=lambda x: True)
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()

# 3. Upload onto HuggingFace
api = HfApi(token="hf_xxx")
api.upload_file(path_or_fileobj="ppo-LunarLander-v3.zip", path_in_repo="ppo-LunarLander-v3.zip", repo_id="pexa8335/ppo-LunarLander-v3", repo_type="model")
api.upload_file(path_or_fileobj="replay-episode-0.mp4", path_in_repo="replay.mp4", repo_id="pexa8335/ppo-LunarLander-v3", repo_type="model")
print("Access link: https://huggingface.co/pexa8335/ppo-LunarLander-v3")
```

    d:\apps\anaconda\envs\rl\Lib\site-packages\gymnasium\wrappers\rendering.py:293: UserWarning: [33mWARN: Overwriting existing videos at d:\Course\RL\Policy Gradient in RL\Practice folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)[0m
      logger.warn(
    ppo-LunarLander-v3.zip: 100%|██████████| 150k/150k [00:01<00:00, 76.6kB/s] 
    

    Access link: https://huggingface.co/pexa8335/ppo-LunarLander-v3
    


```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Score: {mean_reward} +/- {std_reward}")
```

    Score: 249.91206972659762 +/- 27.173472026198272
    
