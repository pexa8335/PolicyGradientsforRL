```python
import gymnasium

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login 

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
```

    d:\apps\anaconda\envs\rl\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

### Gymnasium and how it works

The library contains our environments. We'll use Gymnasium a lot in Deep RL.  
The Gymnasium library provides two things:  
- An interface that allows you to create RL environments.  
- A collection of environments (gym-control, atari, box2d).

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process_game.jpg" alt="The RL process" width="100%">

At each step:
- Our Agent receives a **state ($S_{0}$)** from the **Environment** — we receive the first frame of our game (Environment).
- Based on that **state ($S_{0}$),** the Agent takes an **action ($A_{0}$)** — our Agent will move to the right.
- The environment transitions to a **new** **state ($S_{1}$)** — new frame.
- The environment gives some **reward ($R_{1}$)** to the Agent — we’re not dead *(Positive Reward +1)*.


With Gymnasium:

1. We create our environment using `gymnasium.make(<environment_name>)`

2. We reset the environment to its initial state with `observation, info = env.reset()`

At each step:

3. Get an action using our model (in our example we take a random action)

4. Using `env.step(action)`, we perform this action in the environment and get
- `observation`: The new state (st+1)
- `reward`: The reward we get after executing the action
- `terminated`: Indicates if the episode terminated (agent reach the terminal state)
- `truncated`: Introduced with this new version, it indicates a timelimit or if an agent go out of bounds of the environment for instance.
- `info`: A dictionary that provides additional information (depends on the environment).

For more explanations: https://gymnasium.farama.org/api/env/#gymnasium.Env.step

If the episode is terminated:
- We reset the environment to its initial state with `observation = env.reset()`  

To take a random action:
- action = env.action_space.sample() 



```python
import gymnasium as gym
```


```python
env = gym.make('LunarLander-v3')
```

    /usr/local/lib/python3.12/dist-packages/pygame/pkgdata.py:25: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
      from pkg_resources import resource_stream, resource_exists
    /usr/local/lib/python3.12/dist-packages/pkg_resources/__init__.py:3154: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google')`.
    Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
      declare_namespace(pkg)
    /usr/local/lib/python3.12/dist-packages/pkg_resources/__init__.py:3154: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google.cloud')`.
    Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
      declare_namespace(pkg)
    /usr/local/lib/python3.12/dist-packages/pkg_resources/__init__.py:3154: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('sphinxcontrib')`.
    Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
      declare_namespace(pkg)
    /usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
      return datetime.utcnow().replace(tzinfo=utc)
    /usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
      return datetime.utcnow().replace(tzinfo=utc)
    


```python
observation, info = env.reset()
```


```python
for _ in range(20):
    action = env.action_space.sample()
    print(f'Action taken: {action}')

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print('Environment is reset')
        observation, info = env.reset()

env.close()
```

    Action taken: 1
    Action taken: 2
    Action taken: 3
    Action taken: 0
    Action taken: 3
    Action taken: 3
    Action taken: 1
    Action taken: 0
    Action taken: 0
    Action taken: 2
    Action taken: 0
    Action taken: 2
    Action taken: 2
    Action taken: 1
    Action taken: 0
    Action taken: 0
    Action taken: 3
    Action taken: 2
    Action taken: 1
    Action taken: 2
    

### Create the LunarLander environment and understand how it works


```python
env = gym.make('LunarLander-v3')
observation, info = env.reset()
print('Observation space shape', env.observation_space.shape)
print('Sample observation', env.observation_space.sample)
```

    Observation space shape (8,)
    Sample observation <bound method Box.sample of Box([ -2.5        -2.5       -10.        -10.         -6.2831855 -10.
      -0.         -0.       ], [ 2.5        2.5       10.        10.         6.2831855 10.
      1.         1.       ], (8,), float32)>
    


```python
print('Action space shape', env.action_space.n)
print('Action space sample', env.action_space.sample())
```

    Action space shape 4
    Action space sample 3
    

### Vectorized environment 
A method for stacking multiple independent environments into a single environment - this way, we'll have more diverse experiences during the training phase.


```python
env = make_vec_env('LunarLander-v3', n_envs=16)
```


```python
model = PPO(
    policy = 'MlpPolicy', 
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1
)
```

    Using cpu device
    

    /usr/local/lib/python3.12/dist-packages/jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
      return datetime.utcnow().replace(tzinfo=utc)
    


```python
model.learn(total_timesteps=1e6)
model_name = 'ppo-LunarLander-v3'
model.save(model_name)
```

    ---------------------------------
    | rollout/           |          |
    |    ep_len_mean     | 92.5     |
    |    ep_rew_mean     | -188     |
    | time/              |          |
    |    fps             | 3677     |
    |    iterations      | 1        |
    |    time_elapsed    | 4        |
    |    total_timesteps | 16384    |
    ---------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 91.7         |
    |    ep_rew_mean          | -159         |
    | time/                   |              |
    |    fps                  | 2349         |
    |    iterations           | 2            |
    |    time_elapsed         | 13           |
    |    total_timesteps      | 32768        |
    | train/                  |              |
    |    approx_kl            | 0.0058881836 |
    |    clip_fraction        | 0.0423       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.38        |
    |    explained_variance   | -3.97e-05    |
    |    learning_rate        | 0.0003       |
    |    loss                 | 1.54e+03     |
    |    n_updates            | 4            |
    |    policy_gradient_loss | -0.00546     |
    |    value_loss           | 5.1e+03      |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 91.7         |
    |    ep_rew_mean          | -159         |
    | time/                   |              |
    |    fps                  | 2349         |
    |    iterations           | 2            |
    |    time_elapsed         | 13           |
    |    total_timesteps      | 32768        |
    | train/                  |              |
    |    approx_kl            | 0.0058881836 |
    |    clip_fraction        | 0.0423       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.38        |
    |    explained_variance   | -3.97e-05    |
    |    learning_rate        | 0.0003       |
    |    loss                 | 1.54e+03     |
    |    n_updates            | 4            |
    |    policy_gradient_loss | -0.00546     |
    |    value_loss           | 5.1e+03      |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 89.6         |
    |    ep_rew_mean          | -119         |
    | time/                   |              |
    |    fps                  | 2105         |
    |    iterations           | 3            |
    |    time_elapsed         | 23           |
    |    total_timesteps      | 49152        |
    | train/                  |              |
    |    approx_kl            | 0.0056763757 |
    |    clip_fraction        | 0.0387       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.37        |
    |    explained_variance   | -0.0153      |
    |    learning_rate        | 0.0003       |
    |    loss                 | 1.18e+03     |
    |    n_updates            | 8            |
    |    policy_gradient_loss | -0.00485     |
    |    value_loss           | 2.87e+03     |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 89.6         |
    |    ep_rew_mean          | -119         |
    | time/                   |              |
    |    fps                  | 2105         |
    |    iterations           | 3            |
    |    time_elapsed         | 23           |
    |    total_timesteps      | 49152        |
    | train/                  |              |
    |    approx_kl            | 0.0056763757 |
    |    clip_fraction        | 0.0387       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.37        |
    |    explained_variance   | -0.0153      |
    |    learning_rate        | 0.0003       |
    |    loss                 | 1.18e+03     |
    |    n_updates            | 8            |
    |    policy_gradient_loss | -0.00485     |
    |    value_loss           | 2.87e+03     |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 94.8        |
    |    ep_rew_mean          | -103        |
    | time/                   |             |
    |    fps                  | 2053        |
    |    iterations           | 4           |
    |    time_elapsed         | 31          |
    |    total_timesteps      | 65536       |
    | train/                  |             |
    |    approx_kl            | 0.008137133 |
    |    clip_fraction        | 0.0566      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.36       |
    |    explained_variance   | -0.000256   |
    |    learning_rate        | 0.0003      |
    |    loss                 | 584         |
    |    n_updates            | 12          |
    |    policy_gradient_loss | -0.00504    |
    |    value_loss           | 1.12e+03    |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 94.8        |
    |    ep_rew_mean          | -103        |
    | time/                   |             |
    |    fps                  | 2053        |
    |    iterations           | 4           |
    |    time_elapsed         | 31          |
    |    total_timesteps      | 65536       |
    | train/                  |             |
    |    approx_kl            | 0.008137133 |
    |    clip_fraction        | 0.0566      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.36       |
    |    explained_variance   | -0.000256   |
    |    learning_rate        | 0.0003      |
    |    loss                 | 584         |
    |    n_updates            | 12          |
    |    policy_gradient_loss | -0.00504    |
    |    value_loss           | 1.12e+03    |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 92.9        |
    |    ep_rew_mean          | -93.2       |
    | time/                   |             |
    |    fps                  | 2018        |
    |    iterations           | 5           |
    |    time_elapsed         | 40          |
    |    total_timesteps      | 81920       |
    | train/                  |             |
    |    approx_kl            | 0.006320959 |
    |    clip_fraction        | 0.0754      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.35       |
    |    explained_variance   | -2.86e-06   |
    |    learning_rate        | 0.0003      |
    |    loss                 | 302         |
    |    n_updates            | 16          |
    |    policy_gradient_loss | -0.00468    |
    |    value_loss           | 618         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 92.9        |
    |    ep_rew_mean          | -93.2       |
    | time/                   |             |
    |    fps                  | 2018        |
    |    iterations           | 5           |
    |    time_elapsed         | 40          |
    |    total_timesteps      | 81920       |
    | train/                  |             |
    |    approx_kl            | 0.006320959 |
    |    clip_fraction        | 0.0754      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.35       |
    |    explained_variance   | -2.86e-06   |
    |    learning_rate        | 0.0003      |
    |    loss                 | 302         |
    |    n_updates            | 16          |
    |    policy_gradient_loss | -0.00468    |
    |    value_loss           | 618         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 99.9        |
    |    ep_rew_mean          | -89.4       |
    | time/                   |             |
    |    fps                  | 2021        |
    |    iterations           | 6           |
    |    time_elapsed         | 48          |
    |    total_timesteps      | 98304       |
    | train/                  |             |
    |    approx_kl            | 0.008031471 |
    |    clip_fraction        | 0.0814      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.32       |
    |    explained_variance   | -0.000268   |
    |    learning_rate        | 0.0003      |
    |    loss                 | 407         |
    |    n_updates            | 20          |
    |    policy_gradient_loss | -0.00534    |
    |    value_loss           | 492         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 99.9        |
    |    ep_rew_mean          | -89.4       |
    | time/                   |             |
    |    fps                  | 2021        |
    |    iterations           | 6           |
    |    time_elapsed         | 48          |
    |    total_timesteps      | 98304       |
    | train/                  |             |
    |    approx_kl            | 0.008031471 |
    |    clip_fraction        | 0.0814      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.32       |
    |    explained_variance   | -0.000268   |
    |    learning_rate        | 0.0003      |
    |    loss                 | 407         |
    |    n_updates            | 20          |
    |    policy_gradient_loss | -0.00534    |
    |    value_loss           | 492         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 113         |
    |    ep_rew_mean          | -69.9       |
    | time/                   |             |
    |    fps                  | 1960        |
    |    iterations           | 7           |
    |    time_elapsed         | 58          |
    |    total_timesteps      | 114688      |
    | train/                  |             |
    |    approx_kl            | 0.009450234 |
    |    clip_fraction        | 0.0872      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.29       |
    |    explained_variance   | 0.000642    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 235         |
    |    n_updates            | 24          |
    |    policy_gradient_loss | -0.00487    |
    |    value_loss           | 618         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 113         |
    |    ep_rew_mean          | -69.9       |
    | time/                   |             |
    |    fps                  | 1960        |
    |    iterations           | 7           |
    |    time_elapsed         | 58          |
    |    total_timesteps      | 114688      |
    | train/                  |             |
    |    approx_kl            | 0.009450234 |
    |    clip_fraction        | 0.0872      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.29       |
    |    explained_variance   | 0.000642    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 235         |
    |    n_updates            | 24          |
    |    policy_gradient_loss | -0.00487    |
    |    value_loss           | 618         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 106         |
    |    ep_rew_mean          | -48.1       |
    | time/                   |             |
    |    fps                  | 1935        |
    |    iterations           | 8           |
    |    time_elapsed         | 67          |
    |    total_timesteps      | 131072      |
    | train/                  |             |
    |    approx_kl            | 0.011266822 |
    |    clip_fraction        | 0.0891      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.26       |
    |    explained_variance   | -0.000616   |
    |    learning_rate        | 0.0003      |
    |    loss                 | 285         |
    |    n_updates            | 28          |
    |    policy_gradient_loss | -0.00576    |
    |    value_loss           | 425         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 106         |
    |    ep_rew_mean          | -48.1       |
    | time/                   |             |
    |    fps                  | 1935        |
    |    iterations           | 8           |
    |    time_elapsed         | 67          |
    |    total_timesteps      | 131072      |
    | train/                  |             |
    |    approx_kl            | 0.011266822 |
    |    clip_fraction        | 0.0891      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.26       |
    |    explained_variance   | -0.000616   |
    |    learning_rate        | 0.0003      |
    |    loss                 | 285         |
    |    n_updates            | 28          |
    |    policy_gradient_loss | -0.00576    |
    |    value_loss           | 425         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 113         |
    |    ep_rew_mean          | -28.5       |
    | time/                   |             |
    |    fps                  | 1892        |
    |    iterations           | 9           |
    |    time_elapsed         | 77          |
    |    total_timesteps      | 147456      |
    | train/                  |             |
    |    approx_kl            | 0.009195689 |
    |    clip_fraction        | 0.0628      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.23       |
    |    explained_variance   | -0.00332    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 222         |
    |    n_updates            | 32          |
    |    policy_gradient_loss | -0.00465    |
    |    value_loss           | 463         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 113         |
    |    ep_rew_mean          | -28.5       |
    | time/                   |             |
    |    fps                  | 1892        |
    |    iterations           | 9           |
    |    time_elapsed         | 77          |
    |    total_timesteps      | 147456      |
    | train/                  |             |
    |    approx_kl            | 0.009195689 |
    |    clip_fraction        | 0.0628      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.23       |
    |    explained_variance   | -0.00332    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 222         |
    |    n_updates            | 32          |
    |    policy_gradient_loss | -0.00465    |
    |    value_loss           | 463         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 125         |
    |    ep_rew_mean          | -16.4       |
    | time/                   |             |
    |    fps                  | 1853        |
    |    iterations           | 10          |
    |    time_elapsed         | 88          |
    |    total_timesteps      | 163840      |
    | train/                  |             |
    |    approx_kl            | 0.008346175 |
    |    clip_fraction        | 0.0251      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.2        |
    |    explained_variance   | -0.00654    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 220         |
    |    n_updates            | 36          |
    |    policy_gradient_loss | -0.00317    |
    |    value_loss           | 465         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 125         |
    |    ep_rew_mean          | -16.4       |
    | time/                   |             |
    |    fps                  | 1853        |
    |    iterations           | 10          |
    |    time_elapsed         | 88          |
    |    total_timesteps      | 163840      |
    | train/                  |             |
    |    approx_kl            | 0.008346175 |
    |    clip_fraction        | 0.0251      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.2        |
    |    explained_variance   | -0.00654    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 220         |
    |    n_updates            | 36          |
    |    policy_gradient_loss | -0.00317    |
    |    value_loss           | 465         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 179          |
    |    ep_rew_mean          | -10.4        |
    | time/                   |              |
    |    fps                  | 1739         |
    |    iterations           | 11           |
    |    time_elapsed         | 103          |
    |    total_timesteps      | 180224       |
    | train/                  |              |
    |    approx_kl            | 0.0051751346 |
    |    clip_fraction        | 0.0225       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.18        |
    |    explained_variance   | 5.6e-06      |
    |    learning_rate        | 0.0003       |
    |    loss                 | 283          |
    |    n_updates            | 40           |
    |    policy_gradient_loss | -0.00195     |
    |    value_loss           | 489          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 179          |
    |    ep_rew_mean          | -10.4        |
    | time/                   |              |
    |    fps                  | 1739         |
    |    iterations           | 11           |
    |    time_elapsed         | 103          |
    |    total_timesteps      | 180224       |
    | train/                  |              |
    |    approx_kl            | 0.0051751346 |
    |    clip_fraction        | 0.0225       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.18        |
    |    explained_variance   | 5.6e-06      |
    |    learning_rate        | 0.0003       |
    |    loss                 | 283          |
    |    n_updates            | 40           |
    |    policy_gradient_loss | -0.00195     |
    |    value_loss           | 489          |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 234         |
    |    ep_rew_mean          | -7.04       |
    | time/                   |             |
    |    fps                  | 1615        |
    |    iterations           | 12          |
    |    time_elapsed         | 121         |
    |    total_timesteps      | 196608      |
    | train/                  |             |
    |    approx_kl            | 0.008934779 |
    |    clip_fraction        | 0.0567      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.18       |
    |    explained_variance   | 5.9e-06     |
    |    learning_rate        | 0.0003      |
    |    loss                 | 277         |
    |    n_updates            | 44          |
    |    policy_gradient_loss | -0.00139    |
    |    value_loss           | 649         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 234         |
    |    ep_rew_mean          | -7.04       |
    | time/                   |             |
    |    fps                  | 1615        |
    |    iterations           | 12          |
    |    time_elapsed         | 121         |
    |    total_timesteps      | 196608      |
    | train/                  |             |
    |    approx_kl            | 0.008934779 |
    |    clip_fraction        | 0.0567      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.18       |
    |    explained_variance   | 5.9e-06     |
    |    learning_rate        | 0.0003      |
    |    loss                 | 277         |
    |    n_updates            | 44          |
    |    policy_gradient_loss | -0.00139    |
    |    value_loss           | 649         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 323         |
    |    ep_rew_mean          | 0.484       |
    | time/                   |             |
    |    fps                  | 1480        |
    |    iterations           | 13          |
    |    time_elapsed         | 143         |
    |    total_timesteps      | 212992      |
    | train/                  |             |
    |    approx_kl            | 0.008318882 |
    |    clip_fraction        | 0.0602      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.18       |
    |    explained_variance   | -0.00196    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 335         |
    |    n_updates            | 48          |
    |    policy_gradient_loss | -0.0014     |
    |    value_loss           | 695         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 323         |
    |    ep_rew_mean          | 0.484       |
    | time/                   |             |
    |    fps                  | 1480        |
    |    iterations           | 13          |
    |    time_elapsed         | 143         |
    |    total_timesteps      | 212992      |
    | train/                  |             |
    |    approx_kl            | 0.008318882 |
    |    clip_fraction        | 0.0602      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.18       |
    |    explained_variance   | -0.00196    |
    |    learning_rate        | 0.0003      |
    |    loss                 | 335         |
    |    n_updates            | 48          |
    |    policy_gradient_loss | -0.0014     |
    |    value_loss           | 695         |
    -----------------------------------------
    ----------------------------------------
    | rollout/                |            |
    |    ep_len_mean          | 379        |
    |    ep_rew_mean          | 2.91       |
    | time/                   |            |
    |    fps                  | 1392       |
    |    iterations           | 14         |
    |    time_elapsed         | 164        |
    |    total_timesteps      | 229376     |
    | train/                  |            |
    |    approx_kl            | 0.00904475 |
    |    clip_fraction        | 0.0959     |
    |    clip_range           | 0.2        |
    |    entropy_loss         | -1.19      |
    |    explained_variance   | 4.3e-05    |
    |    learning_rate        | 0.0003     |
    |    loss                 | 379        |
    |    n_updates            | 52         |
    |    policy_gradient_loss | -0.00156   |
    |    value_loss           | 556        |
    ----------------------------------------
    ----------------------------------------
    | rollout/                |            |
    |    ep_len_mean          | 379        |
    |    ep_rew_mean          | 2.91       |
    | time/                   |            |
    |    fps                  | 1392       |
    |    iterations           | 14         |
    |    time_elapsed         | 164        |
    |    total_timesteps      | 229376     |
    | train/                  |            |
    |    approx_kl            | 0.00904475 |
    |    clip_fraction        | 0.0959     |
    |    clip_range           | 0.2        |
    |    entropy_loss         | -1.19      |
    |    explained_variance   | 4.3e-05    |
    |    learning_rate        | 0.0003     |
    |    loss                 | 379        |
    |    n_updates            | 52         |
    |    policy_gradient_loss | -0.00156   |
    |    value_loss           | 556        |
    ----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 464          |
    |    ep_rew_mean          | 6.25         |
    | time/                   |              |
    |    fps                  | 1314         |
    |    iterations           | 15           |
    |    time_elapsed         | 187          |
    |    total_timesteps      | 245760       |
    | train/                  |              |
    |    approx_kl            | 0.0044895452 |
    |    clip_fraction        | 0.00943      |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.18        |
    |    explained_variance   | 0.00365      |
    |    learning_rate        | 0.0003       |
    |    loss                 | 205          |
    |    n_updates            | 56           |
    |    policy_gradient_loss | -0.00255     |
    |    value_loss           | 460          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 464          |
    |    ep_rew_mean          | 6.25         |
    | time/                   |              |
    |    fps                  | 1314         |
    |    iterations           | 15           |
    |    time_elapsed         | 187          |
    |    total_timesteps      | 245760       |
    | train/                  |              |
    |    approx_kl            | 0.0044895452 |
    |    clip_fraction        | 0.00943      |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.18        |
    |    explained_variance   | 0.00365      |
    |    learning_rate        | 0.0003       |
    |    loss                 | 205          |
    |    n_updates            | 56           |
    |    policy_gradient_loss | -0.00255     |
    |    value_loss           | 460          |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 510         |
    |    ep_rew_mean          | 4.98        |
    | time/                   |             |
    |    fps                  | 1240        |
    |    iterations           | 16          |
    |    time_elapsed         | 211         |
    |    total_timesteps      | 262144      |
    | train/                  |             |
    |    approx_kl            | 0.003460229 |
    |    clip_fraction        | 0.00539     |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.2        |
    |    explained_variance   | 0.263       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 159         |
    |    n_updates            | 60          |
    |    policy_gradient_loss | -0.00159    |
    |    value_loss           | 359         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 510         |
    |    ep_rew_mean          | 4.98        |
    | time/                   |             |
    |    fps                  | 1240        |
    |    iterations           | 16          |
    |    time_elapsed         | 211         |
    |    total_timesteps      | 262144      |
    | train/                  |             |
    |    approx_kl            | 0.003460229 |
    |    clip_fraction        | 0.00539     |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.2        |
    |    explained_variance   | 0.263       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 159         |
    |    n_updates            | 60          |
    |    policy_gradient_loss | -0.00159    |
    |    value_loss           | 359         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 567          |
    |    ep_rew_mean          | 17.8         |
    | time/                   |              |
    |    fps                  | 1179         |
    |    iterations           | 17           |
    |    time_elapsed         | 236          |
    |    total_timesteps      | 278528       |
    | train/                  |              |
    |    approx_kl            | 0.0038572643 |
    |    clip_fraction        | 0.0219       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.17        |
    |    explained_variance   | 0.566        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 64.1         |
    |    n_updates            | 64           |
    |    policy_gradient_loss | -0.00202     |
    |    value_loss           | 274          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 567          |
    |    ep_rew_mean          | 17.8         |
    | time/                   |              |
    |    fps                  | 1179         |
    |    iterations           | 17           |
    |    time_elapsed         | 236          |
    |    total_timesteps      | 278528       |
    | train/                  |              |
    |    approx_kl            | 0.0038572643 |
    |    clip_fraction        | 0.0219       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.17        |
    |    explained_variance   | 0.566        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 64.1         |
    |    n_updates            | 64           |
    |    policy_gradient_loss | -0.00202     |
    |    value_loss           | 274          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 642          |
    |    ep_rew_mean          | 31           |
    | time/                   |              |
    |    fps                  | 1135         |
    |    iterations           | 18           |
    |    time_elapsed         | 259          |
    |    total_timesteps      | 294912       |
    | train/                  |              |
    |    approx_kl            | 0.0069073783 |
    |    clip_fraction        | 0.0218       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.14        |
    |    explained_variance   | 0.763        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 49.7         |
    |    n_updates            | 68           |
    |    policy_gradient_loss | -0.00221     |
    |    value_loss           | 164          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 642          |
    |    ep_rew_mean          | 31           |
    | time/                   |              |
    |    fps                  | 1135         |
    |    iterations           | 18           |
    |    time_elapsed         | 259          |
    |    total_timesteps      | 294912       |
    | train/                  |              |
    |    approx_kl            | 0.0069073783 |
    |    clip_fraction        | 0.0218       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.14        |
    |    explained_variance   | 0.763        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 49.7         |
    |    n_updates            | 68           |
    |    policy_gradient_loss | -0.00221     |
    |    value_loss           | 164          |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 647         |
    |    ep_rew_mean          | 36          |
    | time/                   |             |
    |    fps                  | 1098        |
    |    iterations           | 19          |
    |    time_elapsed         | 283         |
    |    total_timesteps      | 311296      |
    | train/                  |             |
    |    approx_kl            | 0.003759494 |
    |    clip_fraction        | 0.028       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.14       |
    |    explained_variance   | 0.774       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 62          |
    |    n_updates            | 72          |
    |    policy_gradient_loss | -0.00121    |
    |    value_loss           | 168         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 647         |
    |    ep_rew_mean          | 36          |
    | time/                   |             |
    |    fps                  | 1098        |
    |    iterations           | 19          |
    |    time_elapsed         | 283         |
    |    total_timesteps      | 311296      |
    | train/                  |             |
    |    approx_kl            | 0.003759494 |
    |    clip_fraction        | 0.028       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.14       |
    |    explained_variance   | 0.774       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 62          |
    |    n_updates            | 72          |
    |    policy_gradient_loss | -0.00121    |
    |    value_loss           | 168         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 665         |
    |    ep_rew_mean          | 38          |
    | time/                   |             |
    |    fps                  | 1062        |
    |    iterations           | 20          |
    |    time_elapsed         | 308         |
    |    total_timesteps      | 327680      |
    | train/                  |             |
    |    approx_kl            | 0.004863321 |
    |    clip_fraction        | 0.0295      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.11       |
    |    explained_variance   | 0.772       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 53.4        |
    |    n_updates            | 76          |
    |    policy_gradient_loss | -0.00301    |
    |    value_loss           | 204         |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 665         |
    |    ep_rew_mean          | 38          |
    | time/                   |             |
    |    fps                  | 1062        |
    |    iterations           | 20          |
    |    time_elapsed         | 308         |
    |    total_timesteps      | 327680      |
    | train/                  |             |
    |    approx_kl            | 0.004863321 |
    |    clip_fraction        | 0.0295      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.11       |
    |    explained_variance   | 0.772       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 53.4        |
    |    n_updates            | 76          |
    |    policy_gradient_loss | -0.00301    |
    |    value_loss           | 204         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 729          |
    |    ep_rew_mean          | 41.6         |
    | time/                   |              |
    |    fps                  | 1024         |
    |    iterations           | 21           |
    |    time_elapsed         | 335          |
    |    total_timesteps      | 344064       |
    | train/                  |              |
    |    approx_kl            | 0.0069686538 |
    |    clip_fraction        | 0.0529       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.11        |
    |    explained_variance   | 0.834        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 79.1         |
    |    n_updates            | 80           |
    |    policy_gradient_loss | -0.00263     |
    |    value_loss           | 173          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 729          |
    |    ep_rew_mean          | 41.6         |
    | time/                   |              |
    |    fps                  | 1024         |
    |    iterations           | 21           |
    |    time_elapsed         | 335          |
    |    total_timesteps      | 344064       |
    | train/                  |              |
    |    approx_kl            | 0.0069686538 |
    |    clip_fraction        | 0.0529       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.11        |
    |    explained_variance   | 0.834        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 79.1         |
    |    n_updates            | 80           |
    |    policy_gradient_loss | -0.00263     |
    |    value_loss           | 173          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 753          |
    |    ep_rew_mean          | 42.2         |
    | time/                   |              |
    |    fps                  | 995          |
    |    iterations           | 22           |
    |    time_elapsed         | 362          |
    |    total_timesteps      | 360448       |
    | train/                  |              |
    |    approx_kl            | 0.0051742056 |
    |    clip_fraction        | 0.0222       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.14        |
    |    explained_variance   | 0.821        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 48.1         |
    |    n_updates            | 84           |
    |    policy_gradient_loss | -0.00037     |
    |    value_loss           | 149          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 753          |
    |    ep_rew_mean          | 42.2         |
    | time/                   |              |
    |    fps                  | 995          |
    |    iterations           | 22           |
    |    time_elapsed         | 362          |
    |    total_timesteps      | 360448       |
    | train/                  |              |
    |    approx_kl            | 0.0051742056 |
    |    clip_fraction        | 0.0222       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.14        |
    |    explained_variance   | 0.821        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 48.1         |
    |    n_updates            | 84           |
    |    policy_gradient_loss | -0.00037     |
    |    value_loss           | 149          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 769          |
    |    ep_rew_mean          | 43.1         |
    | time/                   |              |
    |    fps                  | 968          |
    |    iterations           | 23           |
    |    time_elapsed         | 388          |
    |    total_timesteps      | 376832       |
    | train/                  |              |
    |    approx_kl            | 0.0051718093 |
    |    clip_fraction        | 0.0524       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.11        |
    |    explained_variance   | 0.877        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 99.4         |
    |    n_updates            | 88           |
    |    policy_gradient_loss | -0.00244     |
    |    value_loss           | 95.1         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 769          |
    |    ep_rew_mean          | 43.1         |
    | time/                   |              |
    |    fps                  | 968          |
    |    iterations           | 23           |
    |    time_elapsed         | 388          |
    |    total_timesteps      | 376832       |
    | train/                  |              |
    |    approx_kl            | 0.0051718093 |
    |    clip_fraction        | 0.0524       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.11        |
    |    explained_variance   | 0.877        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 99.4         |
    |    n_updates            | 88           |
    |    policy_gradient_loss | -0.00244     |
    |    value_loss           | 95.1         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 827          |
    |    ep_rew_mean          | 56           |
    | time/                   |              |
    |    fps                  | 945          |
    |    iterations           | 24           |
    |    time_elapsed         | 415          |
    |    total_timesteps      | 393216       |
    | train/                  |              |
    |    approx_kl            | 0.0035088474 |
    |    clip_fraction        | 0.0387       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.1         |
    |    explained_variance   | 0.871        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 32.8         |
    |    n_updates            | 92           |
    |    policy_gradient_loss | -0.00145     |
    |    value_loss           | 108          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 827          |
    |    ep_rew_mean          | 56           |
    | time/                   |              |
    |    fps                  | 945          |
    |    iterations           | 24           |
    |    time_elapsed         | 415          |
    |    total_timesteps      | 393216       |
    | train/                  |              |
    |    approx_kl            | 0.0035088474 |
    |    clip_fraction        | 0.0387       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.1         |
    |    explained_variance   | 0.871        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 32.8         |
    |    n_updates            | 92           |
    |    policy_gradient_loss | -0.00145     |
    |    value_loss           | 108          |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 867         |
    |    ep_rew_mean          | 66.9        |
    | time/                   |             |
    |    fps                  | 922         |
    |    iterations           | 25          |
    |    time_elapsed         | 443         |
    |    total_timesteps      | 409600      |
    | train/                  |             |
    |    approx_kl            | 0.005614726 |
    |    clip_fraction        | 0.0374      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.09       |
    |    explained_variance   | 0.916       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 34.2        |
    |    n_updates            | 96          |
    |    policy_gradient_loss | -0.00196    |
    |    value_loss           | 63.9        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 867         |
    |    ep_rew_mean          | 66.9        |
    | time/                   |             |
    |    fps                  | 922         |
    |    iterations           | 25          |
    |    time_elapsed         | 443         |
    |    total_timesteps      | 409600      |
    | train/                  |             |
    |    approx_kl            | 0.005614726 |
    |    clip_fraction        | 0.0374      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.09       |
    |    explained_variance   | 0.916       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 34.2        |
    |    n_updates            | 96          |
    |    policy_gradient_loss | -0.00196    |
    |    value_loss           | 63.9        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 890          |
    |    ep_rew_mean          | 77           |
    | time/                   |              |
    |    fps                  | 900          |
    |    iterations           | 26           |
    |    time_elapsed         | 472          |
    |    total_timesteps      | 425984       |
    | train/                  |              |
    |    approx_kl            | 0.0052196365 |
    |    clip_fraction        | 0.0456       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.05        |
    |    explained_variance   | 0.957        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 15.9         |
    |    n_updates            | 100          |
    |    policy_gradient_loss | -0.0014      |
    |    value_loss           | 26.3         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 890          |
    |    ep_rew_mean          | 77           |
    | time/                   |              |
    |    fps                  | 900          |
    |    iterations           | 26           |
    |    time_elapsed         | 472          |
    |    total_timesteps      | 425984       |
    | train/                  |              |
    |    approx_kl            | 0.0052196365 |
    |    clip_fraction        | 0.0456       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.05        |
    |    explained_variance   | 0.957        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 15.9         |
    |    n_updates            | 100          |
    |    policy_gradient_loss | -0.0014      |
    |    value_loss           | 26.3         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 871          |
    |    ep_rew_mean          | 77.2         |
    | time/                   |              |
    |    fps                  | 887          |
    |    iterations           | 27           |
    |    time_elapsed         | 498          |
    |    total_timesteps      | 442368       |
    | train/                  |              |
    |    approx_kl            | 0.0047236443 |
    |    clip_fraction        | 0.0314       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.09        |
    |    explained_variance   | 0.957        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 9.14         |
    |    n_updates            | 104          |
    |    policy_gradient_loss | -0.000811    |
    |    value_loss           | 29.9         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 871          |
    |    ep_rew_mean          | 77.2         |
    | time/                   |              |
    |    fps                  | 887          |
    |    iterations           | 27           |
    |    time_elapsed         | 498          |
    |    total_timesteps      | 442368       |
    | train/                  |              |
    |    approx_kl            | 0.0047236443 |
    |    clip_fraction        | 0.0314       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.09        |
    |    explained_variance   | 0.957        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 9.14         |
    |    n_updates            | 104          |
    |    policy_gradient_loss | -0.000811    |
    |    value_loss           | 29.9         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 883          |
    |    ep_rew_mean          | 87           |
    | time/                   |              |
    |    fps                  | 876          |
    |    iterations           | 28           |
    |    time_elapsed         | 523          |
    |    total_timesteps      | 458752       |
    | train/                  |              |
    |    approx_kl            | 0.0028928886 |
    |    clip_fraction        | 0.0168       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.02        |
    |    explained_variance   | 0.926        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 11           |
    |    n_updates            | 108          |
    |    policy_gradient_loss | -0.000931    |
    |    value_loss           | 65.2         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 883          |
    |    ep_rew_mean          | 87           |
    | time/                   |              |
    |    fps                  | 876          |
    |    iterations           | 28           |
    |    time_elapsed         | 523          |
    |    total_timesteps      | 458752       |
    | train/                  |              |
    |    approx_kl            | 0.0028928886 |
    |    clip_fraction        | 0.0168       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.02        |
    |    explained_variance   | 0.926        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 11           |
    |    n_updates            | 108          |
    |    policy_gradient_loss | -0.000931    |
    |    value_loss           | 65.2         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 917         |
    |    ep_rew_mean          | 95.2        |
    | time/                   |             |
    |    fps                  | 863         |
    |    iterations           | 29          |
    |    time_elapsed         | 550         |
    |    total_timesteps      | 475136      |
    | train/                  |             |
    |    approx_kl            | 0.004920424 |
    |    clip_fraction        | 0.0379      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.04       |
    |    explained_variance   | 0.955       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 6.51        |
    |    n_updates            | 112         |
    |    policy_gradient_loss | -0.000947   |
    |    value_loss           | 38.8        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 917         |
    |    ep_rew_mean          | 95.2        |
    | time/                   |             |
    |    fps                  | 863         |
    |    iterations           | 29          |
    |    time_elapsed         | 550         |
    |    total_timesteps      | 475136      |
    | train/                  |             |
    |    approx_kl            | 0.004920424 |
    |    clip_fraction        | 0.0379      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.04       |
    |    explained_variance   | 0.955       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 6.51        |
    |    n_updates            | 112         |
    |    policy_gradient_loss | -0.000947   |
    |    value_loss           | 38.8        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 923         |
    |    ep_rew_mean          | 99.6        |
    | time/                   |             |
    |    fps                  | 850         |
    |    iterations           | 30          |
    |    time_elapsed         | 578         |
    |    total_timesteps      | 491520      |
    | train/                  |             |
    |    approx_kl            | 0.004317927 |
    |    clip_fraction        | 0.0563      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.05       |
    |    explained_variance   | 0.984       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 6.16        |
    |    n_updates            | 116         |
    |    policy_gradient_loss | -0.0012     |
    |    value_loss           | 11.6        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 923         |
    |    ep_rew_mean          | 99.6        |
    | time/                   |             |
    |    fps                  | 850         |
    |    iterations           | 30          |
    |    time_elapsed         | 578         |
    |    total_timesteps      | 491520      |
    | train/                  |             |
    |    approx_kl            | 0.004317927 |
    |    clip_fraction        | 0.0563      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.05       |
    |    explained_variance   | 0.984       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 6.16        |
    |    n_updates            | 116         |
    |    policy_gradient_loss | -0.0012     |
    |    value_loss           | 11.6        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 923         |
    |    ep_rew_mean          | 103         |
    | time/                   |             |
    |    fps                  | 837         |
    |    iterations           | 31          |
    |    time_elapsed         | 606         |
    |    total_timesteps      | 507904      |
    | train/                  |             |
    |    approx_kl            | 0.005127452 |
    |    clip_fraction        | 0.045       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.04       |
    |    explained_variance   | 0.953       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 42.9        |
    |    n_updates            | 120         |
    |    policy_gradient_loss | -0.00103    |
    |    value_loss           | 37.3        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 923         |
    |    ep_rew_mean          | 103         |
    | time/                   |             |
    |    fps                  | 837         |
    |    iterations           | 31          |
    |    time_elapsed         | 606         |
    |    total_timesteps      | 507904      |
    | train/                  |             |
    |    approx_kl            | 0.005127452 |
    |    clip_fraction        | 0.045       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.04       |
    |    explained_variance   | 0.953       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 42.9        |
    |    n_updates            | 120         |
    |    policy_gradient_loss | -0.00103    |
    |    value_loss           | 37.3        |
    -----------------------------------------
    ----------------------------------------
    | rollout/                |            |
    |    ep_len_mean          | 923        |
    |    ep_rew_mean          | 105        |
    | time/                   |            |
    |    fps                  | 827        |
    |    iterations           | 32         |
    |    time_elapsed         | 633        |
    |    total_timesteps      | 524288     |
    | train/                  |            |
    |    approx_kl            | 0.00507261 |
    |    clip_fraction        | 0.0472     |
    |    clip_range           | 0.2        |
    |    entropy_loss         | -1.05      |
    |    explained_variance   | 0.99       |
    |    learning_rate        | 0.0003     |
    |    loss                 | 1.72       |
    |    n_updates            | 124        |
    |    policy_gradient_loss | -0.000984  |
    |    value_loss           | 7.08       |
    ----------------------------------------
    ----------------------------------------
    | rollout/                |            |
    |    ep_len_mean          | 923        |
    |    ep_rew_mean          | 105        |
    | time/                   |            |
    |    fps                  | 827        |
    |    iterations           | 32         |
    |    time_elapsed         | 633        |
    |    total_timesteps      | 524288     |
    | train/                  |            |
    |    approx_kl            | 0.00507261 |
    |    clip_fraction        | 0.0472     |
    |    clip_range           | 0.2        |
    |    entropy_loss         | -1.05      |
    |    explained_variance   | 0.99       |
    |    learning_rate        | 0.0003     |
    |    loss                 | 1.72       |
    |    n_updates            | 124        |
    |    policy_gradient_loss | -0.000984  |
    |    value_loss           | 7.08       |
    ----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 952          |
    |    ep_rew_mean          | 115          |
    | time/                   |              |
    |    fps                  | 819          |
    |    iterations           | 33           |
    |    time_elapsed         | 659          |
    |    total_timesteps      | 540672       |
    | train/                  |              |
    |    approx_kl            | 0.0039739693 |
    |    clip_fraction        | 0.0202       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.07        |
    |    explained_variance   | 0.959        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 46.9         |
    |    n_updates            | 128          |
    |    policy_gradient_loss | -0.000317    |
    |    value_loss           | 41.1         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 952          |
    |    ep_rew_mean          | 115          |
    | time/                   |              |
    |    fps                  | 819          |
    |    iterations           | 33           |
    |    time_elapsed         | 659          |
    |    total_timesteps      | 540672       |
    | train/                  |              |
    |    approx_kl            | 0.0039739693 |
    |    clip_fraction        | 0.0202       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.07        |
    |    explained_variance   | 0.959        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 46.9         |
    |    n_updates            | 128          |
    |    policy_gradient_loss | -0.000317    |
    |    value_loss           | 41.1         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 945         |
    |    ep_rew_mean          | 115         |
    | time/                   |             |
    |    fps                  | 813         |
    |    iterations           | 34          |
    |    time_elapsed         | 685         |
    |    total_timesteps      | 557056      |
    | train/                  |             |
    |    approx_kl            | 0.008476433 |
    |    clip_fraction        | 0.0343      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.03       |
    |    explained_variance   | 0.983       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 1.81        |
    |    n_updates            | 132         |
    |    policy_gradient_loss | -0.0015     |
    |    value_loss           | 14.5        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 945         |
    |    ep_rew_mean          | 115         |
    | time/                   |             |
    |    fps                  | 813         |
    |    iterations           | 34          |
    |    time_elapsed         | 685         |
    |    total_timesteps      | 557056      |
    | train/                  |             |
    |    approx_kl            | 0.008476433 |
    |    clip_fraction        | 0.0343      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -1.03       |
    |    explained_variance   | 0.983       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 1.81        |
    |    n_updates            | 132         |
    |    policy_gradient_loss | -0.0015     |
    |    value_loss           | 14.5        |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 945          |
    |    ep_rew_mean          | 117          |
    | time/                   |              |
    |    fps                  | 804          |
    |    iterations           | 35           |
    |    time_elapsed         | 712          |
    |    total_timesteps      | 573440       |
    | train/                  |              |
    |    approx_kl            | 0.0035138065 |
    |    clip_fraction        | 0.0216       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.985       |
    |    explained_variance   | 0.967        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 8.1          |
    |    n_updates            | 136          |
    |    policy_gradient_loss | -0.00138     |
    |    value_loss           | 38.3         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 951          |
    |    ep_rew_mean          | 120          |
    | time/                   |              |
    |    fps                  | 797          |
    |    iterations           | 36           |
    |    time_elapsed         | 739          |
    |    total_timesteps      | 589824       |
    | train/                  |              |
    |    approx_kl            | 0.0062673613 |
    |    clip_fraction        | 0.0398       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.01        |
    |    explained_variance   | 0.985        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 19.8         |
    |    n_updates            | 140          |
    |    policy_gradient_loss | -0.00137     |
    |    value_loss           | 16.2         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 945          |
    |    ep_rew_mean          | 117          |
    | time/                   |              |
    |    fps                  | 804          |
    |    iterations           | 35           |
    |    time_elapsed         | 712          |
    |    total_timesteps      | 573440       |
    | train/                  |              |
    |    approx_kl            | 0.0035138065 |
    |    clip_fraction        | 0.0216       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.985       |
    |    explained_variance   | 0.967        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 8.1          |
    |    n_updates            | 136          |
    |    policy_gradient_loss | -0.00138     |
    |    value_loss           | 38.3         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 951          |
    |    ep_rew_mean          | 120          |
    | time/                   |              |
    |    fps                  | 797          |
    |    iterations           | 36           |
    |    time_elapsed         | 739          |
    |    total_timesteps      | 589824       |
    | train/                  |              |
    |    approx_kl            | 0.0062673613 |
    |    clip_fraction        | 0.0398       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -1.01        |
    |    explained_variance   | 0.985        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 19.8         |
    |    n_updates            | 140          |
    |    policy_gradient_loss | -0.00137     |
    |    value_loss           | 16.2         |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 945         |
    |    ep_rew_mean          | 122         |
    | time/                   |             |
    |    fps                  | 792         |
    |    iterations           | 37          |
    |    time_elapsed         | 765         |
    |    total_timesteps      | 606208      |
    | train/                  |             |
    |    approx_kl            | 0.005545972 |
    |    clip_fraction        | 0.0493      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.969      |
    |    explained_variance   | 0.994       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 3.92        |
    |    n_updates            | 144         |
    |    policy_gradient_loss | -0.00101    |
    |    value_loss           | 5.39        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 945         |
    |    ep_rew_mean          | 122         |
    | time/                   |             |
    |    fps                  | 792         |
    |    iterations           | 37          |
    |    time_elapsed         | 765         |
    |    total_timesteps      | 606208      |
    | train/                  |             |
    |    approx_kl            | 0.005545972 |
    |    clip_fraction        | 0.0493      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.969      |
    |    explained_variance   | 0.994       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 3.92        |
    |    n_updates            | 144         |
    |    policy_gradient_loss | -0.00101    |
    |    value_loss           | 5.39        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 959         |
    |    ep_rew_mean          | 127         |
    | time/                   |             |
    |    fps                  | 786         |
    |    iterations           | 38          |
    |    time_elapsed         | 791         |
    |    total_timesteps      | 622592      |
    | train/                  |             |
    |    approx_kl            | 0.004641909 |
    |    clip_fraction        | 0.0414      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.978      |
    |    explained_variance   | 0.985       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 37          |
    |    n_updates            | 148         |
    |    policy_gradient_loss | -0.000757   |
    |    value_loss           | 16.8        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 959         |
    |    ep_rew_mean          | 127         |
    | time/                   |             |
    |    fps                  | 786         |
    |    iterations           | 38          |
    |    time_elapsed         | 791         |
    |    total_timesteps      | 622592      |
    | train/                  |             |
    |    approx_kl            | 0.004641909 |
    |    clip_fraction        | 0.0414      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.978      |
    |    explained_variance   | 0.985       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 37          |
    |    n_updates            | 148         |
    |    policy_gradient_loss | -0.000757   |
    |    value_loss           | 16.8        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 960         |
    |    ep_rew_mean          | 129         |
    | time/                   |             |
    |    fps                  | 780         |
    |    iterations           | 39          |
    |    time_elapsed         | 818         |
    |    total_timesteps      | 638976      |
    | train/                  |             |
    |    approx_kl            | 0.004064006 |
    |    clip_fraction        | 0.0393      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.962      |
    |    explained_variance   | 0.995       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 2.36        |
    |    n_updates            | 152         |
    |    policy_gradient_loss | -2.79e-05   |
    |    value_loss           | 5.42        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 960         |
    |    ep_rew_mean          | 129         |
    | time/                   |             |
    |    fps                  | 780         |
    |    iterations           | 39          |
    |    time_elapsed         | 818         |
    |    total_timesteps      | 638976      |
    | train/                  |             |
    |    approx_kl            | 0.004064006 |
    |    clip_fraction        | 0.0393      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.962      |
    |    explained_variance   | 0.995       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 2.36        |
    |    n_updates            | 152         |
    |    policy_gradient_loss | -2.79e-05   |
    |    value_loss           | 5.42        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 968         |
    |    ep_rew_mean          | 132         |
    | time/                   |             |
    |    fps                  | 775         |
    |    iterations           | 40          |
    |    time_elapsed         | 845         |
    |    total_timesteps      | 655360      |
    | train/                  |             |
    |    approx_kl            | 0.004771296 |
    |    clip_fraction        | 0.0428      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.937      |
    |    explained_variance   | 0.989       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 9.37        |
    |    n_updates            | 156         |
    |    policy_gradient_loss | -0.00153    |
    |    value_loss           | 14.5        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 968         |
    |    ep_rew_mean          | 132         |
    | time/                   |             |
    |    fps                  | 775         |
    |    iterations           | 40          |
    |    time_elapsed         | 845         |
    |    total_timesteps      | 655360      |
    | train/                  |             |
    |    approx_kl            | 0.004771296 |
    |    clip_fraction        | 0.0428      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.937      |
    |    explained_variance   | 0.989       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 9.37        |
    |    n_updates            | 156         |
    |    policy_gradient_loss | -0.00153    |
    |    value_loss           | 14.5        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 974         |
    |    ep_rew_mean          | 136         |
    | time/                   |             |
    |    fps                  | 772         |
    |    iterations           | 41          |
    |    time_elapsed         | 870         |
    |    total_timesteps      | 671744      |
    | train/                  |             |
    |    approx_kl            | 0.004322427 |
    |    clip_fraction        | 0.0441      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.918      |
    |    explained_variance   | 0.987       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 1.3         |
    |    n_updates            | 160         |
    |    policy_gradient_loss | -0.00052    |
    |    value_loss           | 14.3        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 974         |
    |    ep_rew_mean          | 136         |
    | time/                   |             |
    |    fps                  | 772         |
    |    iterations           | 41          |
    |    time_elapsed         | 870         |
    |    total_timesteps      | 671744      |
    | train/                  |             |
    |    approx_kl            | 0.004322427 |
    |    clip_fraction        | 0.0441      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.918      |
    |    explained_variance   | 0.987       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 1.3         |
    |    n_updates            | 160         |
    |    policy_gradient_loss | -0.00052    |
    |    value_loss           | 14.3        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 909         |
    |    ep_rew_mean          | 152         |
    | time/                   |             |
    |    fps                  | 757         |
    |    iterations           | 50          |
    |    time_elapsed         | 1081        |
    |    total_timesteps      | 819200      |
    | train/                  |             |
    |    approx_kl            | 0.004018886 |
    |    clip_fraction        | 0.0451      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.762      |
    |    explained_variance   | 0.954       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 45.9        |
    |    n_updates            | 196         |
    |    policy_gradient_loss | -0.00018    |
    |    value_loss           | 70.7        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 909         |
    |    ep_rew_mean          | 153         |
    | time/                   |             |
    |    fps                  | 757         |
    |    iterations           | 51          |
    |    time_elapsed         | 1103        |
    |    total_timesteps      | 835584      |
    | train/                  |             |
    |    approx_kl            | 0.003438447 |
    |    clip_fraction        | 0.0424      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.773      |
    |    explained_variance   | 0.956       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 3.17        |
    |    n_updates            | 200         |
    |    policy_gradient_loss | -0.000693   |
    |    value_loss           | 73          |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 909         |
    |    ep_rew_mean          | 152         |
    | time/                   |             |
    |    fps                  | 757         |
    |    iterations           | 50          |
    |    time_elapsed         | 1081        |
    |    total_timesteps      | 819200      |
    | train/                  |             |
    |    approx_kl            | 0.004018886 |
    |    clip_fraction        | 0.0451      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.762      |
    |    explained_variance   | 0.954       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 45.9        |
    |    n_updates            | 196         |
    |    policy_gradient_loss | -0.00018    |
    |    value_loss           | 70.7        |
    -----------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 909         |
    |    ep_rew_mean          | 153         |
    | time/                   |             |
    |    fps                  | 757         |
    |    iterations           | 51          |
    |    time_elapsed         | 1103        |
    |    total_timesteps      | 835584      |
    | train/                  |             |
    |    approx_kl            | 0.003438447 |
    |    clip_fraction        | 0.0424      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.773      |
    |    explained_variance   | 0.956       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 3.17        |
    |    n_updates            | 200         |
    |    policy_gradient_loss | -0.000693   |
    |    value_loss           | 73          |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 875          |
    |    ep_rew_mean          | 152          |
    | time/                   |              |
    |    fps                  | 757          |
    |    iterations           | 52           |
    |    time_elapsed         | 1124         |
    |    total_timesteps      | 851968       |
    | train/                  |              |
    |    approx_kl            | 0.0040834136 |
    |    clip_fraction        | 0.0487       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.768       |
    |    explained_variance   | 0.979        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 36.4         |
    |    n_updates            | 204          |
    |    policy_gradient_loss | 0.0014       |
    |    value_loss           | 32.7         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 875          |
    |    ep_rew_mean          | 150          |
    | time/                   |              |
    |    fps                  | 757          |
    |    iterations           | 53           |
    |    time_elapsed         | 1146         |
    |    total_timesteps      | 868352       |
    | train/                  |              |
    |    approx_kl            | 0.0032271326 |
    |    clip_fraction        | 0.024        |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.769       |
    |    explained_variance   | 0.947        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 57.7         |
    |    n_updates            | 208          |
    |    policy_gradient_loss | -0.000804    |
    |    value_loss           | 86.7         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 847          |
    |    ep_rew_mean          | 145          |
    | time/                   |              |
    |    fps                  | 757          |
    |    iterations           | 54           |
    |    time_elapsed         | 1167         |
    |    total_timesteps      | 884736       |
    | train/                  |              |
    |    approx_kl            | 0.0037451896 |
    |    clip_fraction        | 0.0495       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.769       |
    |    explained_variance   | 0.975        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 4.82         |
    |    n_updates            | 212          |
    |    policy_gradient_loss | -7.53e-05    |
    |    value_loss           | 38.3         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 834          |
    |    ep_rew_mean          | 143          |
    | time/                   |              |
    |    fps                  | 757          |
    |    iterations           | 55           |
    |    time_elapsed         | 1189         |
    |    total_timesteps      | 901120       |
    | train/                  |              |
    |    approx_kl            | 0.0040088133 |
    |    clip_fraction        | 0.0387       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.757       |
    |    explained_variance   | 0.936        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 39.2         |
    |    n_updates            | 216          |
    |    policy_gradient_loss | -0.000898    |
    |    value_loss           | 99           |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 831         |
    |    ep_rew_mean          | 153         |
    | time/                   |             |
    |    fps                  | 757         |
    |    iterations           | 56          |
    |    time_elapsed         | 1211        |
    |    total_timesteps      | 917504      |
    | train/                  |             |
    |    approx_kl            | 0.004583096 |
    |    clip_fraction        | 0.0501      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.771      |
    |    explained_variance   | 0.926       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 8.63        |
    |    n_updates            | 220         |
    |    policy_gradient_loss | -0.00108    |
    |    value_loss           | 116         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 818          |
    |    ep_rew_mean          | 165          |
    | time/                   |              |
    |    fps                  | 758          |
    |    iterations           | 57           |
    |    time_elapsed         | 1231         |
    |    total_timesteps      | 933888       |
    | train/                  |              |
    |    approx_kl            | 0.0038438963 |
    |    clip_fraction        | 0.0513       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.711       |
    |    explained_variance   | 0.907        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 24.7         |
    |    n_updates            | 224          |
    |    policy_gradient_loss | -0.000768    |
    |    value_loss           | 129          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 763          |
    |    ep_rew_mean          | 183          |
    | time/                   |              |
    |    fps                  | 759          |
    |    iterations           | 58           |
    |    time_elapsed         | 1250         |
    |    total_timesteps      | 950272       |
    | train/                  |              |
    |    approx_kl            | 0.0041993475 |
    |    clip_fraction        | 0.0591       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.698       |
    |    explained_variance   | 0.885        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 60.9         |
    |    n_updates            | 228          |
    |    policy_gradient_loss | -0.00239     |
    |    value_loss           | 195          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 692          |
    |    ep_rew_mean          | 209          |
    | time/                   |              |
    |    fps                  | 761          |
    |    iterations           | 59           |
    |    time_elapsed         | 1269         |
    |    total_timesteps      | 966656       |
    | train/                  |              |
    |    approx_kl            | 0.0052020433 |
    |    clip_fraction        | 0.05         |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.676       |
    |    explained_variance   | 0.847        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 89           |
    |    n_updates            | 232          |
    |    policy_gradient_loss | -0.00283     |
    |    value_loss           | 225          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 875          |
    |    ep_rew_mean          | 152          |
    | time/                   |              |
    |    fps                  | 757          |
    |    iterations           | 52           |
    |    time_elapsed         | 1124         |
    |    total_timesteps      | 851968       |
    | train/                  |              |
    |    approx_kl            | 0.0040834136 |
    |    clip_fraction        | 0.0487       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.768       |
    |    explained_variance   | 0.979        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 36.4         |
    |    n_updates            | 204          |
    |    policy_gradient_loss | 0.0014       |
    |    value_loss           | 32.7         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 875          |
    |    ep_rew_mean          | 150          |
    | time/                   |              |
    |    fps                  | 757          |
    |    iterations           | 53           |
    |    time_elapsed         | 1146         |
    |    total_timesteps      | 868352       |
    | train/                  |              |
    |    approx_kl            | 0.0032271326 |
    |    clip_fraction        | 0.024        |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.769       |
    |    explained_variance   | 0.947        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 57.7         |
    |    n_updates            | 208          |
    |    policy_gradient_loss | -0.000804    |
    |    value_loss           | 86.7         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 847          |
    |    ep_rew_mean          | 145          |
    | time/                   |              |
    |    fps                  | 757          |
    |    iterations           | 54           |
    |    time_elapsed         | 1167         |
    |    total_timesteps      | 884736       |
    | train/                  |              |
    |    approx_kl            | 0.0037451896 |
    |    clip_fraction        | 0.0495       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.769       |
    |    explained_variance   | 0.975        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 4.82         |
    |    n_updates            | 212          |
    |    policy_gradient_loss | -7.53e-05    |
    |    value_loss           | 38.3         |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 834          |
    |    ep_rew_mean          | 143          |
    | time/                   |              |
    |    fps                  | 757          |
    |    iterations           | 55           |
    |    time_elapsed         | 1189         |
    |    total_timesteps      | 901120       |
    | train/                  |              |
    |    approx_kl            | 0.0040088133 |
    |    clip_fraction        | 0.0387       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.757       |
    |    explained_variance   | 0.936        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 39.2         |
    |    n_updates            | 216          |
    |    policy_gradient_loss | -0.000898    |
    |    value_loss           | 99           |
    ------------------------------------------
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 831         |
    |    ep_rew_mean          | 153         |
    | time/                   |             |
    |    fps                  | 757         |
    |    iterations           | 56          |
    |    time_elapsed         | 1211        |
    |    total_timesteps      | 917504      |
    | train/                  |             |
    |    approx_kl            | 0.004583096 |
    |    clip_fraction        | 0.0501      |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -0.771      |
    |    explained_variance   | 0.926       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 8.63        |
    |    n_updates            | 220         |
    |    policy_gradient_loss | -0.00108    |
    |    value_loss           | 116         |
    -----------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 818          |
    |    ep_rew_mean          | 165          |
    | time/                   |              |
    |    fps                  | 758          |
    |    iterations           | 57           |
    |    time_elapsed         | 1231         |
    |    total_timesteps      | 933888       |
    | train/                  |              |
    |    approx_kl            | 0.0038438963 |
    |    clip_fraction        | 0.0513       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.711       |
    |    explained_variance   | 0.907        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 24.7         |
    |    n_updates            | 224          |
    |    policy_gradient_loss | -0.000768    |
    |    value_loss           | 129          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 763          |
    |    ep_rew_mean          | 183          |
    | time/                   |              |
    |    fps                  | 759          |
    |    iterations           | 58           |
    |    time_elapsed         | 1250         |
    |    total_timesteps      | 950272       |
    | train/                  |              |
    |    approx_kl            | 0.0041993475 |
    |    clip_fraction        | 0.0591       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.698       |
    |    explained_variance   | 0.885        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 60.9         |
    |    n_updates            | 228          |
    |    policy_gradient_loss | -0.00239     |
    |    value_loss           | 195          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 692          |
    |    ep_rew_mean          | 209          |
    | time/                   |              |
    |    fps                  | 761          |
    |    iterations           | 59           |
    |    time_elapsed         | 1269         |
    |    total_timesteps      | 966656       |
    | train/                  |              |
    |    approx_kl            | 0.0052020433 |
    |    clip_fraction        | 0.05         |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.676       |
    |    explained_variance   | 0.847        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 89           |
    |    n_updates            | 232          |
    |    policy_gradient_loss | -0.00283     |
    |    value_loss           | 225          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 589          |
    |    ep_rew_mean          | 232          |
    | time/                   |              |
    |    fps                  | 763          |
    |    iterations           | 60           |
    |    time_elapsed         | 1287         |
    |    total_timesteps      | 983040       |
    | train/                  |              |
    |    approx_kl            | 0.0063320864 |
    |    clip_fraction        | 0.0811       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.656       |
    |    explained_variance   | 0.81         |
    |    learning_rate        | 0.0003       |
    |    loss                 | 187          |
    |    n_updates            | 236          |
    |    policy_gradient_loss | -0.00376     |
    |    value_loss           | 285          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 459          |
    |    ep_rew_mean          | 251          |
    | time/                   |              |
    |    fps                  | 767          |
    |    iterations           | 61           |
    |    time_elapsed         | 1302         |
    |    total_timesteps      | 999424       |
    | train/                  |              |
    |    approx_kl            | 0.0063625546 |
    |    clip_fraction        | 0.0781       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.674       |
    |    explained_variance   | 0.799        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 79           |
    |    n_updates            | 240          |
    |    policy_gradient_loss | -0.00391     |
    |    value_loss           | 240          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 351          |
    |    ep_rew_mean          | 252          |
    | time/                   |              |
    |    fps                  | 771          |
    |    iterations           | 62           |
    |    time_elapsed         | 1316         |
    |    total_timesteps      | 1015808      |
    | train/                  |              |
    |    approx_kl            | 0.0069095753 |
    |    clip_fraction        | 0.0707       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.741       |
    |    explained_variance   | 0.734        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 165          |
    |    n_updates            | 244          |
    |    policy_gradient_loss | -0.00149     |
    |    value_loss           | 207          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 589          |
    |    ep_rew_mean          | 232          |
    | time/                   |              |
    |    fps                  | 763          |
    |    iterations           | 60           |
    |    time_elapsed         | 1287         |
    |    total_timesteps      | 983040       |
    | train/                  |              |
    |    approx_kl            | 0.0063320864 |
    |    clip_fraction        | 0.0811       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.656       |
    |    explained_variance   | 0.81         |
    |    learning_rate        | 0.0003       |
    |    loss                 | 187          |
    |    n_updates            | 236          |
    |    policy_gradient_loss | -0.00376     |
    |    value_loss           | 285          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 459          |
    |    ep_rew_mean          | 251          |
    | time/                   |              |
    |    fps                  | 767          |
    |    iterations           | 61           |
    |    time_elapsed         | 1302         |
    |    total_timesteps      | 999424       |
    | train/                  |              |
    |    approx_kl            | 0.0063625546 |
    |    clip_fraction        | 0.0781       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.674       |
    |    explained_variance   | 0.799        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 79           |
    |    n_updates            | 240          |
    |    policy_gradient_loss | -0.00391     |
    |    value_loss           | 240          |
    ------------------------------------------
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 351          |
    |    ep_rew_mean          | 252          |
    | time/                   |              |
    |    fps                  | 771          |
    |    iterations           | 62           |
    |    time_elapsed         | 1316         |
    |    total_timesteps      | 1015808      |
    | train/                  |              |
    |    approx_kl            | 0.0069095753 |
    |    clip_fraction        | 0.0707       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.741       |
    |    explained_variance   | 0.734        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 165          |
    |    n_updates            | 244          |
    |    policy_gradient_loss | -0.00149     |
    |    value_loss           | 207          |
    ------------------------------------------
    


```python
eval_env = Monitor(gym.make('LunarLander-v3', render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f'Mean reward={mean_reward:.2f} +/- {std_reward}')
```

    Mean reward=264.59 +/- 17.297802215705374
    
