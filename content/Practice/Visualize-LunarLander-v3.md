```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub
test_env = gym.make("LunarLander-v3", render_mode="human")
model = PPO(
    policy='MlpPolicy',
    env=test_env,
    n_steps=1024,
    batch_size=128,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1
)

obs, _ = test_env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, _ = test_env.reset()

test_env.close()
```

    Using cpu device
    Wrapping the env with a `Monitor` wrapper
    Wrapping the env in a DummyVecEnv.
    


```python
test_env = gym.make('LunarLander-v3', render_mode='human')
model = PPO.load('ppo-LunarLander-v3.zip')
obs, info = test_env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, _ = test_env.reset()
test_env.close()
```
