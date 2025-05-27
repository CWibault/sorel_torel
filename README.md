Two development folders exist due to different compatibility requirements between Brax and D4RL environments.

# SOReL

SOReL does not automatically save each policy trained during hyperparameter sweeps, as the hyperparameter space is large.

Running SOReL on Gymnax/Brax Datasets:
1. Build the Docker image:
   ```python
   cd brax_dev
   bash build.sh
2. Launch the Docker container:
   ```python
    cd ..
    bash brax_launch_container.sh 0  # If using GPU 0 
3.	Train the posterior (world model and inference): ```python3.10 -m sorel.fit_posterior```.
   The posterior will be saved to: sorel/runs/{dataset_name}/{seed}/{n_value}.
4.	Run a hyperparameter sweep over the BAMPD solver (PPO_RNN): ```wandb sweep sorel/configs/solve_bamdp.yaml```.
5.	Train and save the actor using the hyperparameter configuration that yields the lowest approximate regret:```python3.10 -m sorel.solve_bamdp```.

# TOReL
TOReL automatically saves each trained policy during hyperparameter sweeps.

Note: TOReL (as applied here) is not compatible with Gymnax, as Unifloral restricts actions to continuous values in [-1, 1].

Policies are saved to: torel/runs/{orl_algo}/{dataset_name}/{seed}(/{n_value})/{hyperparameter_combination}_actor.pkl. 

Running TOReL on Brax Datasets:
1.	Build the Docker image:
	```python
    cd brax_dev
    bash build.sh
2.	Launch the Docker container:
    ```python
  	cd ..
    bash brax_launch_container.sh 0  # If using GPU 0
3.	Train the TOReL posterior environment:```python3.10 -m torel.fit_posterior```.
    The posterior will be saved to: torel/runs/torel/{dataset_name}/{seed}/{n_value}.
4.	Sweep over ORL algorithm hyperparameters:```wandb sweep torel/brax_configs/rebrac.yaml```. 
  	(We use different config folders, as we require Python 3.10 for Brax and 3.9 for D4RL).
5.	Run the sample-efficiency experiment:```python3.10 -m torel.unifloral_online_tuning```.
    This will save a dictionary including the following items to the path torel/runs/torel/{dataset_name}/{seed}/{n_value}:
  	- "samples": number of online steps
    - "estimated_bests_mean": regret at each step
    - "estimated_bests_ci_low", "estimated_bests_ci_high": 95% confidence bounds.

Running TOReL on D4RL Datasets:
1.	Build the Docker image:
    ```python
    cd d4rl_dev
    bash build.sh
2. Launch the Docker container:
	```python
    cd ..
    bash d4rl_launch_container.sh 0  # If using GPU 0
3.	Train the TOReL posterior environment:```python3.9 -m torel.fit_posterior```
    The posterior will be saved to: torel/runs/torel/{dataset_name}/{seed}.
4.	Sweep over ORL algorithm hyperparameters:```wandb sweep torel/d4rl_configs/rebrac.yaml```.
    (We use different config folders, as we require Python 3.10 for Brax and 3.9 for D4RL).
5.	Run the sample-efficiency experiment:```python3.9 -m torel.unifloral_online_tuning```
    This will save a dictionary including the following items to the path torel/runs/torel/{dataset_name}/{seed}:
    - "samples": number of online steps
	- "estimated_bests_mean": regret at each step
	- "estimated_bests_ci_low", "estimated_bests_ci_high": 95% confidence bounds.
  	
# Notes
- Full-replay Brax datasets should be stored in the datasets/ folder and named to include the task, e.g.: brax-halfcheetah-full-replay. 
- SOReL can be adapted to work with D4RL datasets as in TOReL, but this is not recommended: SOReL assumes a diverse offline dataset (poor to expert trajectories), which D4RL typically does not provide.
- All paper plots were generated from results downloaded via Weights & Biases (wandb).

# Acknowledgements
We gratefully build on the following works:
- A Clean Slate for Offline Reinforcement Learning; Matthew T. Jackson, Uljad Berdica, Jarek Liesen, Shimon Whiteson, Jakob N. Foerster (2025); https://arxiv.org/abs/2504.11453.
- rejax; Jarek Liesen, Chris Lu, Robert Lange (2024); https://github.com/keraJLi/rejax.
- purejaxrl; Chris Lu, Jakub Kuba, Alistair Letcher, Luke Metz, Christian Schroeder de Witt, Jakob Foerster (2022); https://github.com/luchris429/purejaxrl.
