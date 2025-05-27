Two dev folders exist, due to different compatibilities between brax and D4RL. 

--- SOReL --- 

For SOReL, we do not automatically save each policy trained during hyperparameter sweeps, as we sweep over too large a hyperparameter space. 

SOReL on gymnax/brax datasets: 
    1. Build the docker image using: cd brax_dev, cd.., bash build.sh 
    2. Launch the docker container using: cd.., bash d4rl_launch_container.sh 0 # (if launching on GPU 0, for example, otherwise just bash d4rl_launch_container.sh)
    3. Train the posterior (world model and approximate inference) using:
        python3.9 -m sorel.fit_posterior
        The posterior model will be saved in sorel/runs/{dataset_name}/{seed}/{n_value}
    4. Sweep over the hyperparameters of the bamdp solver (here we implement PPO_RNN)
        wandb sweep sorel/configs/solve_bamdp.yaml 
        Use wandb to choose the hyperparameters with the lowest approximate regret. 
    5. Run python3.10 -m sorel.solve_bamdp with the chosen hyperparameters to save the actor.


--- TOReL --- 

For TOReL, we automatically save each policy trained during hyperparameter sweeps. 
Currently, the TOReL algorithms are not compatible with gymnax (unifloral's implementation clips actions between -1 and 1, and only handles continuous action spaces)
The policies are saved as torel/runs/{orl_algo}/{dataset_name}/{seed}(/{n_value})/{hyperparameter_combination}_actor.pkl

TOReL on brax datasets: 
    1. Build the docker image using: cd brax_dev, cd.., bash build.sh 
    2. Launch the docker container using: cd.., bash brax_launch_container.sh 0 # (if launching on GPU 0, for example, otherwise just bash d4rl_launch_container.sh)
    3. Train the torel environment for calculating the regret metric using: 
        python3.9 -m torel.fit_posterior
        The posterior model will be saved in torel/runs/torel/{dataset_name}/{seed}/{n_value}
    4. Sweep over the hyperparameters of the desired ORL algorithm using wandb
        e.g. wandb sweep torel/d4rl_configs/rebrac.yaml (different config folders and files for brax and D4RL - brax requires python3.10 and D4RL python3.9)
        Use wandb to choose the hyperparameters with the lowest regret metric value.  
    5. To run the sample efficiency experiment, tune online using unifloral:
        run python3.10 -m torel.unifloral_online_tuning to save a dictionary containing (among other values) the number of online samples required ("samples"), the regret with each online sample ("estimated_bests_mean"), and the 95th percentile confidence bound ("estimated_bests_ci_low" to "estimated_bests_ci_hight").
        The dictionary will be saved in torel/runs/torel/{dataset_name}/{seed}/{n_value}

TOReL on D4RL datasets: 
    1. Build the docker image using: cd d4rl_dev, bash build.sh 
    2. Launch the docker container using: cd.., bash d4rl_launch_container.sh 0 # (if launching on GPU 0, otherwise just bash d4rl_launch_container.sh)
    3. Train the torel environment for calculating the regret metric using: 
        python3.9 -m torel.fit_posterior
        The posterior model will be saved in torel/runs/torel/{dataset_name}/{seed}
    4. Sweep over the hyperparameters of the desired ORL algorithm using wandb
        e.g. wandb sweep torel/d4rl_configs/rebrac.yaml (different config folders and files for brax and D4RL - brax requires python3.10 and D4RL python3.9). 
        Use wandb to choose the hyperparameters with the lowest regret metric value. 
    5. To run the sample efficiency experiment, tune online using unifloral:
        run python3.9 -m torel.unifloral_online_tuning to save a dictionary containing (among other values) the number of online samples required ("samples"), the regret with each online sample ("estimated_bests_mean"), and the 95th percentile confidence bound ("estimated_bests_ci_low" to "estimated_bests_ci_hight").
        The dictionary will be saved in torel/runs/torel/{dataset_name}/{seed}

--- NB ---
The diverse brax full-replay datasets should be stored in the datasets folder, and contain the name of the brax task within their description. e.g. "brax-halfcheetah-full-replay". 

The example SOReL code could easily be changed to download the D4RL datasets (like TOReL does). 
We do not do this, since SOReL, as it is, is unlikely to work with D4RL - since we have not placed a prior on the model, the offline dataset must be diverse, covering poor, medium and expert regions of performance. This is not the case for most D4RL datasets.

The plots in the paper were created by downloading the relevant results from wandb. 

Acknowledgement and thanks to the authors of unifloral, rejax, and purejaxrl: 
    A Clean Slate for Offline Reinforcement Learning, Matthew Thomas Jackson and Uljad Berdica and Jarek Liesen and Shimon Whiteson and Jakob Nicolaus Foerster, 2025, https://arxiv.org/abs/2504.11453
    rejax, Liesen, Jarek and Lu, Chris and Lange, Robert, 2024, https://github.com/keraJLi/rejax
    Discovered policy optimisation, Lu, Chris and Kuba, Jakub and Letcher, Alistair and Metz, Luke and Schroeder de Witt, Christian and Foerster, Jakob, 2022, https://github.com/luchris429/purejaxrl