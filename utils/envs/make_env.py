import gymnax
from gymnax.environments import spaces
from utils.envs.env_wrappers import LogWrapper, ClipAction
from rejax.compat.brax2gymnax import create_brax


def make_env(args, **kwargs):
    
    try:
        brax_task = "halfcheetah" if "halfcheetah" in args.task else "hopper" if "hopper" in args.task else "walker2d" if "walker2d" in args.task else args.task
        env, env_params = create_brax(brax_task, **kwargs)
        env = ClipAction(env)
        env = LogWrapper(env)
        print("Brax environment created successfully")
        return env, env_params
    except:
        pass

    try: 
        gymnax_task = "CartPole-v1" if "cartpole" in args.task else "Pendulum-v1" if "pendulum" in args.task else args.task
        env, env_params = gymnax.make(gymnax_task)
        if env.action_space is spaces.Box:
            env = ClipAction(env)
        env = LogWrapper(env)
        print("Gymnax environment created successfully")
        return env, env_params
    except:
        pass

    raise ValueError("Environment not found")