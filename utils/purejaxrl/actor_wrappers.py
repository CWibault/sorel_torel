import jax
import jax.numpy as jnp

from utils.purejaxrl.networks import ScannedRNN

class ActorCriticWrapper:
    """Wrapper class for actor critic network that handles sampling actions."""

    def __init__(
        self,
        actor_critic_net,
        actor_critic_params):
        self.actor_critic_net = actor_critic_net
        self.actor_critic_params = actor_critic_params

    def __call__(self, obs, rng):
        pi, _ = self.actor_critic_net.apply(self.actor_critic_params, obs)
        action = pi.sample(seed = rng)
        return action


class RNNActorCriticWrapper:
    """Wrapper class for recurrent actor critic network that handles sampling actions."""

    def __init__(
        self,
        actor_critic_net,
        actor_critic_params,
        rnn_size):
        self.actor_critic_net = actor_critic_net
        self.actor_critic_params = actor_critic_params
        self.rnn_size = rnn_size

    def __call__(self, obs, done, hstate, rng):

        assert obs.ndim == 1 and done.ndim == 0 and hstate.ndim == 1
        
        # --- add batch (num_envs = 1) and time dimensions --- 
        done = jnp.array([done])
        obs, done = obs[None, None, :], done[None, :]
        hstate = hstate[None, :]
        
        # --- obtain action ---
        rnn_in = (obs, done)
        hstate, pi, _ = self.actor_critic_net.apply(self.actor_critic_params, hstate, rnn_in)
        action = pi.sample(seed = rng)
        
        # --- squeeze batch and time dimensions ---
        action = action.squeeze()
        hstate = hstate.squeeze()

        return action, hstate
        
    def initialize_state(self, batch_size=1):
        return ScannedRNN.initialize_carry(batch_size, self.rnn_size)