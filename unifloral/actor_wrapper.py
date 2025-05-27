class ActorWrapper:
    """Wrapper class for actor network that handles sampling actions."""

    def __init__(
        self,
        actor_net,
        actor_params):
        self.actor_net = actor_net
        self.actor_params = actor_params

    def __call__(self, obs, rng):
        pi = self.actor_net.apply(self.actor_params, obs)
        action = pi.sample(seed=rng)
        return action

