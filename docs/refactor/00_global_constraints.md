# 00 Global Constraints

- Dynamics must not import PPO, reward, training, replay buffer, episode, or neural-network tensor code.
- PPO must interact only through standard environment/policy interfaces, observations, action masks, rewards, and done flags.
- Rewards and logging are read-only with respect to physical state.
- Scenarios define initial conditions and distributions; dynamics define time evolution.
- The canonical coordinate serialization is XZY: `[x, z, y]`, with `y` as altitude.
