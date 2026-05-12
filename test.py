from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
from ev2gym.rl_agent.state import V2G_profit_max
import time

#config_file = "ev2gym/example_config_files/V2GProfitPlusLoads.yaml"
config_file = "custom.yaml"

# Initialize the environment
env = EV2Gym(config_file=config_file,
              save_replay=True,
              save_plots=True,
              state_function=V2G_profit_max)

state, _ = env.reset()

agent = ChargeAsFastAsPossible() # heuristic

start = time.time()
for ep in range(1):
    print("episode", ep)
    env.reset()
    for t in range(env.simulation_length):
        actions = agent.get_action(env) # get action from the agent
        new_state, _, done, truncated, stats = env.step(actions)  # takes action

end = time.time()
print(end - start)
