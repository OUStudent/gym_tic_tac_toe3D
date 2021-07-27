from gym.envs.registration import register

register(
    id='tic_tac_toe3D-v0',
    entry_point='gym_tic_tac_toe3D.envs:tic_tac_toe3DEnv'
)
