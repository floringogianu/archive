""" Centipede environments.
    Original authors: Tingwu (Wilson) Wang, Aug. 30nd, 2017
    https://github.com/WilsonWangTHU/NerveNet
"""
from pathlib import Path

import num2words
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


__all__ = [
    "CentipedeFourEnv",
    "CentipedeSixEnv",
    "CentipedeEightEnv",
    "CentipedeTenEnv",
    "CentipedeTwelveEnv",
]


class CentipedeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
        @brief:
            In the CentipedeEnv, we define a task that is designed to test the
            power of transfer learning for gated graph neural network.
        @children:
            @CentipedeFourEnv
            @CentipedeEightEnv
            @CentipedeTenEnv
            @CentipedeTwelveEnv
    """

    def __init__(self, CentipedeLegNum=4, is_crippled=False):

        # get the path of the environments
        if is_crippled:
            xml_name = "CpCentipede" + self.get_env_num_str(CentipedeLegNum) + ".xml"
        else:
            xml_name = "Centipede" + self.get_env_num_str(CentipedeLegNum) + ".xml"
        crt_path = Path(__file__).resolve().parent
        xml_path = crt_path / "assets" / xml_name
        self.num_body = int(np.ceil(CentipedeLegNum / 2.0))
        self._control_cost_coeff = 0.5 * 4 / CentipedeLegNum
        self._contact_cost_coeff = 0.5 * 1e-3 * 4 / CentipedeLegNum

        self.torso_geom_id = 1 + np.array(range(self.num_body)) * 5
        # make sure the centipede is not born to be end of episode
        self.body_qpos_id = 6 + 6 + np.array(range(self.num_body)) * 6
        self.body_qpos_id[-1] = 5

        mujoco_env.MujocoEnv.__init__(self, xml_path.as_posix(), 5)

        utils.EzPickle.__init__(self)

    def get_env_num_str(self, number):
        num_str = num2words.num2words(number)
        return num_str[0].upper() + num_str[1:]

    def step(self, a):
        xposbefore = self.get_body_com("torso_" + str(self.num_body - 1))[0]
        self.do_simulation(a, self.frame_skip)
        """
        xposafter = np.mean([self.get_body_com("torso_" + str(i_torso))[0]
                             for i_torso in range(self.num_body)])
        """
        xposafter = self.get_body_com("torso_" + str(self.num_body - 1))[0]

        # calculate reward
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = self._control_cost_coeff * np.square(a).sum()
        contact_cost = self._contact_cost_coeff * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        # check if finished
        state = self.state_vector()
        notdone = (
            np.isfinite(state).all()
            and self._check_height()
            and self._check_direction()
        )
        done = not notdone

        ob = self._get_obs()

        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[2:],
                self.data.qvel.flat,
                np.clip(self.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        while True:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-0.1, high=0.1
            )
            qpos[self.body_qpos_id] = self.np_random.uniform(
                size=len(self.body_qpos_id),
                low=-0.1 / (self.num_body - 1),
                high=0.1 / (self.num_body - 1),
            )

            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
            self.set_state(qpos, qvel)
            if self._check_height() and self._check_direction():
                break
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
        body_name = "torso_" + str(int(np.ceil(self.num_body / 2 - 1)))
        self.viewer.cam.trackbodyid = self.model.body_names.index(body_name)

    """
    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.5).all() and (height > 0.35).all()
    """

    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.15).all() and (height > 0.35).all()

    def _check_direction(self):
        y_pos_pre = self.data.geom_xpos[self.torso_geom_id[:-1], 1]
        y_pos_post = self.data.geom_xpos[self.torso_geom_id[1:], 1]
        y_diff = np.abs(y_pos_pre - y_pos_post)
        return (y_diff < 0.45).all()


# the following environments are just models with even number legs


class CpCentipedeFourEnv(CentipedeEnv):
    def __init__(self):
        super(CpCentipedeFourEnv, self).__init__(CentipedeLegNum=4, is_crippled=True)


class CpCentipedeSixEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=6, is_crippled=True)


class CpCentipedeEightEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=8, is_crippled=True)


class CpCentipedeTenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=10, is_crippled=True)


class CpCentipedeTwelveEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=12, is_crippled=True)


class CpCentipedeFourteenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=14, is_crippled=True)


# regular


class CentipedeFourEnv(CentipedeEnv):
    def __init__(self):
        super(CentipedeFourEnv, self).__init__(CentipedeLegNum=4)


class CentipedeSixEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=6)


class CentipedeEightEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=8)


class CentipedeTenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=10)


class CentipedeTwelveEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=12)


class CentipedeFourteenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=14)


class CentipedeTwentyEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=20)


class CentipedeThirtyEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=30)


class CentipedeFortyEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=40)


class CentipedeFiftyEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=50)


class CentipedeOnehundredEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=100)


"""
    the following environments are models with odd number legs
"""


class CentipedeThreeEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=3)


class CentipedeFiveEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=5)


class CentipedeSevenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=7)
