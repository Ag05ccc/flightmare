# DEBUG
import cv2
import subprocess



import os
import pickle
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import gym
import numpy as np
# from gym import spaces
from numpy.core.fromnumeric import shape
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvIndices,
                                                           VecEnvObs,
                                                           VecEnvStepReturn)
from stable_baselines3.common.vec_env.util import (copy_obs_dict, dict_to_obs,
                                                   obs_space_info)

# DEBUG
from gymnasium import spaces

class FlightEnvVec(VecEnv):
    # AgileEv_v1
    def __init__(self, impl):
        self.wrapper = impl
        
        self.act_dim = self.wrapper.getActDim()
        self.obs_dim = self.wrapper.getObsDim()
        self.rew_dim = self.wrapper.getRewDim()
        self.img_width = self.wrapper.getImgWidth()
        self.img_height = self.wrapper.getImgHeight()
        
        self.IMAGE_FLAG = False

        # IMAGE
        # if self.IMAGE_FLAG:
        #     self._observation_space = spaces.Dict({
        #         "state": spaces.Box(
        #             low=np.ones(self.obs_dim) * -np.Inf,
        #             high=np.ones(self.obs_dim) * np.Inf,
        #             dtype=np.float64
        #         ),
        #         # NORMALDE H,W,1 SEKLINDEYDI - 1,H,W YAPTIK, ENSON H,W KALDI
        #         "image": spaces.Box(
        #             low=np.zeros((1, self.img_height, self.img_width)),  # Assuming RGB images
        #             high=np.ones((1, self.img_height, self.img_width)) * 255,
        #             dtype=np.float32
        #         ),
        #     })
        # else:
        #     self._observation_space = spaces.Dict({
        #     "state": spaces.Box(
        #         low=np.ones(self.obs_dim) * -np.Inf,
        #         high=np.ones(self.obs_dim) * np.Inf,
        #         dtype=np.float64
        #     ),
        # })
        
        # LSTM
        self._observation_space = spaces.Box(
            np.ones(self.obs_dim) * -np.Inf,
            np.ones(self.obs_dim) * np.Inf,
            dtype=np.float64,
        )
        
        self._action_space = spaces.Box(
            low=np.ones(self.act_dim) * -1.0,
            high=np.ones(self.act_dim) * 1.0,
            dtype=np.float64,
        )
        # DEFAULT
        self._observation = np.zeros([self.num_envs, self.obs_dim], dtype=np.float64)
        
        # DEBUG
        # if self.IMAGE_FLAG:
        #     self._observation = {
        #         "state": np.zeros([self.num_envs, self.obs_dim], dtype=np.float64),
        #         "image": np.zeros([self.num_envs, 1, self.img_height, self.img_width], dtype=np.float32)
        #     }
        # else:
        #     self._observation = {
        #         "state": np.zeros([self.num_envs, self.obs_dim], dtype=np.float64),
        #     }

        self._rgb_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height * 3], dtype=np.uint8
        )
        self._gray_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height], dtype=np.uint8
        )
        self._depth_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height], dtype=np.float32
        )
        #
        self._reward_components = np.zeros(
            [self.num_envs, self.rew_dim], dtype=np.float64
        )
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self.reward_names = self.wrapper.getRewardNames()
        self._extraInfo = np.zeros(
            [self.num_envs, len(self._extraInfoNames)], dtype=np.float64
        )

        self.rewards = [[] for _ in range(self.num_envs)]
        self.sum_reward_components = np.zeros(
            [self.num_envs, self.rew_dim - 1], dtype=np.float64
        )

        self._quadstate = np.zeros([self.num_envs, 25], dtype=np.float64)
        self._quadact = np.zeros([self.num_envs, 4], dtype=np.float64)
        self._flightmodes = np.zeros([self.num_envs, 1], dtype=np.float64)

        #  state normalization
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.obs_dim])
        self.obs_rms_new = RunningMeanStd(shape=[self.num_envs, self.obs_dim])

        self.max_episode_steps = 1000

        # VecEnv.__init__(self, self.num_envs,
        #                 self._observation_space, self._action_space)
        # self.render_mode = "rgb_array"

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def update_rms(self):
        self.obs_rms = self.obs_rms_new

    def teststep(self, action):
        self.wrapper.testStep(
            action,
            self._observation,
            self._reward_components,
            self._done,
            self._extraInfo,
        )

        # print("Test_step : ", self._observation, "      type: ",type(self._observation))
        # print("Test_step : ", self._observation, "      type: ",type(self._observation))
        # print("Test_step : ", self._observation, "      type: ",type(self._observation))
        # print("Test_step : ", self._observation, "      type: ",type(self._observation))
        # print("Test_step : ", self._observation, "      type: ",type(self._observation))
        obs = self.normalize_obs(self._observation)

        # ADD IMAGE
        if self.IMAGE_FLAG:
            depth_img = self.getDepthImage()
            depth_img = np.reshape(depth_img, (self.num_envs, 1, self.img_height, self.img_width))
            obs = {"state": obs,
                "image": depth_img}
        else:
            obs = {"state": obs}


        return (
            obs,
            self._reward_components[:, -1].copy(), # total_reward kullaniyor sadece 
            self._done.copy(),
            self._extraInfo.copy(),
        )

    def step(self, action):
        if action.ndim <= 1:
            action = action.reshape((-1, self.act_dim))
        
        self.wrapper.step(
            action,
            self._observation,
            # self._observation["state"],
            self._reward_components,
            self._done,
            self._extraInfo,
        )
        # lin_vel_reward, collision_penalty, ang_vel_penalty, survive_rew_, total_reward
        # print("REWARD: ",self._reward_components[0])
        
        
        # print("step : ", self._observation["image"].shape, "      type: ",type(self._observation["image"]))


        # print("step : ", self._observation, "      type: ",type(self._observation))
        # print("step : ", self._observation, "      type: ",type(self._observation))
        # print("step : ", self._observation, "      type: ",type(self._observation))
        # print("step : ", self._observation, "      type: ",type(self._observation))

        # update the mean and variance of the Running Mean STD
        self.obs_rms_new.update(self._observation["state"])
        obs = self.normalize_obs(self._observation["state"])
        
        # MultiInput-V1
        # obs = {"state": obs}

        # ADD IMAGE
        # depth_img = self.getDepthImage()

        # ------------------------------DEBUG------------------------------
        # get_img = self.getImage(rgb=True)
        # 230400 = 240 * 320 * 3
        # print("shape : ", get_img.shape, "type: ", type(get_img))
        # if np.any(get_img):
            # print("There is at least one non-zero element in the array.")
            # temp_img = np.reshape(get_img, (240, 320, 3))
            # depth_img = np.reshape(env.getDepthImage()[
            #             0], (env.img_height, env.img_width))
            # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)

            # cv2.imshow("temp_img",temp_img)
            # cv2.waitKey(1)
        # else:        
        #     print("All elements in the array are zero.")

        
        # temp_img = np.reshape(depth_img, (240, 320, self.num_envs))
        # temp_img = np.reshape(depth_img, (self.num_envs, 160, 240))
        
        # cv2.imshow("temp_img0",temp_img[0,:,:])
        # cv2.imshow("temp_img1",temp_img[1,:,:])
        # cv2.imshow("temp_img2",temp_img[2,:,:])
        # cv2.imshow("temp_img3",temp_img[3,:,:])
        # cv2.imshow("temp_img4",temp_img[4,:,:])
        
        # cv2.waitKey(100)
        # ------------------------------DEBUG------------------------------,
        if self.IMAGE_FLAG:
            depth_img = self.getDepthImage()
            depth_img = np.reshape(depth_img, (self.num_envs, 1, self.img_height, self.img_width))
            obs = {"state": obs,
                "image": depth_img}
        else:
            obs = {"state": obs}
        

        # print("step : ", obs["image"].shape, "      type: ",type(obs["image"]))

        if len(self._extraInfoNames) != 0:
            info = [
                {
                    "extra_info": {
                        self._extraInfoNames[j]: self._extraInfo[i, j]
                        for j in range(0, len(self._extraInfoNames))
                    }
                }
                for i in range(self.num_envs)
            ]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward_components[i, -1])
            for j in range(self.rew_dim - 1):
                self.sum_reward_components[i, j] += self._reward_components[i, j]
            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                for j in range(self.rew_dim - 1):
                    epinfo[self.reward_names[j]] = self.sum_reward_components[i, j]
                    self.sum_reward_components[i, j] = 0.0
                info[i]["episode"] = epinfo

                # print("len(self.rewards)       :  ", len(self.rewards))
                # print("self.rewards            :  ", len(self.rewards[i]))
                # print()
                # print("self.rewards[i]         :  ", self.rewards[i])
                # print("self._reward_components :   ",self._reward_components)
                # print("............................................................")
                self.rewards[i].clear()
                # if len(self.rewards[i])>0:
                #     print(" @@@@@@@  len(self.rewards[i] :      ",len(self.rewards[i]))

        # print("INFO : ", info)
        # print("INFO : ", info)
        # print("INFO : ", info)
        # print("INFO : ", info)
        # print("INFO : ", info)
        
        # print("STEP STEP STEP : ",obs["image"].shape)
        # DEBUG - RENDER NORMALDE YOK
        # TODO: len(self.reward[i]) buradaki "i" yukarıdaki for loop'da kalan son i degeri
        # butun i degerlerinin uzunlugu aynı mı kontrol et. Etkisi ne olur ? Ne ise yariyor
        # butun i degerleri icin farkli deger geliyor - sirali artiyor
        # len(self.rewards) - env sayisi kadar
        # self.rewards[i]   - degisken 
        # self._reward_components
                
        # print("len(self.rewards[i] :      ",len(self.rewards[i]))
        # print("len(self.rewards)   :      ",len(self.rewards))
                
        # DEBUG CIFT RENDER YAPIYOR TEST SIRASINDA
        # self.render(frame_id = len(self.rewards[i]), mode="human")
        # self.render(frame_id = 10, mode="human")
        return (
            obs,
            self._reward_components[:, -1].copy(),
            self._done.copy(),
            info.copy(),
        )

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float64)

    def reset(self, random=True):
        # Dict tipindeki observation icin reset fonksiyonunu duzenle
        self._reward_components = np.zeros(
            [self.num_envs, self.rew_dim], dtype=np.float64
        )
        # reset observation alip size kontrol yapiyor. Sadece state versek sorun degil gibi gozukuyor
        self.wrapper.reset(self._observation, random)
        # self.wrapper.reset(self._observation["state"], random)
        # self.wrapper.reset(self._observation["image"], random)
        # print("reset : ", self._observation, "      type: ",type(self._observation))
        # print("reset : ", self._observation, "      type: ",type(self._observation))
        # print("reset : ", self._observation, "      type: ",type(self._observation))
        # print("reset : ", self._observation, "      type: ",type(self._observation))
        # print("reset : ", self._observation, "      type: ",type(self._observation))
        obs = self._observation
        #
        # self.obs_rms_new.update(self._observation["state"])
        # obs = self.normalize_obs(self._observation["state"])

        self.obs_rms_new.update(self._observation)
        obs = self.normalize_obs(self._observation)

        # MultiInput-V1
        # obs = {"state": obs}

        # ADD IMAGE
        if self.IMAGE_FLAG:
            depth_img = self.getDepthImage()
            depth_img = np.reshape(depth_img, (self.num_envs, 1, self.img_height, self.img_width))
            obs = {"state": obs,
                "image": depth_img}
        else:
            pass
            # obs = {"state": obs}
        
        # print("wrapper reset() : ", obs["image"].shape, "      type: ",type(obs["image"]))

        # @TODO: HER SEFERINDE SIFIR DEGER VERIYORUZ GOZLEMI VERMEK YERINE BU YANLIS
        # SADECE TYPE KABUL EDECEK MI DENEMEK ICIN
        # obs = spaces.Dict({
        #     "state": spaces.Box(
        #         low=np.ones(self.obs_dim) * -np.Inf,
        #         high=np.ones(self.obs_dim) * np.Inf,
        #         dtype=np.float64
        #     ),
        # })

        # print("RESET RESET RESET : ",obs["image"].shape)
        
        if self.num_envs == 1:
            # print(obs["state"].shape)
            # print("---------------------------------")
            # obs2 = spaces.Dict({
            #     "state": spaces.Box(
            #         low=np.ones(self.obs_dim) * -np.Inf,
            #         high=np.ones(self.obs_dim) * np.Inf,
            #         dtype=np.float64
            #     ),
            # })
            # print(obs2)
            # print(obs2["state"].shape)
            # print(obs2["state"])
            # print(self._observation_space["state"].shape)
            # print(self._observation_space["state"])
            return obs
        return obs

    def getObs(self):
        self.wrapper.getObs(self._observation)
        # print("getObs : ", self._observation, "      type: ",type(self._observation))
        # print("getObs : ", self._observation, "      type: ",type(self._observation))
        # print("getObs : ", self._observation, "      type: ",type(self._observation))
        # print("getObs : ", self._observation, "      type: ",type(self._observation))
        # print("getObs : ", self._observation, "      type: ",type(self._observation))
        return self.normalize_obs(self._observation)

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def get_obs_norm(self):
        return self.obs_rms.mean, self.obs_rms.var

    def getProgress(self):
        return self._reward_components[:, 0]

    def getImage(self, rgb=False):
        if rgb:
            self.wrapper.getImage(self._rgb_img_obs, True)
            return self._rgb_img_obs.copy()
        else:
            self.wrapper.getImage(self._gray_img_obs, False)
            return self._gray_img_obs.copy()

    def getDepthImage(self):
        # shape : (100, 76800) ( 320 * 240  = 76800)
        self.wrapper.getDepthImage(self._depth_img_obs)
        return self._depth_img_obs.copy()

    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(
            action,
            self._observation,
            self._reward,
            self._done,
            self._extraInfo,
            send_id,
        )
        # print("stepUnity : ", self._observation, "      type: ",type(self._observation))
        # print("stepUnity : ", self._observation, "      type: ",type(self._observation))
        # print("stepUnity : ", self._observation, "      type: ",type(self._observation))
        # print("stepUnity : ", self._observation, "      type: ",type(self._observation))
        # print("stepUnity : ", self._observation, "      type: ",type(self._observation))

        return receive_id

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)

    def _unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to unnormalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: unnormalized observation
        """
        return (obs * np.sqrt(obs_rms.var + 1e-8)) + obs_rms.mean

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        # obs_ = deepcopy(obs)
        obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float64)
        return obs_

    def getQuadState(self):
        self.wrapper.getQuadState(self._quadstate)
        return self._quadstate

    def getQuadAct(self):
        self.wrapper.getQuadAct(self._quadact)
        return self._quadact

    def getExtraInfo(self):
        return self._extraInfo

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            for j in range(self.rew_dim - 1):
                epinfo[self.reward_names[j]] = self.sum_reward_components[i, j]
                self.sum_reward_components[i, j] = 0.0
            info[i]["episode"] = epinfo
            self.rewards[i].clear()
        return info

    def render(self, frame_id, mode="human"):
        return self.wrapper.updateUnity(frame_id)

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        self.wrapper.connectUnity()

    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    def curriculumUpdate(self):
        self.wrapper.curriculumUpdate()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError("This method is not implemented")

    def stop_recording_video(self):
        raise RuntimeError("This method is not implemented")

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    # DEBUG
    def step_async(self, actual_acts):
        #return self.step(actual_acts)
        raise RuntimeError("This method is not implemented")

    # DEBUG
    def step_wait(self):
        # return self.step()     
        #return self._observation, self._reward_components[:, -1].copy(), self._done.copy(), self._extraInfo.copy()
        raise RuntimeError("This method is not implemented")

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise AttributeError("This method is not implemented")
        # raise RuntimeError("This method is not implemented")

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError("This method is not implemented")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError("This method is not implemented")

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> "VecNormalize":
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    def save_rms(self, save_dir, n_iter) -> None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data_path = save_dir + "/iter_{0:05d}".format(n_iter)
        np.savez(
            data_path,
            mean=np.asarray(self.obs_rms.mean),
            var=np.asarray(self.obs_rms.var),
        )

    def load_rms(self, data_dir) -> None:
        self.mean, self.var = None, None
        np_file = np.load(data_dir)
        #
        self.mean = np_file["mean"]
        self.var = np_file["var"]
        #
        self.obs_rms.mean = np.mean(self.mean, axis=0)
        self.obs_rms.var = np.mean(self.var, axis=0)
