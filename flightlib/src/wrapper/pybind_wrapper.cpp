
// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// flightlib
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_vec_env.hpp"
#include "flightlib/envs/vec_env_base.hpp"
#include "flightlib/envs/vision_env/vision_env.hpp"
#include "flightlib/envs/vision_env/vision_vec_env.hpp"
#include "flightlib/envs/agile_env/agile_env.hpp"
#include "flightlib/envs/agile_env/agile_vec_env.hpp"
namespace py = pybind11;
using namespace flightlib;
//AgileEnv_v1

PYBIND11_MODULE(flightgym, m) {
  py::class_<QuadrotorVecEnv<QuadrotorEnv>>(m, "QuadrotorEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset",
         static_cast<bool (QuadrotorVecEnv<QuadrotorEnv>::*)(
           Ref<MatrixRowMajor<>>)>(&QuadrotorVecEnv<QuadrotorEnv>::reset),
         "reset")
    .def("reset",
         static_cast<bool (QuadrotorVecEnv<QuadrotorEnv>::*)(
           Ref<MatrixRowMajor<>>, bool)>(&QuadrotorVecEnv<QuadrotorEnv>::reset),
         "reset with random option")
    .def("step", &QuadrotorVecEnv<QuadrotorEnv>::step)
    .def("setSeed", &QuadrotorVecEnv<QuadrotorEnv>::setSeed)
    .def("close", &QuadrotorVecEnv<QuadrotorEnv>::close)
    .def("isTerminalState", &QuadrotorVecEnv<QuadrotorEnv>::isTerminalState)
    .def("curriculumUpdate", &QuadrotorVecEnv<QuadrotorEnv>::curriculumUpdate)
    .def("connectUnity", &QuadrotorVecEnv<QuadrotorEnv>::connectUnity)
    .def("disconnectUnity", &QuadrotorVecEnv<QuadrotorEnv>::disconnectUnity)
    .def("updateUnity", &QuadrotorVecEnv<QuadrotorEnv>::updateUnity)
    .def("getObs", &QuadrotorVecEnv<QuadrotorEnv>::getObs)
    .def("getQuadAct", &QuadrotorVecEnv<QuadrotorEnv>::getQuadAct)
    .def("getQuadState", &QuadrotorVecEnv<QuadrotorEnv>::getQuadState)
    .def("getImage", &QuadrotorVecEnv<QuadrotorEnv>::getImage)
    .def("getDepthImage", &QuadrotorVecEnv<QuadrotorEnv>::getDepthImage)
    .def("getNumOfEnvs", &QuadrotorVecEnv<QuadrotorEnv>::getNumOfEnvs)
    .def("getObsDim", &QuadrotorVecEnv<QuadrotorEnv>::getObsDim)
    .def("getActDim", &QuadrotorVecEnv<QuadrotorEnv>::getActDim)
    .def("getRewDim", &QuadrotorVecEnv<QuadrotorEnv>::getRewDim)
    .def("getImgHeight", &QuadrotorVecEnv<QuadrotorEnv>::getImgHeight)
    .def("getImgWidth", &QuadrotorVecEnv<QuadrotorEnv>::getImgWidth)
    .def("getRewardNames", &QuadrotorVecEnv<QuadrotorEnv>::getRewardNames)
    .def("getExtraInfoNames", &QuadrotorVecEnv<QuadrotorEnv>::getExtraInfoNames)
    .def("__repr__", [](const QuadrotorVecEnv<QuadrotorEnv>& a) {
      return "RPG Drone Control Environment";
    });

  py::class_<VisionVecEnv<VisionEnv>>(m, "VisionEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset",
         static_cast<bool (VisionVecEnv<VisionEnv>::*)(Ref<MatrixRowMajor<>>)>(
           &VisionVecEnv<VisionEnv>::reset),
         "reset")
    .def("reset",
         static_cast<bool (VisionVecEnv<VisionEnv>::*)(
           Ref<MatrixRowMajor<>>, bool)>(&VisionVecEnv<VisionEnv>::reset),
         "reset with random option")
    .def("step", &VisionVecEnv<VisionEnv>::step)
    .def("setSeed", &VisionVecEnv<VisionEnv>::setSeed)
    .def("close", &VisionVecEnv<VisionEnv>::close)
    .def("isTerminalState", &VisionVecEnv<VisionEnv>::isTerminalState)
    .def("curriculumUpdate", &VisionVecEnv<VisionEnv>::curriculumUpdate)
    .def("connectUnity", &VisionVecEnv<VisionEnv>::connectUnity)
    .def("disconnectUnity", &VisionVecEnv<VisionEnv>::disconnectUnity)
    .def("updateUnity", &VisionVecEnv<VisionEnv>::updateUnity)
    .def("getObs", &VisionVecEnv<VisionEnv>::getObs)
    .def("getQuadAct", &VisionVecEnv<VisionEnv>::getQuadAct)
    .def("getQuadState", &VisionVecEnv<VisionEnv>::getQuadState)
    .def("getImage", &VisionVecEnv<VisionEnv>::getImage)
    .def("getDepthImage", &VisionVecEnv<VisionEnv>::getDepthImage)
    .def("getNumOfEnvs", &VisionVecEnv<VisionEnv>::getNumOfEnvs)
    .def("getObsDim", &VisionVecEnv<VisionEnv>::getObsDim)
    .def("getActDim", &VisionVecEnv<VisionEnv>::getActDim)
    .def("getRewDim", &VisionVecEnv<VisionEnv>::getRewDim)
    .def("getImgHeight", &VisionVecEnv<VisionEnv>::getImgHeight)
    .def("getImgWidth", &VisionVecEnv<VisionEnv>::getImgWidth)
    .def("getRewardNames", &VisionVecEnv<VisionEnv>::getRewardNames)
    .def("getExtraInfoNames", &VisionVecEnv<VisionEnv>::getExtraInfoNames)
    .def("__repr__", [](const VisionVecEnv<VisionEnv>& a) {
      return "RPG Vision-based Agile Flight Environment";
    });

  py::class_<AgileVecEnv<AgileEnv>>(m, "AgileEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset",
         static_cast<bool (AgileVecEnv<AgileEnv>::*)(Ref<MatrixRowMajor<>>)>(
           &AgileVecEnv<AgileEnv>::reset),
         "reset")
    .def("reset",
         static_cast<bool (AgileVecEnv<AgileEnv>::*)(
           Ref<MatrixRowMajor<>>, bool)>(&AgileVecEnv<AgileEnv>::reset),
         "reset with random option")
    .def("step", &AgileVecEnv<AgileEnv>::step)
    .def("setSeed", &AgileVecEnv<AgileEnv>::setSeed)
    .def("close", &AgileVecEnv<AgileEnv>::close)
    .def("isTerminalState", &AgileVecEnv<AgileEnv>::isTerminalState)
    .def("curriculumUpdate", &AgileVecEnv<AgileEnv>::curriculumUpdate)
    .def("connectUnity", &AgileVecEnv<AgileEnv>::connectUnity)
    .def("disconnectUnity", &AgileVecEnv<AgileEnv>::disconnectUnity)
    .def("updateUnity", &AgileVecEnv<AgileEnv>::updateUnity)
    .def("getObs", &AgileVecEnv<AgileEnv>::getObs)
    .def("getQuadAct", &AgileVecEnv<AgileEnv>::getQuadAct)
    .def("getQuadState", &AgileVecEnv<AgileEnv>::getQuadState)
    .def("getImage", &AgileVecEnv<AgileEnv>::getImage)
    .def("getDepthImage", &AgileVecEnv<AgileEnv>::getDepthImage)
    .def("getNumOfEnvs", &AgileVecEnv<AgileEnv>::getNumOfEnvs)
    .def("getObsDim", &AgileVecEnv<AgileEnv>::getObsDim)
    .def("getActDim", &AgileVecEnv<AgileEnv>::getActDim)
    .def("getRewDim", &AgileVecEnv<AgileEnv>::getRewDim)
    .def("getImgHeight", &AgileVecEnv<AgileEnv>::getImgHeight)
    .def("getImgWidth", &AgileVecEnv<AgileEnv>::getImgWidth)
    .def("getRewardNames", &AgileVecEnv<AgileEnv>::getRewardNames)
    .def("getExtraInfoNames", &AgileVecEnv<AgileEnv>::getExtraInfoNames)
    .def("__repr__", [](const AgileVecEnv<AgileEnv>& a) {
      return "RPG Vision-based Agile Flight Environment";
    });
}