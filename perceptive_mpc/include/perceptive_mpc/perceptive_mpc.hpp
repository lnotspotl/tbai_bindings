#pragma once

#include <memory>

#include <ocs2_anymal_commands/ReferenceExtrapolation.h>
#include <ocs2_anymal_mpc/AnymalInterface.h>
#include <ocs2_quadruped_interface/QuadrupedInterface.h>
#include <ocs2_switched_model_interface/core/Rotations.h>
#include <ocs2_switched_model_interface/logic/SwitchedModelModeScheduleManager.h>
#include <segmented_planes_terrain_model/SegmentedPlanesTerrainModel.h>

std::string loadUrdf(const std::string &urdfFile);
std::unique_ptr<switched_model::QuadrupedInterface> createQuadrupedInterface(const std::string &urdf,
                                                                             const std::string &taskFolder);