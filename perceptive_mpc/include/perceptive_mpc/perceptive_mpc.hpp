#pragma once

#include <memory>

#include <ocs2_anymal_commands/ReferenceExtrapolation.h>
#include <ocs2_anymal_models/FrameDeclaration.h>
#include <ocs2_anymal_models/QuadrupedCom.h>

#include <ocs2_anymal_mpc/AnymalInterface.h>
#include <ocs2_quadruped_interface/QuadrupedInterface.h>
#include <ocs2_switched_model_interface/core/Rotations.h>
#include <ocs2_switched_model_interface/core/SwitchedModel.h>
#include <ocs2_switched_model_interface/logic/SwitchedModelModeScheduleManager.h>
#include <ocs2_switched_model_interface/logic/ModeSequenceTemplate.h>
#include <segmented_planes_terrain_model/SegmentedPlanesTerrainModel.h>


std::string loadUrdf(const std::string &urdfFile);
std::unique_ptr<switched_model::QuadrupedInterface> createQuadrupedInterface(const std::string &urdf,
                                                                             const std::string &taskFolder);