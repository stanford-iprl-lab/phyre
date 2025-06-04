// Copyright (c) Facebook, Inc. and its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "task_utils.h"
#include "task_validation.h"
#include "thrift_box2d_conversion.h"

#include <iostream>
#include <random>

namespace {

struct SimulationRequest {
  int maxSteps;
  int stride;
  bool noise;
};

class PhyreContactListener : public b2ContactListener {
    private:
      std::vector<::collision::Collision> collisionList;
      int currentTimestep;

      // Noise 
      float directionNoise;
      float elasticityNoise;
      std::mt19937 gen;
      std::normal_distribution<float> normalDist;
      // Set to track collisions we've already applied noise to
      std::set<std::pair<int, int>> processedCollisions;

    public:
      PhyreContactListener(float noiseCollisionDirection = 0.0f, float noiseCollisionElasticity = 0.0f) 
        : currentTimestep(0)
        , directionNoise(noiseCollisionDirection)
        , elasticityNoise(noiseCollisionElasticity)
        , gen(std::random_device{}())
        , normalDist(0.0f, 1.0f) {
      }

      void BeginContact(b2Contact* contact) {
        b2Body* body1 = static_cast<b2Body*>(contact->GetFixtureA()->GetBody());
        b2Body* body2 = static_cast<b2Body*>(contact->GetFixtureB()->GetBody());
        Box2dData* body1_data = static_cast<Box2dData*>(body1->GetUserData());
        Box2dData* body2_data = static_cast<Box2dData*>(body2->GetUserData());

        if (body1->GetType() != 2 || body2->GetType() != 2){
          return;
        }
        
        ::collision::Collision collision_record;
        collision_record.bodyId1 = body1_data->object_id;
        collision_record.bodyId2 = body2_data->object_id;
        collision_record.timestep = currentTimestep;
        collisionList.push_back(collision_record);
      }

      void EndContact(b2Contact* contact) override {
        b2Body* body1 = static_cast<b2Body*>(contact->GetFixtureA()->GetBody());
        b2Body* body2 = static_cast<b2Body*>(contact->GetFixtureB()->GetBody());
        Box2dData* body1_data = static_cast<Box2dData*>(body1->GetUserData());
        Box2dData* body2_data = static_cast<Box2dData*>(body2->GetUserData());

        if (body1->GetType() != 2 || body2->GetType() != 2){
          return;
        }

        // std::cout << currentTimestep << " Contact end " << body1_data->object_id << "-" << body2_data->object_id << std::endl;

        processedCollisions.erase(std::make_pair(body1_data->object_id, body2_data->object_id));
      }

      void PreSolve(b2Contact* contact, const b2Manifold* oldManifold) override {
        if (directionNoise == 0.0f && elasticityNoise == 0.0f) {
            return;  // Skip if no noise is configured
        }

        b2Body* body1 = static_cast<b2Body*>(contact->GetFixtureA()->GetBody());
        b2Body* body2 = static_cast<b2Body*>(contact->GetFixtureB()->GetBody());
        Box2dData* body1_data = static_cast<Box2dData*>(body1->GetUserData());
        Box2dData* body2_data = static_cast<Box2dData*>(body2->GetUserData());

        if (body1->GetType() != 2 || body2->GetType() != 2){
          return;
        }

        auto collisionPair = std::make_pair(body1_data->object_id, body2_data->object_id);
        // Check if we've already processed this collision
        if (processedCollisions.find(collisionPair) != processedCollisions.end()) {
          return;  // Skip if we've already added noise to this collision
        }
        
        // Add the collision to our processed set
        processedCollisions.insert(collisionPair);
        
        // std::cout << currentTimestep << " Adding noise " << body1_data->object_id << "-" << body2_data->object_id << std::endl;

        // Get collision normal
        b2WorldManifold worldManifold;
        contact->GetWorldManifold(&worldManifold);
        b2Vec2 normal = worldManifold.normal;

        // std::cout << normal.x << ", " << normal.y << std::endl<< std::endl;
        

        // Add noise to collision normal direction
        if (directionNoise > 0.0f) {
            float angleNoise = directionNoise * normalDist(gen);
            float cos_theta = std::cos(angleNoise);
            float sin_theta = std::sin(angleNoise);
            b2Vec2 noisyNormal(
                normal.x * cos_theta - normal.y * sin_theta,
                normal.x * sin_theta + normal.y * cos_theta
            );
            
            // std::cout << noisyNormal.x << ", " << noisyNormal.y << std::endl << std::endl;

            // Modify the manifold normal
            b2Manifold* manifold = contact->GetManifold();
            manifold->localNormal = contact->GetFixtureA()->GetBody()->GetLocalVector(noisyNormal);
        }

        // Add noise to restitution
        if (elasticityNoise > 0.0f) {
            b2Fixture* fixtureA = contact->GetFixtureA();
            b2Fixture* fixtureB = contact->GetFixtureB();
            
            float baseRestitution = std::max(fixtureA->GetRestitution(), fixtureB->GetRestitution());
            float noisyRestitution = baseRestitution + elasticityNoise * normalDist(gen);
            noisyRestitution = std::max(0.0f, std::min(1.0f, noisyRestitution));
            
            contact->SetRestitution(noisyRestitution);
        }
      }

      void UpdateTimestep(int timestep) {
        currentTimestep = timestep;
      }

      const std::vector<::collision::Collision>& GetCollisionList() const {
        return collisionList;
      }

      void ClearHistory() {
        collisionList.clear();
      }

      void SetNoiseParameters(float direction_noise, float elasticity_noise) {
          directionNoise = direction_noise;
          elasticityNoise = elasticity_noise;
      }
};

// Runs simulation for the scene. If task is not nullptr, is-task-solved checks
// are performed.
::task::TaskSimulation simulateTask(const ::scene::Scene &scene,
                                    const SimulationRequest &request,
                                    const ::task::Task *task) {
  std::unique_ptr<b2WorldWithData> world = convertSceneToBox2dWorld(scene);

  unsigned int continuousSolvedCount = 0;
  std::vector<::scene::Scene> scenes;
  std::vector<bool> solveStateList;
  bool solved = false;
  int step = 0;

  // For different relations number of steps the condition should hold varies.
  // For NOT_TOUCHING relation one of three should be true:
  //   1. Objects are touching at the beginning and then not touching for
  //   kStepsForSolution steps.
  //   2. Objects are not touching at the beginning, touching at some point of
  //   simulation and then not touching for kStepsForSolution steps.
  //   3. Objects are not touching whole sumulation.
  // For TOUCHING_BRIEFLY a single touching is allowed.
  // For all other relations the condition must hold for kStepsForSolution
  // consequent steps.
  bool lookingForSolution =
      (task == nullptr || !isTaskInSolvedState(*task, *world) ||
       task->relationships.size() != 1 ||
       task->relationships[0] != ::task::SpatialRelationship::NOT_TOUCHING);
  const bool allowInstantSolution =
      (task != nullptr && task->relationships.size() == 1 &&
       task->relationships[0] == ::task::SpatialRelationship::TOUCHING_BRIEFLY);

  PhyreContactListener contactListener;
  world->SetContactListener(&contactListener);
  if (request.noise == true) {
    contactListener.SetNoiseParameters(0.1f, 0.1f);
  }
  
  for (; step < request.maxSteps; step++) {
    // Instruct the world to perform a single step of simulation.
    // It is generally best to keep the time step and iterations fixed.
    world->Step(kTimeStep, kVelocityIterations, kPositionIterations);
    contactListener.UpdateTimestep(step);
    if (request.stride > 0 && step % request.stride == 0) {
      scenes.push_back(updateSceneFromWorld(scene, *world));
    }
    if (task == nullptr) {
      solveStateList.push_back(false);
    } else {
      solveStateList.push_back(isTaskInSolvedState(*task, *world));
      if (solveStateList.back()) {
        continuousSolvedCount++;
        if (lookingForSolution) {
          if (continuousSolvedCount >= kStepsForSolution ||
              allowInstantSolution) {
            solved = true;
            break;
          }
        }
      } else {
        lookingForSolution = true;  // Task passed through non-solved state.
        continuousSolvedCount = 0;
      }
    }
  }

  if (!lookingForSolution && continuousSolvedCount == solveStateList.size()) {
    // See condition 3) for NOT_TOUCHING relation above.
    solved = true;
  }

  {
    std::vector<bool> stridedSolveStateList;
    if (request.stride > 0) {
      for (size_t i = 0; i < solveStateList.size(); i += request.stride) {
        stridedSolveStateList.push_back(solveStateList[i]);
      }
    }
    stridedSolveStateList.swap(solveStateList);
  }

  ::task::TaskSimulation taskSimulation;
  taskSimulation.__set_sceneList(scenes);
  taskSimulation.__set_stepsSimulated(step);
  if (task != nullptr) {
    taskSimulation.__set_solvedStateList(solveStateList);
    taskSimulation.__set_isSolution(solved);
    taskSimulation.__set_collisionList(contactListener.GetCollisionList());
  }

  return taskSimulation;
}
}  // namespace

std::vector<::scene::Scene> simulateScene(const ::scene::Scene &scene,
                                          const int num_steps) {
  const SimulationRequest request{num_steps, 1, false};
  const auto simulation = simulateTask(scene, request, /*task=*/nullptr);
  return simulation.sceneList;
}

::task::TaskSimulation simulateTask(const ::task::Task &task,
                                    const int num_steps, const int stride, const bool noise) {
  const SimulationRequest request{num_steps, stride, noise};
  return simulateTask(task.scene, request, &task);
}
