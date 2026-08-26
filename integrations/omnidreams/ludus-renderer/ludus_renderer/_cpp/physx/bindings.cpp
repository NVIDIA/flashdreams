// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <PxPhysicsAPI.h>
#include <extensions/PxDefaultAllocator.h>
#include <extensions/PxDefaultCpuDispatcher.h>
#include <extensions/PxDefaultErrorCallback.h>
#include <extensions/PxDefaultSimulationFilterShader.h>
#include <extensions/PxExtensionsAPI.h>
#include <extensions/PxRigidBodyExt.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace physx;

namespace {

constexpr std::size_t kStateWidth = 13;
constexpr std::size_t kTrackStateWidth = 10;
constexpr float kMaxOffRoadYawRad = 0.4363323129985824f;

PxFilterFlags vehicleFilterShader(
    PxFilterObjectAttributes,
    PxFilterData,
    PxFilterObjectAttributes,
    PxFilterData,
    PxPairFlags& pairFlags,
    const void*,
    PxU32)
{
    pairFlags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eDETECT_CCD_CONTACT;
    return PxFilterFlag::eDEFAULT;
}

PxTransform poseFromArray(const py::array_t<float, py::array::c_style>& values)
{
    if (values.ndim() != 1 || values.shape(0) != 7)
        throw std::invalid_argument("pose must have shape [7]");
    const float* p = values.data();
    PxQuat quaternion(p[3], p[4], p[5], p[6]);
    quaternion.normalize();
    return PxTransform(PxVec3(p[0], p[1], p[2]), quaternion);
}

PxVec3 vectorFromArray(
    const py::array_t<float, py::array::c_style>& values,
    const char* name)
{
    if (values.ndim() != 1 || values.shape(0) != 3)
        throw std::invalid_argument(std::string(name) + " must have shape [3]");
    return PxVec3(values.data()[0], values.data()[1], values.data()[2]);
}

struct BodyRecord {
    PxRigidDynamic* actor = nullptr;
    PxMaterial* material = nullptr;
    std::size_t slot = 0;
    PxVec3 halfExtents{0.0f};
    float mass = 0.0f;
    float restitution = 0.0f;
    bool collisionActive = false;
    bool detached = false;
    bool trackVisible = true;
    bool trackDriveEnabled = true;
    bool overlappingEgo = false;
    std::vector<std::int64_t> timestampsUs;
    std::vector<float> positions;
    std::vector<float> orientations;
    double maxExtrapolationUs = -1.0;
    std::array<PxVec3, 4> suspensionMounts{};
    float wheelRadius = 0.0f;
    float suspensionRestLength = 0.0f;
    float suspensionMaxCompression = 0.0f;
    float springStiffness = 0.0f;
    float damperRate = 0.0f;
    float tireFriction = 0.0f;
    float corneringStiffness = 0.0f;
    float rollingResistance = 0.0f;
    float maxEngineForce = 0.0f;
    float maxBrakeForce = 0.0f;
    float maxDriveSpeed = 0.0f;
    bool driveIntentActive = false;
    PxVec3 desiredLinearVelocity{0.0f};
    PxVec3 desiredAngularVelocity{0.0f};
    bool verticalTrackControl = false;
    float targetHeight = 0.0f;
    float targetVerticalVelocity = 0.0f;

    bool hasTrack() const { return !timestampsUs.empty(); }
    bool hasVehicle() const { return wheelRadius > 0.0f; }
};

struct BarrierRecord {
    PxRigidStatic* actor = nullptr;
    PxMaterial* material = nullptr;
    PxVec2 start{0.0f};
    PxVec2 end{0.0f};
    PxVec2 segment{0.0f};
    PxVec2 minimum{0.0f};
    PxVec2 maximum{0.0f};
    float lengthSquared = 0.0f;
    float yaw = 0.0f;
    float thickness = 0.0f;
};

struct TrackSample {
    PxTransform transform;
    PxVec3 velocity{0.0f};
    PxVec3 angularVelocity{0.0f};
};

using StepClock = std::chrono::steady_clock;

double elapsedMs(const StepClock::time_point& begin, const StepClock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

float yawFromQuaternion(const PxQuat& quaternion)
{
    return std::atan2(
        2.0f * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0f - 2.0f * (quaternion.y * quaternion.y + quaternion.z * quaternion.z));
}

float wrappedAngle(float angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

class NativeScene {
public:
    explicit NativeScene(std::size_t capacity)
        : mCapacity(capacity), mStates(capacity * kStateWidth, 0.0f),
          mTrackStates(capacity * kTrackStateWidth, 0.0f),
          mIds(capacity, -1), mActive(capacity, 0),
          mCollisionActive(capacity, 0), mDetached(capacity, 0),
          mStruck(capacity, 0)
    {
        if (capacity == 0)
            throw std::invalid_argument("capacity must be positive");
        mFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, mAllocator, mError);
        if (!mFoundation)
            throw std::runtime_error("PxCreateFoundation failed");
        mPhysics = PxCreatePhysics(
            PX_PHYSICS_VERSION, *mFoundation, PxTolerancesScale(), false, nullptr);
        if (!mPhysics)
            throw std::runtime_error("PxCreatePhysics failed");
        if (!PxInitExtensions(*mPhysics, nullptr))
            throw std::runtime_error("PxInitExtensions failed");
        mExtensionsInitialized = true;
        mDispatcher = PxDefaultCpuDispatcherCreate(2);
        if (!mDispatcher)
            throw std::runtime_error("PxDefaultCpuDispatcherCreate failed");

        PxSceneDesc description(mPhysics->getTolerancesScale());
        description.gravity = PxVec3(0.0f, 0.0f, -9.81f);
        description.cpuDispatcher = mDispatcher;
        description.filterShader = vehicleFilterShader;
        description.broadPhaseType = PxBroadPhaseType::eABP;
        description.solverType = PxSolverType::eTGS;
        description.flags |= PxSceneFlag::eENABLE_CCD;
        mScene = mPhysics->createScene(description);
        if (!mScene)
            throw std::runtime_error("PxPhysics::createScene failed");

        addGround();
    }

    NativeScene(const NativeScene&) = delete;
    NativeScene& operator=(const NativeScene&) = delete;

    ~NativeScene() { close(); }

    std::size_t addBody(
        std::int64_t objectId,
        const py::array_t<float, py::array::c_style>& halfExtents,
        const py::array_t<float, py::array::c_style>& chassisHalfExtents,
        const py::array_t<float, py::array::c_style>& chassisOffset,
        float mass,
        float friction,
        float restitution,
        const py::array_t<float, py::array::c_style>& pose,
        const py::array_t<float, py::array::c_style>& linearVelocity,
        const py::array_t<float, py::array::c_style>& angularVelocity,
        bool kinematic,
        bool collisionEnabled,
        const py::array_t<float, py::array::c_style>& suspensionMounts,
        float wheelRadius,
        float suspensionRestLength,
        float suspensionMaxCompression,
        float springStiffness,
        float damperRate,
        float tireFriction,
        float corneringStiffness,
        float rollingResistance,
        float maxEngineForce,
        float maxBrakeForce,
        float maxDriveSpeed)
    {
        ensureOpen();
        if (mBodies.count(objectId))
            throw std::invalid_argument("body id already exists");
        if (mass <= 0.0f)
            throw std::invalid_argument("mass must be positive");
        const PxVec3 half = vectorFromArray(halfExtents, "half_extents");
        const PxVec3 chassisHalf = vectorFromArray(
            chassisHalfExtents, "chassis_half_extents");
        const PxVec3 chassisCenter = vectorFromArray(chassisOffset, "chassis_offset");
        if (half.x <= 0.0f || half.y <= 0.0f || half.z <= 0.0f)
            throw std::invalid_argument("half_extents must be positive");
        if (chassisHalf.x <= 0.0f || chassisHalf.y <= 0.0f || chassisHalf.z <= 0.0f)
            throw std::invalid_argument("chassis_half_extents must be positive");
        const bool hasVehicle = suspensionMounts.ndim() == 2
            && suspensionMounts.shape(0) == 4 && suspensionMounts.shape(1) == 3;
        const bool noVehicle = suspensionMounts.ndim() == 2
            && suspensionMounts.shape(0) == 0 && suspensionMounts.shape(1) == 3;
        if (!hasVehicle && !noVehicle)
            throw std::invalid_argument("suspension_mounts must have shape [4, 3] or [0, 3]");
        if (hasVehicle
            && (wheelRadius <= 0.0f || suspensionRestLength <= 0.0f
                || suspensionMaxCompression <= 0.0f || springStiffness <= 0.0f
                || damperRate < 0.0f || tireFriction < 0.0f
                || corneringStiffness < 0.0f || rollingResistance < 0.0f
                || maxEngineForce <= 0.0f || maxBrakeForce <= 0.0f))
            throw std::invalid_argument("vehicle wheel and suspension parameters are invalid");
        if (!std::isfinite(maxDriveSpeed) || maxDriveSpeed < 0.0f)
            throw std::invalid_argument(
                "max_drive_speed must be finite and non-negative");

        const std::size_t slot = allocateSlot();
        PxMaterial* material = mPhysics->createMaterial(friction, friction, restitution);
        PxRigidDynamic* actor = mPhysics->createRigidDynamic(poseFromArray(pose));
        if (!material || !actor) {
            if (actor)
                actor->release();
            if (material)
                material->release();
            releaseSlot(slot);
            throw std::runtime_error("failed to create PhysX body");
        }
        PxShape* shape = mPhysics->createShape(
            PxBoxGeometry(chassisHalf), *material, true);
        if (!shape) {
            actor->release();
            material->release();
            releaseSlot(slot);
            throw std::runtime_error("failed to create PhysX box shape");
        }
        shape->setLocalPose(PxTransform(chassisCenter));
        shape->setRestOffset(0.01f);
        shape->setContactOffset(0.04f);
        shape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, collisionEnabled);
        actor->attachShape(*shape);
        shape->release();
        actor->setMass(mass);
        actor->setMassSpaceInertiaTensor(PxVec3(
            mass * (half.y * half.y + half.z * half.z) / 3.0f,
            mass * (half.x * half.x + half.z * half.z) / 3.0f,
            mass * (half.x * half.x + half.y * half.y) / 3.0f));
        actor->setSolverIterationCounts(8, 2);
        actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, !kinematic);
        actor->setRigidBodyFlag(
            PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, !kinematic);
        actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, kinematic);
        actor->setLinearVelocity(vectorFromArray(linearVelocity, "linear_velocity"));
        actor->setAngularVelocity(vectorFromArray(angularVelocity, "angular_velocity"));
        mScene->addActor(*actor);
        BodyRecord record;
        record.actor = actor;
        record.material = material;
        record.slot = slot;
        record.halfExtents = half;
        record.mass = mass;
        record.restitution = restitution;
        record.collisionActive = collisionEnabled;
        record.maxDriveSpeed = maxDriveSpeed;
        if (hasVehicle) {
            const float* mounts = suspensionMounts.data();
            for (std::size_t index = 0; index < record.suspensionMounts.size(); ++index)
                record.suspensionMounts[index] = PxVec3(
                    mounts[index * 3], mounts[index * 3 + 1], mounts[index * 3 + 2]);
            record.wheelRadius = wheelRadius;
            record.suspensionRestLength = suspensionRestLength;
            record.suspensionMaxCompression = suspensionMaxCompression;
            record.springStiffness = springStiffness;
            record.damperRate = damperRate;
            record.tireFriction = tireFriction;
            record.corneringStiffness = corneringStiffness;
            record.rollingResistance = rollingResistance;
            record.maxEngineForce = maxEngineForce;
            record.maxBrakeForce = maxBrakeForce;
        }
        mBodies.emplace(objectId, std::move(record));
        mIds[slot] = objectId;
        mActive[slot] = 1;
        mCollisionActive[slot] = collisionEnabled ? 1 : 0;
        writeState(mBodies.at(objectId));
        return slot;
    }

    void setBodyTrack(
        std::int64_t objectId,
        const py::array_t<std::int64_t, py::array::c_style>& timestamps,
        const py::array_t<float, py::array::c_style>& positions,
        const py::array_t<float, py::array::c_style>& orientations,
        double maxExtrapolationUs)
    {
        BodyRecord& body = bodyAt(objectId);
        if (timestamps.ndim() != 1 || timestamps.shape(0) == 0)
            throw std::invalid_argument("timestamps must have shape [samples]");
        const py::ssize_t count = timestamps.shape(0);
        if (positions.ndim() != 2 || positions.shape(0) != count || positions.shape(1) != 3)
            throw std::invalid_argument("positions must have shape [samples, 3]");
        if (orientations.ndim() != 2 || orientations.shape(0) != count || orientations.shape(1) != 4)
            throw std::invalid_argument("orientations must have shape [samples, 4]");
        body.timestampsUs.assign(timestamps.data(), timestamps.data() + count);
        body.positions.assign(positions.data(), positions.data() + count * 3);
        body.orientations.assign(orientations.data(), orientations.data() + count * 4);
        body.maxExtrapolationUs = maxExtrapolationUs;
    }

    void updateBody(
        std::int64_t objectId,
        const py::array_t<float, py::array::c_style>& pose,
        const py::array_t<float, py::array::c_style>& linearVelocity,
        const py::array_t<float, py::array::c_style>& angularVelocity,
        bool kinematic)
    {
        BodyRecord& body = bodyAt(objectId);
        PxRigidDynamic* actor = body.actor;
        const bool wasKinematic = actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC);
        if (wasKinematic != kinematic) {
            if (kinematic) {
                actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, false);
                actor->setRigidBodyFlag(
                    PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, false);
                actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
            } else {
                actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, false);
                actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
                actor->setRigidBodyFlag(
                    PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
            }
        }
        const PxTransform transform = poseFromArray(pose);
        if (kinematic && wasKinematic)
            actor->setKinematicTarget(transform);
        else
            actor->setGlobalPose(transform, false);
        actor->setLinearVelocity(vectorFromArray(linearVelocity, "linear_velocity"));
        actor->setAngularVelocity(vectorFromArray(angularVelocity, "angular_velocity"));
        actor->wakeUp();
        writeState(body);
    }

    void removeBody(std::int64_t objectId)
    {
        auto found = mBodies.find(objectId);
        if (found == mBodies.end())
            return;
        BodyRecord body = std::move(found->second);
        mScene->removeActor(*body.actor);
        body.actor->release();
        body.material->release();
        releaseSlot(body.slot);
        mBodies.erase(found);
    }

    void setBodyCollisionEnabled(std::int64_t objectId, bool enabled)
    {
        BodyRecord& body = bodyAt(objectId);
        setCollisionEnabled(body, enabled);
    }

    void setBodyTrackDriveEnabled(std::int64_t objectId, bool enabled)
    {
        BodyRecord& body = bodyAt(objectId);
        body.trackDriveEnabled = enabled;
        if (!enabled) {
            body.driveIntentActive = false;
            body.verticalTrackControl = false;
        }
    }

    void setBodyDetached(std::int64_t objectId, bool detached)
    {
        BodyRecord& body = bodyAt(objectId);
        body.detached = detached;
        mDetached[body.slot] = detached ? 1 : 0;
    }

    void setBodyTrackControls(
        const py::array_t<std::int64_t, py::array::c_style>& objectIds,
        const py::array_t<std::uint8_t, py::array::c_style>& driveEnabled,
        const py::array_t<std::uint8_t, py::array::c_style>& detached)
    {
        if (objectIds.ndim() != 1 || driveEnabled.ndim() != 1 || detached.ndim() != 1)
            throw std::invalid_argument("track-control arrays must be one-dimensional");
        const py::ssize_t count = objectIds.shape(0);
        if (driveEnabled.shape(0) != count || detached.shape(0) != count)
            throw std::invalid_argument("track-control arrays must have equal lengths");
        for (py::ssize_t index = 0; index < count; ++index) {
            BodyRecord& body = bodyAt(objectIds.data()[index]);
            const bool enabled = driveEnabled.data()[index] != 0;
            body.trackDriveEnabled = enabled;
            if (!enabled) {
                body.driveIntentActive = false;
                body.verticalTrackControl = false;
            }
            body.detached = detached.data()[index] != 0;
            mDetached[body.slot] = body.detached ? 1 : 0;
        }
    }

    void setCollisionEnabled(BodyRecord& body, bool enabled)
    {
        if (body.collisionActive == enabled)
            return;
        PxShape* shape = nullptr;
        if (body.actor->getShapes(&shape, 1) != 1 || !shape)
            throw std::runtime_error("PhysX body has no collision shape");
        shape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, enabled);
        body.collisionActive = enabled;
        mCollisionActive[body.slot] = enabled ? 1 : 0;
    }

    void addBarrier(
        std::int64_t barrierId,
        const py::array_t<float, py::array::c_style>& start,
        const py::array_t<float, py::array::c_style>& end,
        float thickness,
        float height,
        float friction,
        float restitution)
    {
        ensureOpen();
        if (mBarriers.count(barrierId))
            throw std::invalid_argument("barrier id already exists");
        if (start.ndim() != 1 || start.shape(0) != 2 || end.ndim() != 1 || end.shape(0) != 2)
            throw std::invalid_argument("barrier endpoints must have shape [2]");
        const float dx = end.data()[0] - start.data()[0];
        const float dy = end.data()[1] - start.data()[1];
        const float length = std::sqrt(dx * dx + dy * dy);
        if (length <= 1.0e-4f || thickness <= 0.0f || height <= 0.0f)
            throw std::invalid_argument("barrier dimensions must be positive");
        const float yaw = std::atan2(dy, dx);
        const PxTransform transform(
            PxVec3(
                (start.data()[0] + end.data()[0]) * 0.5f,
                (start.data()[1] + end.data()[1]) * 0.5f,
                height * 0.5f),
            PxQuat(yaw, PxVec3(0.0f, 0.0f, 1.0f)));
        PxMaterial* material = mPhysics->createMaterial(friction, friction, restitution);
        PxRigidStatic* actor = mPhysics->createRigidStatic(transform);
        PxShape* shape = material
            ? mPhysics->createShape(
                  PxBoxGeometry(length * 0.5f, thickness * 0.5f, height * 0.5f),
                  *material,
                  true)
            : nullptr;
        if (!material || !actor || !shape) {
            if (shape)
                shape->release();
            if (actor)
                actor->release();
            if (material)
                material->release();
            throw std::runtime_error("failed to create PhysX barrier");
        }
        actor->attachShape(*shape);
        shape->release();
        mScene->addActor(*actor);
        mBarriers.emplace(
            barrierId,
            BarrierRecord{
                actor,
                material,
                PxVec2(start.data()[0], start.data()[1]),
                PxVec2(end.data()[0], end.data()[1]),
                PxVec2(dx, dy),
                PxVec2(
                    std::min(start.data()[0], end.data()[0]),
                    std::min(start.data()[1], end.data()[1])),
                PxVec2(
                    std::max(start.data()[0], end.data()[0]),
                    std::max(start.data()[1], end.data()[1])),
                length * length,
                yaw,
                thickness});
    }

    void removeBarrier(std::int64_t barrierId)
    {
        auto found = mBarriers.find(barrierId);
        if (found == mBarriers.end())
            return;
        mScene->removeActor(*found->second.actor);
        found->second.actor->release();
        found->second.material->release();
        mBarriers.erase(found);
    }

    void step(float dt)
    {
        ensureOpen();
        if (!(dt > 0.0f))
            throw std::invalid_argument("dt must be positive");
        {
            py::gil_scoped_release release;
            simulateSubsteps(dt);
        }
        for (const auto& entry : mBodies)
            writeState(entry.second);
    }

    py::tuple stepTracked(
        const py::array_t<float, py::array::c_style>& egoPose,
        const py::array_t<float, py::array::c_style>& egoLinearVelocity,
        const py::array_t<float, py::array::c_style>& egoAngularVelocity,
        std::int64_t timestampUs,
        float dt,
        bool actorCollisionEnabled)
    {
        ensureOpen();
        if (!(dt > 0.0f))
            throw std::invalid_argument("dt must be positive");
        const PxTransform requestedEgoPose = poseFromArray(egoPose);
        const PxVec3 requestedEgoLinear = vectorFromArray(egoLinearVelocity, "linear_velocity");
        const PxVec3 requestedEgoAngular = vectorFromArray(egoAngularVelocity, "angular_velocity");
        BodyRecord& ego = bodyAt(0);
        bool impact = false;
        std::size_t visibleCount = 0;
        std::size_t detachedCount = 0;
        double actorUpdateMs = 0.0;
        double solverMs = 0.0;
        double readbackMs = 0.0;
        std::fill(mStruck.begin(), mStruck.end(), 0);

        {
            py::gil_scoped_release release;
            const auto actorBegin = StepClock::now();
            if (!mEgoPoseInitialized) {
                updateActorState(
                    ego,
                    requestedEgoPose,
                    requestedEgoLinear,
                    requestedEgoAngular,
                    false);
                mEgoPoseInitialized = true;
            } else {
                applyDriveIntent(
                    ego, requestedEgoLinear, requestedEgoAngular, dt);
            }
            for (auto& entry : mBodies) {
                BodyRecord& body = entry.second;
                if (!body.hasTrack())
                    continue;
                if (!isTrackVisible(body, timestampUs)) {
                    body.trackVisible = false;
                    body.overlappingEgo = false;
                    body.driveIntentActive = false;
                    body.verticalTrackControl = false;
                    setCollisionEnabled(body, false);
                    continue;
                }
                body.trackVisible = true;
                ++visibleCount;
                const TrackSample track = sampleTrack(body, timestampUs);
                writeTrackState(body, track);
                const PxTransform actorTransform = body.actor->getGlobalPose();
                const PxVec3 actorVelocity = body.actor->getLinearVelocity();
                const bool overlapsEgo = actorCollisionEnabled
                    && overlaps2d(
                        requestedEgoPose,
                        ego.halfExtents,
                        actorTransform,
                        body.halfExtents);
                const PxVec3 separation = actorTransform.p - requestedEgoPose.p;
                const PxVec3 relativeVelocity = actorVelocity - requestedEgoLinear;
                if (
                    overlapsEgo
                    && !body.overlappingEgo
                    && relativeVelocity.dot(separation) < 0.0f) {
                    body.detached = true;
                    mStruck[body.slot] = 1;
                    impact = true;
                }
                body.overlappingEgo = overlapsEgo;
                if (body.trackDriveEnabled)
                    driveTowardsTrack(body, track, dt);
                else {
                    body.driveIntentActive = false;
                    body.verticalTrackControl = false;
                }
                setCollisionEnabled(body, actorCollisionEnabled);
                mDetached[body.slot] = body.detached ? 1 : 0;
                if (body.detached)
                    ++detachedCount;
            }
            const auto actorEnd = StepClock::now();

            simulateSubsteps(dt);
            const auto solverEnd = StepClock::now();

            for (const auto& entry : mBodies) {
                const BodyRecord& body = entry.second;
                if (!body.hasTrack() || body.trackVisible || body.detached)
                    writeState(body);
            }
            PxVec3 rebound;
            if (barrierReboundVelocity(requestedEgoPose, requestedEgoLinear, ego, rebound)) {
                float* output = mStates.data() + ego.slot * kStateWidth;
                output[7] = rebound.x;
                output[8] = rebound.y;
                output[9] = rebound.z;
                impact = true;
            }
            const auto readbackEnd = StepClock::now();
            actorUpdateMs = elapsedMs(actorBegin, actorEnd);
            solverMs = elapsedMs(actorEnd, solverEnd);
            readbackMs = elapsedMs(solverEnd, readbackEnd);
        }
        return py::make_tuple(
            impact,
            actorUpdateMs,
            solverMs,
            readbackMs,
            visibleCount,
            detachedCount);
    }

    py::array stateBuffer()
    {
        return py::array_t<float>(
            {static_cast<py::ssize_t>(mCapacity), static_cast<py::ssize_t>(kStateWidth)},
            {static_cast<py::ssize_t>(kStateWidth * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
            mStates.data(),
            py::cast(this, py::return_value_policy::reference));
    }

    py::array trackStateBuffer()
    {
        return py::array_t<float>(
            {static_cast<py::ssize_t>(mCapacity), static_cast<py::ssize_t>(kTrackStateWidth)},
            {static_cast<py::ssize_t>(kTrackStateWidth * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
            mTrackStates.data(),
            py::cast(this, py::return_value_policy::reference));
    }

    py::array idBuffer()
    {
        return py::array_t<std::int64_t>(
            mCapacity,
            mIds.data(),
            py::cast(this, py::return_value_policy::reference));
    }

    py::array activeBuffer()
    {
        return py::array_t<std::uint8_t>(
            mCapacity,
            mActive.data(),
            py::cast(this, py::return_value_policy::reference));
    }

    py::array collisionActiveBuffer()
    {
        return py::array_t<std::uint8_t>(
            mCapacity,
            mCollisionActive.data(),
            py::cast(this, py::return_value_policy::reference));
    }

    py::array detachedBuffer()
    {
        return py::array_t<std::uint8_t>(
            mCapacity,
            mDetached.data(),
            py::cast(this, py::return_value_policy::reference));
    }

    py::array struckBuffer()
    {
        return py::array_t<std::uint8_t>(
            mCapacity,
            mStruck.data(),
            py::cast(this, py::return_value_policy::reference));
    }

    std::size_t bodyCount() const { return mBodies.size(); }
    std::size_t barrierCount() const { return mBarriers.size(); }

    void close()
    {
        if (!mPhysics)
            return;
        for (auto& entry : mBodies) {
            entry.second.actor->release();
            entry.second.material->release();
        }
        mBodies.clear();
        for (auto& entry : mBarriers) {
            entry.second.actor->release();
            entry.second.material->release();
        }
        mBarriers.clear();
        if (mGround)
            mGround->release();
        if (mGroundMaterial)
            mGroundMaterial->release();
        if (mScene)
            mScene->release();
        if (mDispatcher)
            mDispatcher->release();
        if (mExtensionsInitialized)
            PxCloseExtensions();
        mPhysics->release();
        mFoundation->release();
        mPhysics = nullptr;
        mFoundation = nullptr;
        mScene = nullptr;
        mDispatcher = nullptr;
        mGround = nullptr;
        mGroundMaterial = nullptr;
    }

private:
    void simulateSubsteps(float dt)
    {
        constexpr float maxSubstepS = 1.0f / 120.0f;
        const std::size_t substepCount = static_cast<std::size_t>(
            std::max(1.0f, std::ceil(dt / maxSubstepS)));
        const float substepDt = dt / static_cast<float>(substepCount);
        for (std::size_t substep = 0; substep < substepCount; ++substep) {
            applyVehicleForces(substepDt);
            mScene->simulate(substepDt);
            if (!mScene->fetchResults(true))
                throw std::runtime_error("PhysX fetchResults failed");
            constrainVehicleYawsAtBarriers();
        }
    }

    void constrainVehicleYawsAtBarriers()
    {
        for (auto& entry : mBodies) {
            BodyRecord& body = entry.second;
            if (!body.hasVehicle()
                || body.actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
                continue;

            PxTransform pose = body.actor->getGlobalPose();
            const float yaw = yawFromQuaternion(pose.q);
            const PxVec2 position(pose.p.x, pose.p.y);
            const PxVec2 forward(std::cos(yaw), std::sin(yaw));
            const PxVec2 left(-forward.y, forward.x);
            const BarrierRecord* nearestBoundary = nullptr;
            float nearestClearance = std::numeric_limits<float>::max();
            const float bodyRadius = std::sqrt(
                body.halfExtents.x * body.halfExtents.x
                + body.halfExtents.y * body.halfExtents.y) + 0.05f;
            for (const auto& barrierEntry : mBarriers) {
                const BarrierRecord& barrier = barrierEntry.second;
                const float broadPhaseRadius =
                    bodyRadius + barrier.thickness * 0.5f;
                if (position.x < barrier.minimum.x - broadPhaseRadius
                    || position.x > barrier.maximum.x + broadPhaseRadius
                    || position.y < barrier.minimum.y - broadPhaseRadius
                    || position.y > barrier.maximum.y + broadPhaseRadius)
                    continue;
                const float alpha = std::clamp(
                    (position - barrier.start).dot(barrier.segment)
                        / barrier.lengthSquared,
                    0.0f,
                    1.0f);
                const PxVec2 offset =
                    position - (barrier.start + barrier.segment * alpha);
                const float distance = offset.magnitude();
                const PxVec2 normal = distance > 1.0e-6f
                    ? offset / distance
                    : PxVec2(-barrier.segment.y, barrier.segment.x).getNormalized();
                const float support = std::abs(normal.dot(forward)) * body.halfExtents.x
                    + std::abs(normal.dot(left)) * body.halfExtents.y;
                const float clearance = distance - support - barrier.thickness * 0.5f;
                if (clearance <= 0.05f && clearance < nearestClearance) {
                    nearestBoundary = &barrier;
                    nearestClearance = clearance;
                }
            }
            // A vehicle whose complete footprint clears every road boundary
            // remains free to take any heading while it is fully on the road.
            if (!nearestBoundary)
                continue;

            float yawError = wrappedAngle(yaw - nearestBoundary->yaw);
            if (yawError > 1.5707963267948966f)
                yawError -= 3.1415926535897932f;
            else if (yawError < -1.5707963267948966f)
                yawError += 3.1415926535897932f;
            const float constrainedYawError = std::clamp(
                yawError, -kMaxOffRoadYawRad, kMaxOffRoadYawRad);
            if (constrainedYawError == yawError)
                continue;

            pose.q = PxQuat(
                constrainedYawError - yawError,
                PxVec3(0.0f, 0.0f, 1.0f)) * pose.q;
            pose.q.normalize();
            body.actor->setGlobalPose(pose, false);

            PxVec3 angularVelocity = body.actor->getAngularVelocity();
            if (angularVelocity.z * yawError > 0.0f) {
                angularVelocity.z = 0.0f;
                body.actor->setAngularVelocity(angularVelocity, false);
            }
        }
    }

    void applyVehicleForces(float dt)
    {
        PxQueryFilterData staticQuery;
        staticQuery.flags = PxQueryFlag::eSTATIC;
        for (auto& entry : mBodies) {
            BodyRecord& body = entry.second;
            applyDriveForces(body, dt);
            if (!body.hasVehicle()
                || body.actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
                continue;

            const PxTransform pose = body.actor->getGlobalPose();
            const PxVec3 suspensionUp = pose.q.rotate(PxVec3(0.0f, 0.0f, 1.0f));
            const PxVec3 suspensionDown = -suspensionUp;
            const float rayLength = body.suspensionRestLength + body.wheelRadius;
            for (const PxVec3& localMount : body.suspensionMounts) {
                const PxVec3 mount = pose.transform(localMount);
                PxRaycastBuffer hit;
                if (!mScene->raycast(
                        mount,
                        suspensionDown,
                        rayLength,
                        hit,
                        PxHitFlag::ePOSITION | PxHitFlag::eNORMAL,
                        staticQuery))
                    continue;

                const float suspensionLength = std::max(
                    0.0f, hit.block.distance - body.wheelRadius);
                const float compression = std::clamp(
                    body.suspensionRestLength - suspensionLength,
                    0.0f,
                    body.suspensionMaxCompression);
                if (compression <= 0.0f)
                    continue;

                const PxVec3 mountVelocity = PxRigidBodyExt::getVelocityAtPos(
                    *body.actor, mount);
                const float compressionSpeed = -mountVelocity.dot(suspensionUp);
                const float maxWheelLoad = body.mass * 9.81f;
                const float wheelLoad = std::clamp(
                    body.springStiffness * compression
                        + body.damperRate * compressionSpeed,
                    0.0f,
                    maxWheelLoad);
                if (wheelLoad <= 0.0f)
                    continue;

                PxRigidBodyExt::addForceAtPos(
                    *body.actor,
                    suspensionUp * wheelLoad,
                    mount,
                    PxForceMode::eFORCE);

                const PxVec3 normal = hit.block.normal;
                PxVec3 forward = pose.q.rotate(PxVec3(1.0f, 0.0f, 0.0f));
                forward -= normal * forward.dot(normal);
                if (forward.normalize() <= 1.0e-6f)
                    continue;
                PxVec3 lateral = normal.cross(forward);
                if (lateral.normalize() <= 1.0e-6f)
                    continue;
                const PxVec3 contactVelocity = PxRigidBodyExt::getVelocityAtPos(
                    *body.actor, hit.block.position);
                const float longitudinalSpeed = contactVelocity.dot(forward);
                const float lateralSpeed = contactVelocity.dot(lateral);
                const float slipAngle = std::atan2(
                    lateralSpeed, std::abs(longitudinalSpeed) + 0.5f);
                const float frictionLimit = body.tireFriction * wheelLoad;
                const float lateralForce = std::clamp(
                    -0.25f * body.corneringStiffness * slipAngle,
                    -frictionLimit,
                    frictionLimit);
                const float rollingMagnitude = std::min(
                    body.rollingResistance * wheelLoad,
                    std::sqrt(std::max(
                        0.0f,
                        frictionLimit * frictionLimit - lateralForce * lateralForce)));
                const float rollingForce = std::abs(longitudinalSpeed) > 1.0e-3f
                    ? -std::copysign(rollingMagnitude, longitudinalSpeed)
                    : 0.0f;
                PxRigidBodyExt::addForceAtPos(
                    *body.actor,
                    lateral * lateralForce + forward * rollingForce,
                    hit.block.position,
                    PxForceMode::eFORCE);
            }
        }
    }

    static bool isTrackVisible(const BodyRecord& body, std::int64_t timestampUs)
    {
        if (body.maxExtrapolationUs < 0.0)
            return true;
        const std::int64_t first = body.timestampsUs.front();
        if (timestampUs < first)
            return body.timestampsUs.size() >= 2
                && static_cast<double>(first - timestampUs) <= body.maxExtrapolationUs;
        // The HD-map renderer holds an object's final sample after its track
        // starts. Keep the PhysX shape alive for the same interval so a box
        // that is still rendered cannot silently stop colliding or disappear
        // from the PhysX debug view.
        return true;
    }

    static TrackSample sampleTrack(const BodyRecord& body, std::int64_t timestampUs)
    {
        const auto found = std::lower_bound(
            body.timestampsUs.begin(), body.timestampsUs.end(), timestampUs);
        std::size_t lo = 0;
        std::size_t hi = 0;
        float alpha = 0.0f;
        if (found == body.timestampsUs.begin()) {
            lo = hi = 0;
        } else if (found == body.timestampsUs.end()) {
            lo = hi = body.timestampsUs.size() - 1;
        } else {
            hi = static_cast<std::size_t>(found - body.timestampsUs.begin());
            lo = hi - 1;
            const std::int64_t span = body.timestampsUs[hi] - body.timestampsUs[lo];
            if (span != 0)
                alpha = static_cast<float>(timestampUs - body.timestampsUs[lo])
                    / static_cast<float>(span);
        }
        const float oneMinusAlpha = 1.0f - alpha;
        const float* loPosition = body.positions.data() + lo * 3;
        const float* hiPosition = body.positions.data() + hi * 3;
        const PxVec3 position(
            loPosition[0] * oneMinusAlpha + hiPosition[0] * alpha,
            loPosition[1] * oneMinusAlpha + hiPosition[1] * alpha,
            loPosition[2] * oneMinusAlpha + hiPosition[2] * alpha);
        const float* loQuaternion = body.orientations.data() + lo * 4;
        const float* hiQuaternion = body.orientations.data() + hi * 4;
        PxQuat quaternion(
            loQuaternion[0] * oneMinusAlpha + hiQuaternion[0] * alpha,
            loQuaternion[1] * oneMinusAlpha + hiQuaternion[1] * alpha,
            loQuaternion[2] * oneMinusAlpha + hiQuaternion[2] * alpha,
            loQuaternion[3] * oneMinusAlpha + hiQuaternion[3] * alpha);
        if (quaternion.magnitudeSquared() > 1.0e-16f)
            quaternion.normalize();
        else
            quaternion = PxQuat(PxIdentity);
        PxVec3 velocity(0.0f);
        if (lo != hi) {
            const float dt = static_cast<float>(body.timestampsUs[hi] - body.timestampsUs[lo])
                / 1'000'000.0f;
            if (dt > 0.0f) {
                velocity = PxVec3(
                    hiPosition[0] - loPosition[0],
                    hiPosition[1] - loPosition[1],
                    hiPosition[2] - loPosition[2]);
                velocity *= 1.0f / dt;
            }
        }
        return TrackSample{PxTransform(position, quaternion), velocity, PxVec3(0.0f)};
    }

    static bool overlaps2d(
        const PxTransform& first,
        const PxVec3& firstHalf,
        const PxTransform& second,
        const PxVec3& secondHalf)
    {
        const float firstYaw = yawFromQuaternion(first.q);
        const float secondYaw = yawFromQuaternion(second.q);
        const std::array<PxVec2, 4> axes{
            PxVec2(std::cos(firstYaw), std::sin(firstYaw)),
            PxVec2(-std::sin(firstYaw), std::cos(firstYaw)),
            PxVec2(std::cos(secondYaw), std::sin(secondYaw)),
            PxVec2(-std::sin(secondYaw), std::cos(secondYaw))};
        const PxVec2 delta(second.p.x - first.p.x, second.p.y - first.p.y);
        for (const PxVec2& axis : axes) {
            const float firstRadius = firstHalf.x * std::abs(axes[0].dot(axis))
                + firstHalf.y * std::abs(axes[1].dot(axis));
            const float secondRadius = secondHalf.x * std::abs(axes[2].dot(axis))
                + secondHalf.y * std::abs(axes[3].dot(axis));
            if (std::abs(delta.dot(axis)) > firstRadius + secondRadius)
                return false;
        }
        return true;
    }

    static void applyDriveIntent(
        BodyRecord& body,
        const PxVec3& desiredVelocity,
        const PxVec3& desiredAngularVelocity,
        float)
    {
        body.desiredLinearVelocity = desiredVelocity;
        body.desiredAngularVelocity = desiredAngularVelocity;
        body.driveIntentActive = true;
    }

    static void applyDriveForces(BodyRecord& body, float dt)
    {
        if (!body.driveIntentActive
            || body.actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
            return;
        PxRigidDynamic* actor = body.actor;
        const PxTransform pose = actor->getGlobalPose();
        PxVec3 forward = pose.q.rotate(PxVec3(1.0f, 0.0f, 0.0f));
        forward.z = 0.0f;
        if (forward.normalize() <= 1.0e-6f)
            return;
        const PxVec3 lateral(-forward.y, forward.x, 0.0f);

        const PxVec3 currentVelocity = actor->getLinearVelocity();
        const float currentSpeed = currentVelocity.dot(forward);
        const float desiredSpeed = body.desiredLinearVelocity.dot(forward);
        const float speedError = desiredSpeed - currentSpeed;
        const bool braking = std::abs(desiredSpeed) < std::abs(currentSpeed)
            || desiredSpeed * currentSpeed < 0.0f;
        const float forceLimit = body.hasVehicle()
            ? (braking ? body.maxBrakeForce : body.maxEngineForce)
            : body.mass * 6.5f;
        const float driveForce = std::clamp(
            body.mass * speedError / std::max(dt, 1.0e-3f),
            -forceLimit,
            forceLimit);
        const float lateralForceLimit = body.hasVehicle()
            ? body.tireFriction * body.mass * 9.81f
            : body.mass * 6.5f;
        const float lateralForce = std::clamp(
            body.mass
                * (body.desiredLinearVelocity - currentVelocity).dot(lateral)
                / std::max(dt, 1.0e-3f),
            -lateralForceLimit,
            lateralForceLimit);
        actor->addForce(
            forward * driveForce + lateral * lateralForce,
            PxForceMode::eFORCE,
            true);

        const float yawInertia = actor->getMassSpaceInertiaTensor().z;
        const float unconstrainedYawTorque = yawInertia
            * (body.desiredAngularVelocity.z - actor->getAngularVelocity().z)
            / std::max(dt, 1.0e-3f);
        const float halfWheelBase = body.hasVehicle()
            ? std::max(std::abs(body.suspensionMounts[0].x), 0.5f)
            : 0.5f;
        const float maxYawTorque = body.hasVehicle()
            ? body.tireFriction * body.mass * 9.81f * halfWheelBase
            : yawInertia * 4.0f;
        actor->addTorque(
            PxVec3(
                0.0f,
                0.0f,
                std::clamp(
                    unconstrainedYawTorque, -maxYawTorque, maxYawTorque)),
            PxForceMode::eFORCE,
            true);

        if (body.verticalTrackControl) {
            constexpr float maxVerticalAcceleration = 12.0f;
            const float verticalAcceleration = std::clamp(
                (body.targetHeight - pose.p.z) * 8.0f
                    + (body.targetVerticalVelocity - currentVelocity.z) * 4.0f
                    + 9.81f,
                -maxVerticalAcceleration,
                maxVerticalAcceleration);
            actor->addForce(
                PxVec3(0.0f, 0.0f, verticalAcceleration * body.mass),
                PxForceMode::eFORCE,
                true);
        }
    }

    static void driveTowardsTrack(
        BodyRecord& body, const TrackSample& track, float dt)
    {
        const PxTransform current = body.actor->getGlobalPose();
        PxVec2 correction(
            track.transform.p.x - current.p.x,
            track.transform.p.y - current.p.y);
        correction *= 1.5f;
        const float correctionSpeed = correction.magnitude();
        if (correctionSpeed > 8.0f)
            correction *= 8.0f / correctionSpeed;
        PxVec3 desiredVelocity = track.velocity;
        desiredVelocity.x += correction.x;
        desiredVelocity.y += correction.y;
        const float desiredHorizontalSpeed = std::sqrt(
            desiredVelocity.x * desiredVelocity.x
            + desiredVelocity.y * desiredVelocity.y);
        if (body.maxDriveSpeed > 0.0f
            && desiredHorizontalSpeed > body.maxDriveSpeed) {
            const float scale = body.maxDriveSpeed / desiredHorizontalSpeed;
            desiredVelocity.x *= scale;
            desiredVelocity.y *= scale;
        }

        const float headingError = wrappedAngle(
            yawFromQuaternion(track.transform.q) - yawFromQuaternion(current.q));
        PxVec3 desiredAngularVelocity = track.angularVelocity;
        desiredAngularVelocity.z = std::clamp(
            headingError * 3.0f, -1.5f, 1.5f);
        applyDriveIntent(body, desiredVelocity, desiredAngularVelocity, dt);

        if (!body.hasVehicle() || !body.detached) {
            body.verticalTrackControl = true;
            body.targetHeight = track.transform.p.z;
            body.targetVerticalVelocity = track.velocity.z;
        } else
            body.verticalTrackControl = false;
    }

    static void updateActorState(
        BodyRecord& body,
        const PxTransform& transform,
        const PxVec3& linearVelocity,
        const PxVec3& angularVelocity,
        bool kinematic)
    {
        PxRigidDynamic* actor = body.actor;
        const bool wasKinematic = actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC);
        if (wasKinematic != kinematic) {
            if (kinematic) {
                actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, false);
                actor->setRigidBodyFlag(
                    PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, false);
                actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
            } else {
                actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, false);
                actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
                actor->setRigidBodyFlag(
                    PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
            }
        }
        if (kinematic && wasKinematic)
            actor->setKinematicTarget(transform);
        else
            actor->setGlobalPose(transform, false);
        actor->setLinearVelocity(linearVelocity);
        actor->setAngularVelocity(angularVelocity);
        actor->wakeUp();
    }

    bool barrierReboundVelocity(
        const PxTransform& egoTransform,
        const PxVec3& egoVelocity,
        const BodyRecord& ego,
        PxVec3& rebound) const
    {
        const PxVec2 position(egoTransform.p.x, egoTransform.p.y);
        const float yaw = yawFromQuaternion(egoTransform.q);
        const PxVec2 forward(std::cos(yaw), std::sin(yaw));
        const PxVec2 left(-forward.y, forward.x);
        const float egoRadius = std::sqrt(
            ego.halfExtents.x * ego.halfExtents.x
            + ego.halfExtents.y * ego.halfExtents.y) + 0.05f;
        for (const auto& entry : mBarriers) {
            const BarrierRecord& barrier = entry.second;
            const float broadPhaseRadius = egoRadius + barrier.thickness * 0.5f;
            if (position.x < barrier.minimum.x - broadPhaseRadius
                || position.x > barrier.maximum.x + broadPhaseRadius
                || position.y < barrier.minimum.y - broadPhaseRadius
                || position.y > barrier.maximum.y + broadPhaseRadius)
                continue;
            const float alpha = std::clamp(
                (position - barrier.start).dot(barrier.segment)
                    / barrier.lengthSquared,
                0.0f,
                1.0f);
            const PxVec2 offset =
                position - (barrier.start + barrier.segment * alpha);
            const float distance = offset.magnitude();
            PxVec2 normal;
            if (distance > 1.0e-6f) {
                normal = offset / distance;
            } else {
                const PxVec2 horizontalVelocity(egoVelocity.x, egoVelocity.y);
                const float speed = horizontalVelocity.magnitude();
                normal = speed > 1.0e-6f
                    ? -horizontalVelocity / speed
                    : PxVec2(1.0f, 0.0f);
            }
            const float support = std::abs(normal.dot(forward)) * ego.halfExtents.x
                + std::abs(normal.dot(left)) * ego.halfExtents.y;
            if (distance > support + barrier.thickness * 0.5f + 0.05f)
                continue;
            const float normalSpeed = egoVelocity.x * normal.x + egoVelocity.y * normal.y;
            rebound = egoVelocity;
            if (normalSpeed < 0.0f) {
                rebound.x -= (1.0f + ego.restitution) * normalSpeed * normal.x;
                rebound.y -= (1.0f + ego.restitution) * normalSpeed * normal.y;
            }
            return true;
        }
        return false;
    }

    void ensureOpen() const
    {
        if (!mPhysics || !mScene)
            throw std::runtime_error("PhysX scene is closed");
    }

    std::size_t allocateSlot()
    {
        if (!mFreeSlots.empty()) {
            const std::size_t slot = mFreeSlots.back();
            mFreeSlots.pop_back();
            return slot;
        }
        if (mNextSlot >= mCapacity)
            throw std::runtime_error("PhysX body capacity exceeded");
        return mNextSlot++;
    }

    void releaseSlot(std::size_t slot)
    {
        mIds[slot] = -1;
        mActive[slot] = 0;
        mCollisionActive[slot] = 0;
        mDetached[slot] = 0;
        mStruck[slot] = 0;
        std::fill_n(mStates.data() + slot * kStateWidth, kStateWidth, 0.0f);
        std::fill_n(
            mTrackStates.data() + slot * kTrackStateWidth,
            kTrackStateWidth,
            0.0f);
        mFreeSlots.push_back(slot);
    }

    BodyRecord& bodyAt(std::int64_t objectId)
    {
        auto found = mBodies.find(objectId);
        if (found == mBodies.end())
            throw std::out_of_range("unknown body id");
        return found->second;
    }

    void writeState(const BodyRecord& body)
    {
        const PxTransform pose = body.actor->getGlobalPose();
        const PxVec3 linear = body.actor->getLinearVelocity();
        const PxVec3 angular = body.actor->getAngularVelocity();
        float* output = mStates.data() + body.slot * kStateWidth;
        output[0] = pose.p.x;
        output[1] = pose.p.y;
        output[2] = pose.p.z;
        output[3] = pose.q.x;
        output[4] = pose.q.y;
        output[5] = pose.q.z;
        output[6] = pose.q.w;
        output[7] = linear.x;
        output[8] = linear.y;
        output[9] = linear.z;
        output[10] = angular.x;
        output[11] = angular.y;
        output[12] = angular.z;
    }

    void writeTrackState(const BodyRecord& body, const TrackSample& track)
    {
        float* output = mTrackStates.data() + body.slot * kTrackStateWidth;
        output[0] = track.transform.p.x;
        output[1] = track.transform.p.y;
        output[2] = track.transform.p.z;
        output[3] = track.transform.q.x;
        output[4] = track.transform.q.y;
        output[5] = track.transform.q.z;
        output[6] = track.transform.q.w;
        output[7] = track.velocity.x;
        output[8] = track.velocity.y;
        output[9] = track.velocity.z;
    }

    void addGround()
    {
        mGroundMaterial = mPhysics->createMaterial(0.8f, 0.8f, 0.05f);
        mGround = mPhysics->createRigidStatic(PxTransform(PxVec3(0.0f, 0.0f, -0.5f)));
        PxShape* shape = mPhysics->createShape(
            PxBoxGeometry(50000.0f, 50000.0f, 0.5f), *mGroundMaterial, true);
        mGround->attachShape(*shape);
        shape->release();
        mScene->addActor(*mGround);
    }

    PxDefaultAllocator mAllocator;
    PxDefaultErrorCallback mError;
    PxFoundation* mFoundation = nullptr;
    PxPhysics* mPhysics = nullptr;
    PxDefaultCpuDispatcher* mDispatcher = nullptr;
    PxScene* mScene = nullptr;
    PxRigidStatic* mGround = nullptr;
    PxMaterial* mGroundMaterial = nullptr;
    bool mExtensionsInitialized = false;
    bool mEgoPoseInitialized = false;
    std::size_t mCapacity;
    std::size_t mNextSlot = 0;
    std::vector<std::size_t> mFreeSlots;
    std::vector<float> mStates;
    std::vector<float> mTrackStates;
    std::vector<std::int64_t> mIds;
    std::vector<std::uint8_t> mActive;
    std::vector<std::uint8_t> mCollisionActive;
    std::vector<std::uint8_t> mDetached;
    std::vector<std::uint8_t> mStruck;
    std::unordered_map<std::int64_t, BodyRecord> mBodies;
    std::unordered_map<std::int64_t, BarrierRecord> mBarriers;
};

} // namespace

PYBIND11_MODULE(ludus_physx_native, module)
{
    module.doc() = "Standalone native PhysX scene for Ludus";
    py::class_<NativeScene>(module, "NativeScene")
        .def(py::init<std::size_t>(), py::arg("capacity"))
        .def("add_body", &NativeScene::addBody)
        .def("set_body_track", &NativeScene::setBodyTrack)
        .def("update_body", &NativeScene::updateBody)
        .def("set_body_collision_enabled", &NativeScene::setBodyCollisionEnabled)
        .def("set_body_track_drive_enabled", &NativeScene::setBodyTrackDriveEnabled)
        .def("set_body_detached", &NativeScene::setBodyDetached)
        .def("set_body_track_controls", &NativeScene::setBodyTrackControls)
        .def("remove_body", &NativeScene::removeBody)
        .def("add_barrier", &NativeScene::addBarrier)
        .def("remove_barrier", &NativeScene::removeBarrier)
        .def("step", &NativeScene::step)
        .def("step_tracked", &NativeScene::stepTracked)
        .def("state_buffer", &NativeScene::stateBuffer)
        .def("track_state_buffer", &NativeScene::trackStateBuffer)
        .def("id_buffer", &NativeScene::idBuffer)
        .def("active_buffer", &NativeScene::activeBuffer)
        .def("collision_active_buffer", &NativeScene::collisionActiveBuffer)
        .def("detached_buffer", &NativeScene::detachedBuffer)
        .def("struck_buffer", &NativeScene::struckBuffer)
        .def_property_readonly("body_count", &NativeScene::bodyCount)
        .def_property_readonly("barrier_count", &NativeScene::barrierCount)
        .def("close", &NativeScene::close);
}
