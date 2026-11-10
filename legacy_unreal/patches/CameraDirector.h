#pragma once

#include "CoreMinimal.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "PawnSimApi.h"
#include "PIPCamera.h"
#include "GameFramework/Actor.h"
#include "ManualPoseController.h"
#include "common/common_utils/Utils.hpp"
#include "GameFramework/SpringArmComponent.h"
#include "CameraDirector.generated.h"

UENUM(BlueprintType)
enum class ECameraDirectorMode : uint8
{
    CAMERA_DIRECTOR_MODE_FPV = 0 UMETA(DisplayName = "FPV"),
    CAMERA_DIRECTOR_MODE_GROUND_OBSERVER = 1 UMETA(DisplayName = "GroundObserver"),
    CAMERA_DIRECTOR_MODE_FLY_WITH_ME = 2 UMETA(DisplayName = "FlyWithMe"),
    CAMERA_DIRECTOR_MODE_MANUAL = 3 UMETA(DisplayName = "Manual"),
    CAMERA_DIRECTOR_MODE_SPRINGARM_CHASE = 4 UMETA(DisplayName = "SpringArmChase"),
    CAMERA_DIRECTOR_MODE_BACKUP = 5 UMETA(DisplayName = "Backup"),
    CAMERA_DIRECTOR_MODE_NODISPLAY = 6 UMETA(DisplayName = "No Display"),
    CAMERA_DIRECTOR_MODE_FRONT = 7 UMETA(DisplayName = "Front")
};

UCLASS()
class AIRSIM_API ACameraDirector : public AActor
{
    GENERATED_BODY()

public:
    /** Spring arm that will offset the camera */
    UPROPERTY(Category = Camera, VisibleDefaultsOnly, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
    USpringArmComponent* SpringArm;

    UPROPERTY(Category = Camera, VisibleDefaultsOnly, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
    APIPCamera* ExternalCamera;

public:
    void inputEventFpvView();
    void inputEventGroundView();
    void inputEventManualView();
    void inputEventFlyWithView();
    void inputEventSpringArmChaseView();
    void inputEventBackupView();
    void inputEventNoDisplayView();
    void inputEventFrontView();

    // GTA-style orbit + view toggle
    void inputEventFpvToggle();          // key 4 → FPV
    void inputEventChaseToggle();        // key 5 → SpringArmChase
    void onMouseOrbitYaw(float val);     // MouseX axis
    void onMouseOrbitPitch(float val);   // MouseY axis

public:
    ACameraDirector();
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void Tick(float DeltaSeconds) override;

    UFUNCTION(BlueprintCallable, Category = "Modes")
    ECameraDirectorMode getMode();
    UFUNCTION(BlueprintCallable, Category = "Modes")
    void setMode(ECameraDirectorMode mode);

    void initializeForBeginPlay(ECameraDirectorMode view_mode,
                                AActor* follow_actor, APIPCamera* fpv_camera, APIPCamera* front_camera, APIPCamera* back_camera);

    APIPCamera* getFpvCamera() const;
    APIPCamera* getExternalCamera() const;
    APIPCamera* getBackupCamera() const;
    void setFollowDistance(const int follow_distance) { this->follow_distance_ = follow_distance; }
    void setCameraRotationLagEnabled(const bool lag_enabled) { this->camera_rotation_lag_enabled_ = lag_enabled; }

private:
    void setupInputBindings();
    void attachSpringArm(bool attach);
    void disableCameras(bool fpv, bool backup, bool external, bool front);
    void notifyViewModeChanged();
    void writeOrbitState();              // writes yaw/pitch to file for Python
    void captureMouse(bool capture);     // lock/unlock mouse cursor

private:
    typedef common_utils::Utils Utils;

    APIPCamera* fpv_camera_;
    APIPCamera* backup_camera_;
    APIPCamera* front_camera_;
    AActor* follow_actor_;

    USceneComponent* last_parent_ = nullptr;

    ECameraDirectorMode mode_;
    UPROPERTY()
    UManualPoseController* manual_pose_controller_;

    FVector camera_start_location_;
    FVector initial_ground_obs_offset_;
    FRotator camera_start_rotation_;
    bool ext_obs_fixed_z_;
    int follow_distance_;
    bool camera_rotation_lag_enabled_;

    // GTA-style orbit camera state
    float orbit_yaw_ = 0.0f;
    float orbit_pitch_ = -20.0f;
    float mouse_sensitivity_ = 0.15f;
    FString camera_state_path_;          // path to camera_state.txt
    bool mouse_captured_ = false;

    // Mouse axis bindings (stored so we can remove them)
    FInputAxisKeyMapping orbit_yaw_mapping_;
    FInputAxisKeyMapping orbit_pitch_mapping_;
    FInputAxisBinding* orbit_yaw_binding_ = nullptr;
    FInputAxisBinding* orbit_pitch_binding_ = nullptr;
};