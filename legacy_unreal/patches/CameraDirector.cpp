#include "CameraDirector.h"
#include "GameFramework/PlayerController.h"
#include "AirBlueprintLib.h"

ACameraDirector::ACameraDirector()
{
    PrimaryActorTick.bCanEverTick = true;

    // Create a spring arm component for GTA-style chase camera
    SpringArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("SpringArm"));
    SpringArm->SetRelativeLocation(FVector(0.0f, 0.0f, 34.0f));
    SpringArm->SetWorldRotation(FRotator(-20.0f, 0.0f, 0.0f));
    SpringArm->TargetArmLength = 300.0f;  // 3m behind drone (GTA distance)
    SpringArm->bEnableCameraLag = true;
    SpringArm->CameraLagSpeed = 12.0f;    // smooth follow
    SpringArm->bEnableCameraRotationLag = true;
    SpringArm->CameraRotationLagSpeed = 15.0f;  // smooth orbit
    SpringArm->bInheritPitch = false;     // mouse controls pitch
    SpringArm->bInheritYaw = false;       // mouse controls yaw
    SpringArm->bInheritRoll = false;      // no roll wobble
    SpringArm->bDoCollisionTest = false;  // don't clip through walls

    // Pre-compute camera state file path
    camera_state_path_ = FPaths::Combine(
        FPlatformProcess::UserDir(), TEXT("Documents/AirSim/camera_state.txt"));
}

void ACameraDirector::BeginPlay()
{
    Super::BeginPlay();

    // GTA-style: capture mouse so orbit works immediately
    captureMouse(true);
}

void ACameraDirector::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (mode_ == ECameraDirectorMode::CAMERA_DIRECTOR_MODE_MANUAL) {
        manual_pose_controller_->updateActorPose(DeltaTime);
    }
    else if (mode_ == ECameraDirectorMode::CAMERA_DIRECTOR_MODE_SPRINGARM_CHASE) {
        // GTA-style: apply mouse orbit rotation to spring arm
        SpringArm->SetWorldRotation(FRotator(orbit_pitch_, orbit_yaw_, 0.0f));
        writeOrbitState();
    }
    else if (mode_ == ECameraDirectorMode::CAMERA_DIRECTOR_MODE_NODISPLAY) {
        //do nothing, we have camera turned off
    }
    else { //make camera move in desired way
        UAirBlueprintLib::FollowActor(ExternalCamera, follow_actor_, initial_ground_obs_offset_, ext_obs_fixed_z_);
    }
}

ECameraDirectorMode ACameraDirector::getMode()
{
    return mode_;
}

void ACameraDirector::initializeForBeginPlay(ECameraDirectorMode view_mode,
                                             AActor* follow_actor, APIPCamera* fpv_camera, APIPCamera* front_camera, APIPCamera* back_camera)
{
    manual_pose_controller_ = NewObject<UManualPoseController>(this, "CameraDirector_ManualPoseController");
    manual_pose_controller_->initializeForPlay();

    setupInputBindings();

    mode_ = view_mode;

    follow_actor_ = follow_actor;
    fpv_camera_ = fpv_camera;
    front_camera_ = front_camera;
    backup_camera_ = back_camera;
    camera_start_location_ = ExternalCamera->GetActorLocation();
    camera_start_rotation_ = ExternalCamera->GetActorRotation();
    initial_ground_obs_offset_ = camera_start_location_ -
                                 (follow_actor_ ? follow_actor_->GetActorLocation() : FVector::ZeroVector);

    //set initial view mode
    switch (mode_) {
    case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_FLY_WITH_ME:
        inputEventFlyWithView();
        break;
    case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_FPV:
        inputEventFpvView();
        break;
    case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_GROUND_OBSERVER:
        inputEventGroundView();
        break;
    case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_MANUAL:
        inputEventManualView();
        break;
    case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_SPRINGARM_CHASE:
        inputEventSpringArmChaseView();
        break;
    case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_BACKUP:
        inputEventBackupView();
        break;
    case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_NODISPLAY:
        inputEventNoDisplayView();
        break;
    case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_FRONT:
        inputEventFrontView();
        break;
    default:
        throw std::out_of_range("Unsupported view mode specified in CameraDirector::initializeForBeginPlay");
    }
}

void ACameraDirector::attachSpringArm(bool attach)
{
    if (attach) {
        //If we do have actor to follow AND don't have sprint arm attached to that actor, we will attach it
        if (follow_actor_ && ExternalCamera->GetRootComponent()->GetAttachParent() != SpringArm) {
            //For car, we want a bit of camera lag, as that is customary of racing video games
            //If the lag is missing, the camera will also occasionally shake.
            //But, lag is not desired when piloting a drone
            SpringArm->bEnableCameraRotationLag = camera_rotation_lag_enabled_;

            //attach spring arm to actor
            SpringArm->AttachToComponent(follow_actor_->GetRootComponent(), FAttachmentTransformRules::KeepRelativeTransform);
            SpringArm->SetRelativeLocation(FVector(0.0f, 0.0f, 34.0f));

            //remember current parent for external camera. Later when we remove external
            //camera from spring arm, we will attach it back to its last parent
            last_parent_ = ExternalCamera->GetRootComponent()->GetAttachParent();
            ExternalCamera->DetachFromActor(FDetachmentTransformRules::KeepWorldTransform);
            //now attach camera to spring arm
            ExternalCamera->AttachToComponent(SpringArm, FAttachmentTransformRules::KeepRelativeTransform);
        }

        //For car, we need to move the camera back a little more than for a drone.
        //Otherwise, the camera will be stuck inside the car
        ExternalCamera->SetActorRelativeLocation(FVector(follow_distance_ * 100.0f, 0.0f, 0.0f));
        ExternalCamera->SetActorRelativeRotation(FRotator(10.0f, 0.0f, 0.0f));
        //ExternalCamera->bUsePawnControlRotation = false;
    }
    else { //detach
        if (last_parent_ && ExternalCamera->GetRootComponent()->GetAttachParent() == SpringArm) {
            ExternalCamera->DetachFromActor(FDetachmentTransformRules::KeepRelativeTransform);
            ExternalCamera->AttachToComponent(last_parent_, FAttachmentTransformRules::KeepRelativeTransform);
        }
    }
}

void ACameraDirector::setMode(ECameraDirectorMode mode)
{
    { //first remove any settings done by previous mode

        //detach spring arm
        if (mode_ == ECameraDirectorMode::CAMERA_DIRECTOR_MODE_SPRINGARM_CHASE &&
            mode != ECameraDirectorMode::CAMERA_DIRECTOR_MODE_SPRINGARM_CHASE) {
            attachSpringArm(false);
        }

        // Re-enable rendering
        if (mode_ == ECameraDirectorMode::CAMERA_DIRECTOR_MODE_NODISPLAY &&
            mode != ECameraDirectorMode::CAMERA_DIRECTOR_MODE_NODISPLAY) {
            UAirBlueprintLib::enableViewportRendering(this, true);
        }

        //Remove any existing key bindings for manual mode
        if (mode != ECameraDirectorMode::CAMERA_DIRECTOR_MODE_MANUAL) {
            if (ExternalCamera != nullptr && manual_pose_controller_->getActor() == ExternalCamera) {

                manual_pose_controller_->setActor(nullptr);
            }
            //else someone else is bound to manual pose controller, leave it alone
        }
    }

    { //perform any settings to enter in to this mode

        switch (mode) {
        case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_MANUAL:
            //if new mode is manual mode then add key bindings
            manual_pose_controller_->setActor(ExternalCamera);
            break;
        case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_SPRINGARM_CHASE:
            //if we switched to spring arm mode then attach to spring arm (detachment was done earlier in method)
            attachSpringArm(true);
            break;
        case ECameraDirectorMode::CAMERA_DIRECTOR_MODE_NODISPLAY:
            UAirBlueprintLib::enableViewportRendering(this, false);
            break;
        default:
            //other modes don't need special setup
            break;
        }
    }

    //make switch official
    mode_ = mode;
}

void ACameraDirector::setupInputBindings()
{
    UAirBlueprintLib::EnableInput(this);

    // Original view mode keys
    UAirBlueprintLib::BindActionToKey("inputEventFpvView", EKeys::F, this, &ACameraDirector::inputEventFpvView);
    UAirBlueprintLib::BindActionToKey("inputEventFlyWithView", EKeys::B, this, &ACameraDirector::inputEventFlyWithView);
    UAirBlueprintLib::BindActionToKey("inputEventGroundView", EKeys::Backslash, this, &ACameraDirector::inputEventGroundView);
    UAirBlueprintLib::BindActionToKey("inputEventManualView", EKeys::M, this, &ACameraDirector::inputEventManualView);
    UAirBlueprintLib::BindActionToKey("inputEventSpringArmChaseView", EKeys::Slash, this, &ACameraDirector::inputEventSpringArmChaseView);
    UAirBlueprintLib::BindActionToKey("inputEventBackupView", EKeys::K, this, &ACameraDirector::inputEventBackupView);
    UAirBlueprintLib::BindActionToKey("inputEventNoDisplayView", EKeys::Hyphen, this, &ACameraDirector::inputEventNoDisplayView);
    UAirBlueprintLib::BindActionToKey("inputEventFrontView", EKeys::I, this, &ACameraDirector::inputEventFrontView);

    // GTA-style: 4 = FPV, 5 = TPV (SpringArmChase)
    UAirBlueprintLib::BindActionToKey("inputEventFpvToggle", EKeys::Four, this, &ACameraDirector::inputEventFpvToggle);
    UAirBlueprintLib::BindActionToKey("inputEventChaseToggle", EKeys::Five, this, &ACameraDirector::inputEventChaseToggle);

    // Mouse orbit axes for SpringArmChase
    orbit_yaw_mapping_ = FInputAxisKeyMapping("orbitYaw", EKeys::MouseX);
    orbit_pitch_mapping_ = FInputAxisKeyMapping("orbitPitch", EKeys::MouseY);
    orbit_yaw_binding_ = &UAirBlueprintLib::BindAxisToKey(orbit_yaw_mapping_, this, this, &ACameraDirector::onMouseOrbitYaw);
    orbit_pitch_binding_ = &UAirBlueprintLib::BindAxisToKey(orbit_pitch_mapping_, this, this, &ACameraDirector::onMouseOrbitPitch);
}

void ACameraDirector::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    manual_pose_controller_ = nullptr;
    SpringArm = nullptr;
    ExternalCamera = nullptr;
}

APIPCamera* ACameraDirector::getFpvCamera() const
{
    return fpv_camera_;
}

APIPCamera* ACameraDirector::getExternalCamera() const
{
    return ExternalCamera;
}

APIPCamera* ACameraDirector::getBackupCamera() const
{
    return backup_camera_;
}

void ACameraDirector::inputEventSpringArmChaseView()
{
    if (ExternalCamera) {
        setMode(ECameraDirectorMode::CAMERA_DIRECTOR_MODE_SPRINGARM_CHASE);
        ExternalCamera->showToScreen();
        disableCameras(true, true, false, true);
    }
    else
        UAirBlueprintLib::LogMessageString("Camera is not available: ", "ExternalCamera", LogDebugLevel::Failure);

    notifyViewModeChanged();
}

void ACameraDirector::inputEventGroundView()
{
    if (ExternalCamera) {
        setMode(ECameraDirectorMode::CAMERA_DIRECTOR_MODE_GROUND_OBSERVER);
        ExternalCamera->showToScreen();
        disableCameras(true, true, false, true);
        ext_obs_fixed_z_ = true;
    }
    else
        UAirBlueprintLib::LogMessageString("Camera is not available: ", "ExternalCamera", LogDebugLevel::Failure);

    notifyViewModeChanged();
}

void ACameraDirector::inputEventManualView()
{
    if (ExternalCamera) {
        setMode(ECameraDirectorMode::CAMERA_DIRECTOR_MODE_MANUAL);
        ExternalCamera->showToScreen();
        disableCameras(true, true, false, true);
    }
    else
        UAirBlueprintLib::LogMessageString("Camera is not available: ", "ExternalCamera", LogDebugLevel::Failure);

    notifyViewModeChanged();
}

void ACameraDirector::inputEventNoDisplayView()
{
    if (ExternalCamera) {
        setMode(ECameraDirectorMode::CAMERA_DIRECTOR_MODE_NODISPLAY);
        disableCameras(true, true, true, true);
    }
    else
        UAirBlueprintLib::LogMessageString("Camera is not available: ", "ExternalCamera", LogDebugLevel::Failure);

    notifyViewModeChanged();
}

void ACameraDirector::inputEventBackupView()
{
    if (backup_camera_) {
        setMode(ECameraDirectorMode::CAMERA_DIRECTOR_MODE_BACKUP);
        backup_camera_->showToScreen();
        disableCameras(true, false, true, true);
    }
    else
        UAirBlueprintLib::LogMessageString("Camera is not available: ", "backup_camera", LogDebugLevel::Failure);

    notifyViewModeChanged();
}

void ACameraDirector::inputEventFrontView()
{
    if (front_camera_) {
        setMode(ECameraDirectorMode::CAMERA_DIRECTOR_MODE_FRONT);
        front_camera_->showToScreen();
        disableCameras(true, true, true, false);
    }
    else
        UAirBlueprintLib::LogMessageString("Camera is not available: ", "backup_camera", LogDebugLevel::Failure);

    notifyViewModeChanged();
}

void ACameraDirector::inputEventFlyWithView()
{
    if (ExternalCamera) {
        setMode(ECameraDirectorMode::CAMERA_DIRECTOR_MODE_FLY_WITH_ME);
        ExternalCamera->showToScreen();

        if (follow_actor_)
            ExternalCamera->SetActorLocationAndRotation(
                follow_actor_->GetActorLocation() + initial_ground_obs_offset_, camera_start_rotation_);
        disableCameras(true, true, false, true);
        ext_obs_fixed_z_ = false;
    }
    else
        UAirBlueprintLib::LogMessageString("Camera is not available: ", "ExternalCamera", LogDebugLevel::Failure);

    notifyViewModeChanged();
}

void ACameraDirector::inputEventFpvView()
{
    if (fpv_camera_) {
        setMode(ECameraDirectorMode::CAMERA_DIRECTOR_MODE_FPV);
        fpv_camera_->showToScreen();
        disableCameras(false, true, true, true);
    }
    else
        UAirBlueprintLib::LogMessageString("Camera is not available: ", "fpv_camera", LogDebugLevel::Failure);

    notifyViewModeChanged();
}

void ACameraDirector::disableCameras(bool fpv, bool backup, bool external, bool front)
{
    if (fpv && fpv_camera_)
        fpv_camera_->disableMain();
    if (backup && backup_camera_)
        backup_camera_->disableMain();
    if (external && ExternalCamera)
        ExternalCamera->disableMain();
    if (front && front_camera_)
        front_camera_->disableMain();
}

void ACameraDirector::notifyViewModeChanged()
{
    bool nodisplay = ECameraDirectorMode::CAMERA_DIRECTOR_MODE_NODISPLAY == mode_;

    UWorld* world = GetWorld();
    UGameViewportClient* gameViewport = world->GetGameViewport();
    gameViewport->bDisableWorldRendering = nodisplay;
}

// ──────────────────────────────────────────────────────────────────────────────
//  GTA-STYLE: FPV/TPV toggle (keys 4 and 5)
// ──────────────────────────────────────────────────────────────────────────────
void ACameraDirector::inputEventFpvToggle()
{
    // Key 4 → switch to FPV (same as pressing F)
    inputEventFpvView();
    captureMouse(false); // release mouse in FPV
}

void ACameraDirector::inputEventChaseToggle()
{
    // Key 5 → switch to SpringArmChase (GTA TPV)
    inputEventSpringArmChaseView();
    captureMouse(true);  // re-capture mouse for orbit
}

// ──────────────────────────────────────────────────────────────────────────────
//  GTA-STYLE: Mouse orbit handlers
// ──────────────────────────────────────────────────────────────────────────────
void ACameraDirector::onMouseOrbitYaw(float val)
{
    if (mode_ == ECameraDirectorMode::CAMERA_DIRECTOR_MODE_SPRINGARM_CHASE) {
        orbit_yaw_ += val * mouse_sensitivity_;
    }
}

void ACameraDirector::onMouseOrbitPitch(float val)
{
    if (mode_ == ECameraDirectorMode::CAMERA_DIRECTOR_MODE_SPRINGARM_CHASE) {
        orbit_pitch_ = FMath::Clamp(orbit_pitch_ - val * mouse_sensitivity_, -80.0f, 80.0f);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
//  GTA-STYLE: Write orbit state to file for Python to read
// ──────────────────────────────────────────────────────────────────────────────
void ACameraDirector::writeOrbitState()
{
    // Write yaw and pitch as text so drone_control4.py can read camera direction
    // Format: "yaw_deg pitch_deg\n"
    FString content = FString::Printf(TEXT("%.2f %.2f"), orbit_yaw_, orbit_pitch_);
    FFileHelper::SaveStringToFile(content, *camera_state_path_,
        FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM,
        &IFileManager::Get(), FILEWRITE_None);
}

// ──────────────────────────────────────────────────────────────────────────────
//  GTA-STYLE: Mouse capture / release
// ──────────────────────────────────────────────────────────────────────────────
void ACameraDirector::captureMouse(bool capture)
{
    APlayerController* PC = GetWorld()->GetFirstPlayerController();
    if (!PC) return;

    if (capture) {
        PC->bShowMouseCursor = false;
        FInputModeGameOnly InputMode;
        PC->SetInputMode(InputMode);
        mouse_captured_ = true;
    }
    else {
        PC->bShowMouseCursor = true;
        FInputModeGameAndUI InputMode;
        InputMode.SetLockMouseToViewportBehavior(EMouseLockMode::DoNotLock);
        PC->SetInputMode(InputMode);
        mouse_captured_ = false;
    }
}
