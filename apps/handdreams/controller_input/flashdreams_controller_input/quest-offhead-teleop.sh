#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Manage Quest wake state for the WebXR input host.

set -Eeuo pipefail

QUEST_SERIAL="${QUEST_SERIAL:-}"
QUEST_ADB="${QUEST_ADB:-}"
QUEST_OWNER_PID="${QUEST_OWNER_PID:-$PPID}"
REQUIRE_PHYSICAL_WAKE="${QUEST_REQUIRE_PHYSICAL_WAKE:-0}"
QUEST_LAUNCHER_LOCK_FILE="${QUEST_LAUNCHER_LOCK_FILE:-${XDG_RUNTIME_DIR:-/tmp}/cosmos-interactive-quest-${UID}.lock}"

SETTINGS_CHANGED=0
PROXIMITY_CHANGED=0
ORIGINAL_AUTOSLEEP=""
ORIGINAL_PROXIMITY=""
META_PIN="${QUEST_META_PIN:-}"

LAUNCHER_LOCK_FD=""
LAUNCHER_LOCK_ACQUIRED=0

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}


find_adb() {
    if [[ -n "$QUEST_ADB" && -x "$QUEST_ADB" ]]; then
        printf '%s\n' "$QUEST_ADB"
        return
    fi
    if command -v adb >/dev/null 2>&1; then
        command -v adb
        return
    fi
    local sdk_root
    for sdk_root in "${ANDROID_SDK_ROOT:-}" "${ANDROID_HOME:-}" "${HOME:-}/Android/Sdk"; do
        if [[ -n "$sdk_root" && -x "$sdk_root/platform-tools/adb" ]]; then
            printf '%s\n' "$sdk_root/platform-tools/adb"
            return
        fi
    done
    return 1
}


adb_run() {
    timeout --signal=KILL 5s "${ADB[@]}" "$@"
}


adb_cleanup() {
    timeout --signal=KILL 2s "${ADB[@]}" "$@"
}


snapshot_browser_settings() {
    local properties autosleep_pattern='(^|[,{[:space:]])disable_autosleep=(true|false)([^[:alnum:]_]|$)'
    local proximity_pattern='(^|[,{[:space:]])set_proximity_close=(true|false)([^[:alnum:]_]|$)'
    if properties="$(adb_run shell content call --uri content://com.oculus.rc --method GET_PROPERTY)"; then
        if [[ "$properties" =~ $autosleep_pattern ]]; then
            ORIGINAL_AUTOSLEEP="${BASH_REMATCH[2]}"
        fi
        if [[ "$properties" =~ $proximity_pattern ]]; then
            ORIGINAL_PROXIMITY="${BASH_REMATCH[2]}"
        fi
    fi
    if [[ -n "$META_PIN" && -z "$ORIGINAL_AUTOSLEEP" ]]; then
        die "could not snapshot Quest auto-sleep; no protected settings were changed"
    fi
    if [[ -z "$ORIGINAL_PROXIMITY" ]]; then
        printf '[quest] proximity state unavailable; cleanup will send prox_far as best effort.\n'
    fi
}


select_quest() {
    local adb_bin
    adb_bin="$(find_adb || true)"
    [[ -n "$adb_bin" ]] ||
        die "adb is missing; install Android platform-tools and authorize the headset"

    if [[ -n "$QUEST_SERIAL" ]]; then
        ADB=("$adb_bin" -s "$QUEST_SERIAL")
    else
        local -a devices=()
        mapfile -t devices < <(
            timeout --signal=KILL 5s "$adb_bin" devices |
                awk 'NR > 1 && $2 == "device" { print $1 }'
        )

        case "${#devices[@]}" in
            0)
                timeout --signal=KILL 5s "$adb_bin" devices >&2 || true
                die "no authorized Quest found; approve the USB-debugging prompt"
                ;;
            1)
                QUEST_SERIAL="${devices[0]}"
                ADB=("$adb_bin" -s "$QUEST_SERIAL")
                ;;
            *)
                printf 'Authorized Android devices:\n' >&2
                printf '  %s\n' "${devices[@]}" >&2
                die "multiple devices found; rerun with QUEST_SERIAL=<serial>"
                ;;
        esac
    fi

    [[ "$(adb_run get-state 2>/dev/null || true)" == "device" ]] ||
        die "Quest $QUEST_SERIAL is not authorized"
}


acquire_launcher_lock() {
    command -v flock >/dev/null 2>&1 ||
        die "flock is required to prevent overlapping Quest launchers"
    [[ "$QUEST_OWNER_PID" =~ ^[1-9][0-9]*$ ]] ||
        die "QUEST_OWNER_PID must be a positive process ID"
    kill -0 "$QUEST_OWNER_PID" 2>/dev/null ||
        die "the owning controller-input process $QUEST_OWNER_PID is not running"

    exec {LAUNCHER_LOCK_FD}>>"$QUEST_LAUNCHER_LOCK_FILE"
    if ! flock -n "$LAUNCHER_LOCK_FD"; then
        local existing_pid=""
        IFS= read -r existing_pid < "$QUEST_LAUNCHER_LOCK_FILE" || true
        die "another Quest launcher is already running${existing_pid:+ (PID $existing_pid)}"
    fi
    LAUNCHER_LOCK_ACQUIRED=1
    printf '%s\n' "$BASHPID" > "$QUEST_LAUNCHER_LOCK_FILE"
}


owner_is_alive() {
    kill -0 "$QUEST_OWNER_PID" 2>/dev/null
}


check_owner() {
    if ! owner_is_alive; then
        printf '\nOwning controller-input process exited; canceling headset startup.\n'
        exit 0
    fi
}


read_wakefulness() {
    local power pattern='(^|[[:space:]])mWakefulness=(Awake|Asleep|Dozing|Dreaming)([[:space:]]|$)'
    check_owner
    power="$(adb_run shell dumpsys power)" || die "could not read Quest power state; check the USB connection"
    check_owner
    [[ "$power" =~ $pattern ]] || die "Quest returned no recognized mWakefulness state"
    WAKEFULNESS="${BASH_REMATCH[2]}"
}


await_physical_wake() {
    # Do not simulate worn state until the operator has woken the headset.
    # Awake is not proof of unlock.
    check_owner
    PROXIMITY_CHANGED=1
    adb_run shell am broadcast -a com.oculus.vrpowermanager.prox_far >/dev/null
    read_wakefulness
    if [[ "$WAKEFULNESS" != Asleep ]]; then
        printf '[quest] putting the headset to sleep...\n'
        adb_run shell input keyevent KEYCODE_SLEEP >/dev/null
    fi

    local deadline=$((SECONDS + 10)) confirmed=0
    while (( confirmed < 2 )); do
        read_wakefulness
        if [[ "$WAKEFULNESS" == Asleep ]]; then
            confirmed=$((confirmed + 1))
        else
            confirmed=0
        fi
        (( SECONDS < deadline )) || die "could not confirm Quest sleep; remove the headset and stop other wake automation, then retry"
        sleep 0.25
    done

    if [[ -n "${QUEST_WAKE_READY_FD:-}" ]]; then
        printf 'W' >&"$QUEST_WAKE_READY_FD"
    fi
    printf '[quest] press the headset power button once to continue (Ctrl-C to cancel).\n'
    confirmed=0
    while (( confirmed < 2 )); do
        read_wakefulness
        if [[ "$WAKEFULNESS" == Awake ]]; then
            confirmed=$((confirmed + 1))
        else
            confirmed=0
        fi
        sleep 0.25
    done
    check_owner
    printf '[quest] headset awake. Clear any lock screen in the headset.\n'
}


restore_headset() {
    local result action=prox_far
    if [[ "$SETTINGS_CHANGED" -eq 1 ]]; then
        if ! result="$(adb_cleanup shell content call \
            --uri content://com.oculus.rc --method SET_PROPERTY \
            --extra "disable_autosleep:b:${ORIGINAL_AUTOSLEEP}" \
            --extra "PIN:s:${META_PIN}" 2>/dev/null)" || [[ "$result" != *"Success=true"* ]]; then
            printf 'warning: could not restore previous Quest auto-sleep setting; reconnect the headset and restore it manually\n' >&2
        fi
    fi
    if [[ "$PROXIMITY_CHANGED" -eq 1 ]]; then
        [[ "$ORIGINAL_PROXIMITY" == true ]] && action=prox_close
        adb_cleanup shell am broadcast -a "com.oculus.vrpowermanager.${action}" \
            >/dev/null 2>&1 || printf 'warning: could not restore Quest proximity state\n' >&2
    fi
}


cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    set +e

    if [[ "$LAUNCHER_LOCK_ACQUIRED" -eq 1 ]] && declare -p ADB >/dev/null 2>&1; then
        if [[ "$PROXIMITY_CHANGED" -eq 1 || "$SETTINGS_CHANGED" -eq 1 ]]; then
            restore_headset
            # Sleep explicitly: releasing the hold alone waits for the idle timer.
            # KEYCODE_SLEEP cannot wake an already sleeping headset like POWER can.
            adb_cleanup shell input keyevent KEYCODE_SLEEP >/dev/null 2>&1 ||
                printf 'warning: could not put Quest to sleep; press its power button manually\n' >&2
        fi
    fi

    META_PIN=""
    printf 'Quest browser headset hold stopped; cleanup attempted for this session.\n'
    exit "$status"
}


trap cleanup EXIT INT TERM HUP

[[ "$REQUIRE_PHYSICAL_WAKE" == 0 || "$REQUIRE_PHYSICAL_WAKE" == 1 ]] ||
    die "QUEST_REQUIRE_PHYSICAL_WAKE must be 0 or 1"

select_quest
acquire_launcher_lock
[[ -z "$META_PIN" || "$META_PIN" =~ ^[0-9]+$ ]] ||
    die "the Meta Store PIN must contain only digits"
snapshot_browser_settings

printf 'Quest:     %s\n' "$QUEST_SERIAL"
printf 'Mode:      browser headset hold\n'

if [[ "$REQUIRE_PHYSICAL_WAKE" == 1 ]]; then
    await_physical_wake
fi
check_owner
if [[ -n "$META_PIN" ]]; then
    SETTINGS_CHANGED=1
    result="$(adb_run shell content call --uri content://com.oculus.rc \
        --method SET_PROPERTY --extra 'disable_autosleep:b:true' \
        --extra "PIN:s:${META_PIN}")" || die "could not disable Quest auto-sleep"
    [[ "$result" == *"Success=true"* ]] || die "Quest rejected the auto-sleep change; check the supplied PIN"
else
    printf 'No explicit PIN supplied; using proximity hold without changing protected settings.\n'
fi

PROXIMITY_CHANGED=1
adb_run shell am broadcast \
    -a com.oculus.vrpowermanager.prox_close >/dev/null

check_owner
if [[ -n "${QUEST_WAKE_READY_FD:-}" ]]; then
    printf '1' >&"$QUEST_WAKE_READY_FD"
    exec {QUEST_WAKE_READY_FD}>&-
fi
printf '\nQuest headset hold ready for browser input.\n'
printf 'Close the controller-input host or press Ctrl-C to stop and restore normal headset behavior.\n'

# Reinforce the simulated worn state after a temporary USB disconnect.
# Poll the owner once a second so even SIGKILL or an
# interpreter crash tears the headset down without leaving an orphaned loop.
while owner_is_alive; do
    adb_run shell am broadcast \
        -a com.oculus.vrpowermanager.prox_close >/dev/null 2>&1 || true
    for ((second = 0; second < 30; second++)); do
        sleep 1
        owner_is_alive || break 2
    done
done
printf '\nOwning controller-input process exited; restoring the headset.\n'
