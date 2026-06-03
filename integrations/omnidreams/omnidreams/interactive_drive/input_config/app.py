# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tkinter wizard for ``interactive-drive-configuration``.

Walks the user through selecting an input device (steering wheel or game
controller), calibrating its steering / throttle / brake axes by listening
to live input, optionally binding reverse / reset buttons and testing force
feedback, then writing a local profile YAML the demo runtime auto-discovers.

The GUI is intentionally thin: all calibration logic lives in
:mod:`omnidreams.interactive_drive.input_config.capture` and all profile
IO in :mod:`omnidreams.interactive_drive.input.wheel_profiles`.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import yaml
from omnidreams.interactive_drive.input.wheel_profiles import (
    AutocenterFFB,
    EvdevDevice,
    apply_steering_curve,
    delete_profile_file,
    list_device_axes,
    load_wheel_profile_files,
    name_match_strength,
    profile_filename,
    save_wheel_profile,
    scan_evdev_devices,
    update_profile_file,
    user_wheel_profiles_dir,
    wheel_profile_to_yaml_dict,
)
from omnidreams.interactive_drive.input_config.capture import (
    CaptureSession,
    build_profile,
    infer_pedal_inverted,
    infer_steering_invert,
    peak_from_observed,
    select_axis_by_span,
)

try:  # Tkinter is stdlib but needs the system Tk package installed.
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # pragma: no cover - exercised only on Tk-less hosts
    tk = None  # type: ignore[assignment]

# Common absolute-axis code names, purely for nicer labels in the override
# menu and live readout. Falls back to the raw code for anything unlisted.
_ABS_NAMES = {
    0x00: "ABS_X",
    0x01: "ABS_Y",
    0x02: "ABS_Z",
    0x03: "ABS_RX",
    0x04: "ABS_RY",
    0x05: "ABS_RZ",
    0x06: "ABS_THROTTLE",
    0x07: "ABS_RUDDER",
    0x08: "ABS_WHEEL",
    0x09: "ABS_GAS",
    0x0A: "ABS_BRAKE",
    0x10: "ABS_HAT0X",
    0x11: "ABS_HAT0Y",
}

# Steps whose live panel shows axis activity / status text.
_AXIS_LIVE_STEPS = ("device", "controls")
# Max wheel rotation drawn in the live panel, degrees each direction.
_WHEEL_MAX_DEG = 120.0
# Live-panel canvas size. Fixed so populating it never reflows the window
# (which previously shifted the device list under the cursor mid-click).
_CANVAS_W = 690
_CANVAS_H = 150
# Minimum movement (fraction of an axis' full range) before a calibration
# step auto-binds that axis. The leeway keeps idle jitter or an accidental
# nudge of a different control from being picked.
_DETECT_FRACTION = 0.18


def _axis_label(code: int) -> str:
    return f"0x{code:02x} ({_ABS_NAMES.get(code, f'ABS_{code}')})"


def _axis_from_label(label: str) -> int:
    return int(label.split()[0], 16)


class ConfigApp:
    """Wizard controller built around a single ``tk.Tk`` root."""

    def __init__(self, root) -> None:
        self.root = root
        self.root.title("interactive-drive input configuration")
        self.root.geometry("780x740")
        self.root.minsize(760, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.state: dict = {}
        self.session: CaptureSession | None = None
        self.devices: tuple[EvdevDevice, ...] = ()
        self._ffb: AutocenterFFB | None = None
        self._saved = False
        self._step_index = 0
        # When set to ``(path, profile)`` the editor screen is shown instead
        # of the new-profile wizard.
        self._editing: tuple | None = None
        # Capture coordination: only one axis capture can listen at a time
        # (they share the session's observed buffer). ``_recording_key`` is
        # the section currently listening; ``_button_listening`` is the
        # button binding currently waiting for a press.
        self._recording_key: str | None = None
        self._record_buttons: dict = {}
        self._detect_callbacks: dict = {}
        self._button_listening: str | None = None
        self._button_result_vars: dict = {}

        self.device_type_var = tk.StringVar(value="wheel")
        self.activity_var = tk.StringVar(value="")

        self._build_chrome()
        self._render()
        self._tick()

    # -- window chrome ---------------------------------------------------

    def _build_chrome(self) -> None:
        # Pack the footer and live panel against the bottom FIRST so they
        # always keep their space. The content area is packed last with
        # ``expand`` so it gives up room when the window is small, instead
        # of squeezing the Back/Next buttons off-screen.
        footer = ttk.Frame(self.root, padding=(16, 10))
        footer.pack(side="bottom", fill="x")
        ttk.Button(footer, text="Cancel", command=self._on_close).pack(side="left")
        self.primary_btn = ttk.Button(footer, text="Next", command=self._on_primary)
        self.primary_btn.pack(side="right")
        self.back_btn = ttk.Button(footer, text="Back", command=self._on_back)
        self.back_btn.pack(side="right", padx=(0, 8))

        live = ttk.LabelFrame(self.root, text="Live inputs", padding=(8, 4))
        live.pack(side="bottom", fill="x", padx=12, pady=(0, 4))
        ttk.Label(live, textvariable=self.activity_var, foreground="#2f8f2f").pack(
            anchor="w"
        )
        # Fixed-size canvas: a steering-wheel + pedal visualization plus a
        # compact per-axis activity strip. Its dimensions are fixed, so the
        # widgets above never shift when it starts drawing on device select.
        self.live_canvas = tk.Canvas(
            live, width=_CANVAS_W, height=_CANVAS_H, highlightthickness=0
        )
        self.live_canvas.pack(anchor="w")

        header = ttk.Frame(self.root, padding=(16, 10))
        header.pack(side="top", fill="x")
        self.title_var = tk.StringVar()
        ttk.Label(
            header, textvariable=self.title_var, font=("TkDefaultFont", 15, "bold")
        ).pack(anchor="w")
        self.step_var = tk.StringVar()
        ttk.Label(header, textvariable=self.step_var, foreground="#888").pack(
            anchor="w"
        )

        self.content = ttk.Frame(self.root, padding=(16, 4))
        self.content.pack(side="top", fill="both", expand=True)

    def _clear_content(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()

    # -- step navigation -------------------------------------------------

    def _steps(self) -> list[str]:
        steps = ["welcome", "device", "controls", "buttons"]
        if self.state.get("device_type") == "wheel":
            steps.append("ffb")
        steps += ["details", "review"]
        return steps

    def _current_step(self) -> str:
        steps = self._steps()
        return steps[min(self._step_index, len(steps) - 1)]

    def _render(self) -> None:
        self._clear_content()
        self._recording_key = None
        self._record_buttons = {}
        self._detect_callbacks = {}
        self._button_listening = None
        self._button_result_vars = {}
        if self._editing is not None:
            self.step_var.set("Editing an existing profile")
            self.back_btn.state(["!disabled"])
            self.primary_btn.config(text="Save changes")
            self._build_edit()
            return
        step = self._current_step()
        steps = self._steps()
        self.step_var.set(f"Step {self._step_index + 1} of {len(steps)}")
        self.back_btn.state(["!disabled"] if self._step_index > 0 else ["disabled"])
        self.primary_btn.config(text="Save profile" if step == "review" else "Next")
        getattr(self, f"_build_{step}")()

    def _on_primary(self) -> None:
        if self._saved:
            self._on_close()
            return
        if self._editing is not None:
            self._save_edit()
            return
        step = self._current_step()
        ok, message = self._validate(step)
        if not ok:
            messagebox.showwarning("Not ready", message)
            return
        if step == "review":
            self._save()
            return
        self._step_index += 1
        self._render()

    def _on_back(self) -> None:
        if self._editing is not None:
            self._stop_session()
            self._editing = None
            self._render()
            return
        if self._step_index > 0:
            self._step_index -= 1
            self._render()

    # -- steps -----------------------------------------------------------

    def _build_welcome(self) -> None:
        self.title_var.set("Input device configuration")
        entries = load_wheel_profile_files(user_wheel_profiles_dir())
        saved = ttk.LabelFrame(self.content, text="Saved profiles", padding=(10, 6))
        saved.pack(fill="x", pady=(0, 10))
        if not entries:
            ttk.Label(saved, text="No saved profiles yet.").pack(anchor="w")
        else:
            for path, profile in entries:
                row = ttk.Frame(saved)
                row.pack(fill="x", pady=2)
                tag = "  [default]" if profile.is_default else ""
                ttk.Label(
                    row, text=f"{profile.display_name}{tag}", width=30, anchor="w"
                ).pack(side="left")
                ttk.Button(
                    row,
                    text="Edit",
                    width=6,
                    command=lambda p=path, pr=profile: self._start_edit(p, pr),
                ).pack(side="left", padx=2)
                ttk.Button(
                    row,
                    text=("Unset default" if profile.is_default else "Make default"),
                    width=13,
                    command=lambda p=path, pr=profile: self._toggle_default(p, pr),
                ).pack(side="left", padx=2)
                ttk.Button(
                    row,
                    text="Delete",
                    width=7,
                    command=lambda p=path, pr=profile: self._delete_profile(p, pr),
                ).pack(side="left", padx=2)

        ttk.Label(
            self.content,
            text="Create a new profile",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", pady=(6, 2))
        ttk.Label(
            self.content,
            wraplength=680,
            justify="left",
            text="Pick the device type, then click Next to detect and calibrate it.",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Radiobutton(
            self.content,
            text="Steering wheel + pedals",
            value="wheel",
            variable=self.device_type_var,
        ).pack(anchor="w")
        ttk.Radiobutton(
            self.content,
            text="Game controller / gamepad (stick + triggers)",
            value="controller",
            variable=self.device_type_var,
        ).pack(anchor="w")

    # -- existing-profile management + editor ---------------------------

    def _refresh_welcome(self) -> None:
        self._stop_session()
        self._editing = None
        self._step_index = 0
        self._render()

    def _stop_session(self) -> None:
        if self.session is not None:
            self.session.stop()
            self.session = None

    def _start_edit(self, path, profile) -> None:
        self._editing = (path, profile)
        self._render()

    def _delete_profile(self, path, profile) -> None:
        if messagebox.askyesno("Delete profile", f"Delete '{profile.display_name}'?"):
            delete_profile_file(path)
            self._refresh_welcome()

    def _toggle_default(self, path, profile) -> None:
        make_default = not profile.is_default
        for other_path, other in load_wheel_profile_files(user_wheel_profiles_dir()):
            if other_path == path:
                desired = make_default
            elif make_default:
                desired = False  # single default: clear the others
            else:
                desired = other.is_default
            if desired != other.is_default:
                update_profile_file(other_path, replace(other, is_default=desired))
        self._refresh_welcome()

    def _slider_row(self, label: str, var, low: float, high: float) -> None:
        row = ttk.Frame(self.content)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=26, anchor="w").pack(side="left")
        ttk.Scale(
            row, from_=low, to=high, orient="horizontal", length=220, variable=var
        ).pack(side="left", padx=8)
        value_label = ttk.Label(row, width=5)
        value_label.pack(side="left")

        def _update(*_args) -> None:
            value_label.config(text=f"{float(var.get()):.2f}")

        var.trace_add("write", _update)
        _update()

    def _build_edit(self) -> None:
        path, profile = self._editing
        self.title_var.set(f"Edit: {profile.display_name}")
        self._edit_display_name = tk.StringVar(value=profile.display_name)
        self._edit_invert_steer = tk.BooleanVar(value=profile.invert_steering)
        self._edit_invert_pedals = tk.BooleanVar(value=profile.inverted_pedals)
        self._edit_ffb = tk.BooleanVar(value=profile.ffb_enabled)
        self._edit_ffb_gain = tk.DoubleVar(value=profile.ffb_gain)
        self._edit_range = tk.DoubleVar(value=profile.steering_range)
        self._edit_deadzone = tk.DoubleVar(value=profile.steering_deadzone)
        self._edit_default = tk.BooleanVar(value=profile.is_default)

        # Open a live session for this profile's connected device so the
        # steering preview below reflects the deadzone / sensitivity sliders
        # as you drag them.
        self._stop_session()
        device = self._find_profile_device(profile)
        if device is not None:
            try:
                session = CaptureSession(device.path)
                session.start()
                self.session = session
            except OSError:
                self.session = None
        ttk.Label(
            self.content,
            foreground="#2f8f2f",
            wraplength=680,
            text=(
                f"Live preview from {device.name} -- operate the controls to see the feel."
                if device is not None
                else "Connect this device to preview the steering feel live."
            ),
        ).pack(anchor="w", pady=(0, 6))

        form = ttk.Frame(self.content)
        form.pack(fill="x")
        ttk.Label(form, text="Display name").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self._edit_display_name, width=44).grid(
            row=0, column=1, sticky="w"
        )

        axis_map = profile.axis_map
        ttk.Label(
            self.content,
            foreground="#666",
            text=(
                "Axes (recalibrate by creating a new profile): "
                f"steering 0x{axis_map.get('steering', 0):02x}, "
                f"throttle 0x{axis_map.get('throttle', 0):02x}, "
                f"brake 0x{axis_map.get('brake', 0):02x}"
            ),
        ).pack(anchor="w", pady=(4, 6))

        ttk.Checkbutton(
            self.content, text="Invert steering", variable=self._edit_invert_steer
        ).pack(anchor="w")
        ttk.Checkbutton(
            self.content, text="Invert pedals", variable=self._edit_invert_pedals
        ).pack(anchor="w")
        self._slider_row("Steering range (sensitivity)", self._edit_range, 0.1, 1.0)
        self._slider_row("Steering deadzone", self._edit_deadzone, 0.0, 0.3)
        ffb_row = ttk.Frame(self.content)
        ffb_row.pack(fill="x", pady=2)
        ttk.Checkbutton(ffb_row, text="Force feedback", variable=self._edit_ffb).pack(
            side="left"
        )
        ttk.Scale(
            ffb_row,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            length=200,
            variable=self._edit_ffb_gain,
        ).pack(side="left", padx=8)

        ttk.Label(self.content, text="Detection patterns (one per line):").pack(
            anchor="w", pady=(6, 0)
        )
        self._edit_patterns = tk.Text(self.content, height=3, width=60)
        self._edit_patterns.pack(anchor="w", pady=2)
        self._edit_patterns.insert("1.0", "\n".join(profile.detection_patterns))

        ttk.Checkbutton(
            self.content, text="Use as the default profile", variable=self._edit_default
        ).pack(anchor="w", pady=(6, 0))
        ttk.Button(
            self.content,
            text="Delete this profile",
            command=lambda: self._delete_profile(path, profile),
        ).pack(anchor="w", pady=(10, 0))

    def _save_edit(self) -> None:
        path, profile = self._editing
        patterns = tuple(
            line.strip()
            for line in self._edit_patterns.get("1.0", "end").splitlines()
            if line.strip()
        )
        if not patterns:
            messagebox.showwarning("Not ready", "Add at least one detection pattern.")
            return
        updated = replace(
            profile,
            display_name=self._edit_display_name.get().strip() or profile.display_name,
            invert_steering=bool(self._edit_invert_steer.get()),
            inverted_pedals=bool(self._edit_invert_pedals.get()),
            ffb_enabled=bool(self._edit_ffb.get()),
            ffb_gain=float(self._edit_ffb_gain.get()),
            steering_range=float(self._edit_range.get()),
            steering_deadzone=float(self._edit_deadzone.get()),
            detection_patterns=patterns,
            is_default=bool(self._edit_default.get()),
        )
        update_profile_file(path, updated)
        if updated.is_default:
            for other_path, other in load_wheel_profile_files(
                user_wheel_profiles_dir()
            ):
                if other_path != path and other.is_default:
                    update_profile_file(other_path, replace(other, is_default=False))
        messagebox.showinfo("Saved", f"Updated {path.name}.")
        self._refresh_welcome()

    def _build_device(self) -> None:
        self.title_var.set("Select your device")
        ttk.Label(
            self.content,
            wraplength=680,
            justify="left",
            text=(
                "Pick your device, then operate any control -- the Live inputs panel "
                "below updates so you can confirm you chose the right one (some "
                "devices expose several nodes). Then click Next."
            ),
        ).pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(self.content)
        row.pack(fill="both", expand=True)
        self.device_list = tk.Listbox(row, height=7, exportselection=False)
        self.device_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(row, command=self.device_list.yview)
        scroll.pack(side="right", fill="y")
        self.device_list.config(yscrollcommand=scroll.set)
        self.device_list.bind("<<ListboxSelect>>", self._on_device_selected)

        ttk.Button(self.content, text="Rescan", command=self._refresh_devices).pack(
            anchor="w", pady=8
        )
        self._refresh_devices()

    def _refresh_devices(self) -> None:
        self.devices = scan_evdev_devices()
        self.device_list.delete(0, "end")
        for device in self.devices:
            self.device_list.insert("end", f"{device.name}  [{device.path}]")
        if not self.devices:
            self.device_list.insert("end", "(no readable input devices found)")

    def _on_device_selected(self, _event=None) -> None:
        selection = self.device_list.curselection()
        if not selection or not self.devices:
            return
        device = self.devices[selection[0]]
        if self.session is not None and self.session.device_path == device.path:
            return
        if self.session is not None:
            self.session.stop()
            self.session = None
        session = CaptureSession(device.path)
        try:
            session.start()
        except PermissionError:
            messagebox.showerror(
                "Permission denied",
                f"Cannot read {device.path}.\n\nAdd your user to the 'input' group "
                "(then log out/in) or add a udev rule, and try again.",
            )
            return
        except OSError as exc:
            messagebox.showerror("Cannot open device", f"{device.path}: {exc}")
            return
        self.session = session
        self.state["device"] = device

    def _build_controls(self) -> None:
        self.title_var.set("Calibrate controls")
        self.invert_pedals_var = tk.BooleanVar(
            value=self.state.get("inverted_pedals", True)
        )
        ttk.Label(
            self.content,
            wraplength=680,
            justify="left",
            text=(
                "For each control: click Start listening, then move it a little -- "
                "it auto-detects the axis (one at a time)."
            ),
        ).pack(anchor="w", pady=(0, 8))

        is_wheel = self.state.get("device_type") == "wheel"
        steer = ttk.LabelFrame(self.content, text="Steering", padding=(10, 6))
        steer.pack(fill="x", pady=3)
        self._steering_section(steer, "wheel" if is_wheel else "stick")
        thr = ttk.LabelFrame(self.content, text="Throttle", padding=(10, 6))
        thr.pack(fill="x", pady=3)
        self._pedal_section(
            thr, "throttle", "throttle pedal" if is_wheel else "throttle trigger"
        )
        brk = ttk.LabelFrame(self.content, text="Brake", padding=(10, 6))
        brk.pack(fill="x", pady=3)
        self._pedal_section(
            brk, "brake", "brake pedal" if is_wheel else "brake trigger"
        )
        # Single shared pedal-invert flag (the schema stores one for both
        # pedals). Auto-set from calibration; here as a manual safety net.
        ttk.Checkbutton(
            self.content,
            text="Invert pedals (toggle if pressing throttle/brake reads backwards)",
            variable=self.invert_pedals_var,
            command=self._apply_pedal_invert,
        ).pack(anchor="w", pady=(6, 0))

    def _steering_section(self, parent, control: str) -> None:
        self.invert_steering_var = tk.BooleanVar(
            value=self.state.get("invert_steering", False)
        )
        self._steering_axis_choice = tk.StringVar(
            value=self.state.get("_steering_axis_label", "")
        )
        result = tk.StringVar(value=self.state.get("_steering_summary", ""))
        ttk.Label(
            parent, text=f"Click Start, then turn the {control} a little to the LEFT."
        ).pack(anchor="w")

        def on_detect(axis: int, base: int, peak: int) -> None:
            rng = self.session.axis_ranges[axis]
            # ``peak`` is the deflected value; if turning left lowered the raw
            # value the steering sign must be flipped.
            invert = bool(infer_steering_invert(peak, base))
            self.state["steering_axis"] = axis
            self.state["invert_steering"] = invert
            self.state["_steering_axis_label"] = _axis_label(axis)
            self.state["_steering_summary"] = (
                f"Steering on {_axis_label(axis)} (range {rng.minimum}..{rng.maximum}), invert={invert}"
            )
            self.invert_steering_var.set(invert)
            self._steering_axis_choice.set(_axis_label(axis))
            result.set(self.state["_steering_summary"])

        row = ttk.Frame(parent)
        row.pack(anchor="w", fill="x", pady=4)
        self._record_button(row, "steering", on_detect, result)
        self._axis_override(row, self._steering_axis_choice)
        ttk.Label(
            parent, textvariable=result, foreground="#2f6fbf", wraplength=660
        ).pack(anchor="w")
        ttk.Checkbutton(
            parent,
            text="Invert steering (toggle if left/right feels reversed)",
            variable=self.invert_steering_var,
            command=self._apply_invert,
        ).pack(anchor="w")

    def _pedal_section(self, parent, key: str, control: str) -> None:
        axis_choice = tk.StringVar(value=self.state.get(f"_{key}_axis_label", ""))
        result = tk.StringVar(value=self.state.get(f"_{key}_summary", ""))
        setattr(self, f"_{key}_axis_choice", axis_choice)
        ttk.Label(parent, text=f"Click Start, then press the {control} a little.").pack(
            anchor="w"
        )

        def on_detect(axis: int, base: int, peak: int) -> None:
            inverted = bool(infer_pedal_inverted(base, peak))
            self.state[f"{key}_axis"] = axis
            self.state[f"{key}_inverted"] = inverted
            # Throttle defines the shared pedal-invert flag; brake seeds it
            # only when throttle hasn't been calibrated yet.
            if key == "throttle" or "throttle_inverted" not in self.state:
                self.state["inverted_pedals"] = inverted
                self.invert_pedals_var.set(inverted)
            self.state[f"_{key}_axis_label"] = _axis_label(axis)
            self.state[f"_{key}_summary"] = (
                f"{key.capitalize()} on {_axis_label(axis)} (inverted={inverted})"
            )
            axis_choice.set(_axis_label(axis))
            result.set(self.state[f"_{key}_summary"])

        row = ttk.Frame(parent)
        row.pack(anchor="w", fill="x", pady=4)
        self._record_button(row, key, on_detect, result)
        self._axis_override(row, axis_choice)
        ttk.Label(
            parent, textvariable=result, foreground="#2f6fbf", wraplength=660
        ).pack(anchor="w")

    def _record_button(self, parent, key: str, on_detect, result_var) -> None:
        """A Start/cancel listening toggle.

        Clicking starts listening; the axis auto-binds as soon as one moves
        past the leeway threshold (handled in :meth:`_tick`), so there is no
        Stop click. Clicking again before that cancels. Only one section
        listens at a time, since they share the session's observed buffer.
        """
        btn = ttk.Button(parent, text="Start listening")
        self._record_buttons[key] = btn
        self._detect_callbacks[key] = on_detect

        def toggle() -> None:
            if self.session is None:
                messagebox.showwarning(
                    "No device", "Go back and select a device first."
                )
                return
            if self._recording_key == key:
                self._recording_key = None
                btn.config(text="Start listening")
                result_var.set("Cancelled.")
            else:
                if self._recording_key is not None:
                    other = self._record_buttons.get(self._recording_key)
                    if other is not None:
                        other.config(text="Start listening")
                self.session.reset_observed()
                self._recording_key = key
                btn.config(text="Listening... (click to cancel)")
                result_var.set("Move the control a little to bind it.")

        btn.config(command=toggle)
        btn.pack(side="left")

    def _apply_invert(self) -> None:
        # Reflect the checkbox in state immediately so the live wheel flips.
        self.state["invert_steering"] = bool(self.invert_steering_var.get())

    def _apply_pedal_invert(self) -> None:
        # Reflect the checkbox in state immediately so the live pedal bars flip.
        self.state["inverted_pedals"] = bool(self.invert_pedals_var.get())

    def _axis_override(self, parent, choice_var) -> None:
        ttk.Label(parent, text="  Axis:").pack(side="left")
        codes = sorted(self.session.axis_ranges) if self.session is not None else []
        labels = [_axis_label(code) for code in codes] or ["(none)"]
        if not choice_var.get():
            choice_var.set(labels[0])
        ttk.OptionMenu(parent, choice_var, choice_var.get(), *labels).pack(
            side="left", padx=6
        )

    def _build_buttons(self) -> None:
        self.title_var.set("Bind buttons (optional)")
        ttk.Label(
            self.content,
            wraplength=680,
            justify="left",
            text=(
                "Optionally bind one button to toggle reverse, one to reset / "
                "respawn, and one to exit the scene (back to the scene selector). "
                "Click Bind, then press the button on your device. Leave unbound "
                "to skip -- you can always reset with the R key and exit with the "
                "X key."
            ),
        ).pack(anchor="w", pady=(0, 8))
        for key, label in (
            ("reverse", "Reverse"),
            ("reset", "Reset / respawn"),
            ("exit", "Exit scene"),
        ):
            frame = ttk.LabelFrame(self.content, text=label, padding=(10, 6))
            frame.pack(fill="x", pady=4)
            result = tk.StringVar(value=self._button_summary(key))
            self._button_result_vars[key] = result
            row = ttk.Frame(frame)
            row.pack(anchor="w", fill="x")
            ttk.Button(
                row,
                text=f"Bind {label.split()[0].lower()} button",
                command=lambda k=key: self._start_button_listen(k),
            ).pack(side="left")
            ttk.Button(
                row, text="Clear", command=lambda k=key: self._clear_button(k)
            ).pack(side="left", padx=8)
            ttk.Label(frame, textvariable=result, foreground="#2f6fbf").pack(
                anchor="w", pady=(4, 0)
            )

    def _button_summary(self, key: str) -> str:
        codes = self.state.get(f"{key}_buttons", ())
        return f"Bound to button code {codes[0]}" if codes else "Not bound"

    def _start_button_listen(self, key: str) -> None:
        if self.session is None:
            messagebox.showwarning("No device", "Go back and select a device first.")
            return
        self.session.reset_observed()
        self._button_listening = key
        var = self._button_result_vars.get(key)
        if var is not None:
            var.set("Press the button on your device now...")

    def _clear_button(self, key: str) -> None:
        self.state[f"{key}_buttons"] = ()
        if self._button_listening == key:
            self._button_listening = None
        var = self._button_result_vars.get(key)
        if var is not None:
            var.set("Not bound")

    def _build_ffb(self) -> None:
        self.title_var.set("Force feedback (optional)")
        self.ffb_enabled_var = tk.BooleanVar(value=self.state.get("ffb_enabled", True))
        self.ffb_gain_var = tk.DoubleVar(value=self.state.get("ffb_gain", 0.6))
        ttk.Label(
            self.content,
            wraplength=680,
            justify="left",
            text=(
                "Autocenter force feedback makes the wheel resist turning and return "
                "to center, scaled by speed. Enable it and test the feel; leave it "
                "off if your device has no autocenter motor."
            ),
        ).pack(anchor="w", pady=(0, 10))
        ttk.Checkbutton(
            self.content,
            text="Enable autocenter force feedback",
            variable=self.ffb_enabled_var,
        ).pack(anchor="w")
        gain_row = ttk.Frame(self.content)
        gain_row.pack(anchor="w", pady=8, fill="x")
        ttk.Label(gain_row, text="Gain").pack(side="left")
        ttk.Scale(
            gain_row,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            length=300,
            variable=self.ffb_gain_var,
        ).pack(side="left", padx=8)
        test_row = ttk.Frame(self.content)
        test_row.pack(anchor="w", pady=4)
        ttk.Button(test_row, text="Test", command=self._ffb_test).pack(side="left")
        ttk.Button(test_row, text="Stop", command=self._ffb_stop).pack(
            side="left", padx=8
        )

    def _ffb_test(self) -> None:
        if self.session is None:
            return
        self._ffb_stop()
        ffb = AutocenterFFB()
        ffb.init(self.session.device_path, float(self.ffb_gain_var.get()))
        if not ffb.available:
            messagebox.showinfo(
                "Force feedback unavailable",
                "Could not open the device for force feedback (it may not support "
                "autocenter, or write permission is missing).",
            )
            return
        ffb.set_autocenter(float(self.ffb_gain_var.get()))
        self._ffb = ffb

    def _ffb_stop(self) -> None:
        if self._ffb is not None:
            self._ffb.cleanup()
            self._ffb = None

    def _build_details(self) -> None:
        self.title_var.set("Settings & detection")
        device = self.state.get("device")
        default_name = device.name if device else "My device"
        controller = self.state.get("device_type") == "controller"
        self.display_name_var = tk.StringVar(
            value=self.state.get("display_name", default_name)
        )
        self.profile_name_var = tk.StringVar(
            value=self.state.get("name", profile_filename(default_name)[:-5])
        )
        self.is_default_var = tk.BooleanVar(value=self.state.get("is_default", True))
        # Controllers default to a reduced range + small deadzone (sticks are
        # sensitive and tend to drift); wheels default to full range, none.
        self.steering_range_var = tk.DoubleVar(
            value=self.state.get("steering_range", 0.6 if controller else 1.0)
        )
        self.steering_deadzone_var = tk.DoubleVar(
            value=self.state.get("steering_deadzone", 0.08 if controller else 0.0)
        )
        # Seed state so the live wheel preview reflects these immediately, and
        # keep it updated as the sliders move.
        self.state["steering_range"] = float(self.steering_range_var.get())
        self.state["steering_deadzone"] = float(self.steering_deadzone_var.get())
        self.steering_range_var.trace_add(
            "write",
            lambda *_: self.state.__setitem__(
                "steering_range", float(self.steering_range_var.get())
            ),
        )
        self.steering_deadzone_var.trace_add(
            "write",
            lambda *_: self.state.__setitem__(
                "steering_deadzone", float(self.steering_deadzone_var.get())
            ),
        )

        form = ttk.Frame(self.content)
        form.pack(fill="x")
        ttk.Label(form, text="Display name").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.display_name_var, width=46).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(form, text="Profile name").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.profile_name_var, width=46).grid(
            row=1, column=1, sticky="w"
        )

        ttk.Label(
            self.content, text="Steering feel (turn the device to preview):"
        ).pack(anchor="w", pady=(8, 0))
        self._slider_row(
            "Steering range (sensitivity)", self.steering_range_var, 0.1, 1.0
        )
        self._slider_row("Steering deadzone", self.steering_deadzone_var, 0.0, 0.3)

        ttk.Label(
            self.content,
            wraplength=680,
            justify="left",
            text=(
                "\nDetection patterns -- one per line. The device is auto-selected at "
                "launch when its name contains any of these."
            ),
        ).pack(anchor="w")
        self.patterns_text = tk.Text(self.content, height=3, width=62)
        self.patterns_text.pack(anchor="w", pady=4)
        existing = self.state.get("detection_patterns")
        self.patterns_text.insert(
            "1.0", "\n".join(existing) if existing else (device.name if device else "")
        )

        ttk.Checkbutton(
            self.content,
            text="Use as the default profile",
            variable=self.is_default_var,
        ).pack(anchor="w", pady=6)

    def _build_review(self) -> None:
        self.title_var.set("Review and save")
        profile = self._compose_profile()
        self.state["_profile"] = profile
        preview = yaml.safe_dump(
            wheel_profile_to_yaml_dict(profile),
            sort_keys=False,
            default_flow_style=False,
        )
        ttk.Label(
            self.content,
            text=f"Will be written to:\n{user_wheel_profiles_dir() / profile_filename(profile.name)}",
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        text = tk.Text(self.content, height=15, width=66)
        text.pack(fill="both", expand=True)
        text.insert("1.0", preview)
        text.config(state="disabled")

    # -- validation + compose -------------------------------------------

    def _validate(self, step: str) -> tuple[bool, str]:
        if step == "welcome":
            self.state["device_type"] = self.device_type_var.get()
            return True, ""
        if step == "device":
            if self.session is None:
                return False, "Select a device from the list first."
            return True, ""
        if step == "controls":
            for axis_key, human in (
                ("steering", "steering"),
                ("throttle", "throttle"),
                ("brake", "brake"),
            ):
                if f"{axis_key}_axis" not in self.state:
                    return (
                        False,
                        f"Calibrate {human} first (Start listening, move it, Stop).",
                    )
            self.state["steering_axis"] = _axis_from_label(
                self._steering_axis_choice.get()
            )
            self.state["invert_steering"] = bool(self.invert_steering_var.get())
            self.state["throttle_axis"] = _axis_from_label(
                self._throttle_axis_choice.get()
            )
            self.state["brake_axis"] = _axis_from_label(self._brake_axis_choice.get())
            self.state["inverted_pedals"] = bool(self.invert_pedals_var.get())
            return True, ""
        if step == "buttons":
            self.state.setdefault("reverse_buttons", ())
            self.state.setdefault("reset_buttons", ())
            self.state.setdefault("exit_buttons", ())
            return True, ""
        if step == "ffb":
            self._ffb_stop()
            self.state["ffb_enabled"] = bool(self.ffb_enabled_var.get())
            self.state["ffb_gain"] = float(self.ffb_gain_var.get())
            return True, ""
        if step == "details":
            name = self.profile_name_var.get().strip()
            if not name:
                return False, "Profile name cannot be empty."
            patterns = [
                line.strip()
                for line in self.patterns_text.get("1.0", "end").splitlines()
                if line.strip()
            ]
            if not patterns:
                return False, "Add at least one detection pattern."
            self.state["name"] = name
            self.state["display_name"] = self.display_name_var.get().strip() or name
            self.state["detection_patterns"] = tuple(patterns)
            self.state["is_default"] = bool(self.is_default_var.get())
            self.state["steering_range"] = float(self.steering_range_var.get())
            self.state["steering_deadzone"] = float(self.steering_deadzone_var.get())
            return True, ""
        return True, ""

    def _compose_profile(self):
        if self.state.get("device_type") == "controller":
            ffb_enabled, ffb_gain = False, 0.0
        else:
            ffb_enabled = bool(self.state.get("ffb_enabled", False))
            ffb_gain = float(self.state.get("ffb_gain", 0.0))
        return build_profile(
            name=self.state["name"],
            display_name=self.state["display_name"],
            detection_patterns=self.state["detection_patterns"],
            steering_axis=self.state["steering_axis"],
            throttle_axis=self.state["throttle_axis"],
            brake_axis=self.state["brake_axis"],
            invert_steering=self.state["invert_steering"],
            inverted_pedals=bool(self.state.get("inverted_pedals", True)),
            ffb_enabled=ffb_enabled,
            ffb_gain=ffb_gain,
            is_default=self.state["is_default"],
            reverse_buttons=tuple(self.state.get("reverse_buttons", ())),
            reset_buttons=tuple(self.state.get("reset_buttons", ())),
            exit_buttons=tuple(self.state.get("exit_buttons", ())),
            steering_range=float(self.state.get("steering_range", 1.0)),
            steering_deadzone=float(self.state.get("steering_deadzone", 0.0)),
        )

    def _save(self) -> None:
        profile = self.state.get("_profile") or self._compose_profile()
        brake_inverted = self.state.get("brake_inverted")
        if brake_inverted is not None and brake_inverted != profile.inverted_pedals:
            if not messagebox.askyesno(
                "Pedal direction mismatch",
                "Throttle and brake appear to rest in opposite directions, but the "
                "profile stores a single shared 'Invert pedals' setting. Check that "
                "checkbox matches your throttle. Save anyway?",
            ):
                return
        try:
            path = save_wheel_profile(profile, user_wheel_profiles_dir())
        except OSError as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self._saved = True
        self.primary_btn.config(text="Close")
        self.back_btn.state(["disabled"])
        messagebox.showinfo(
            "Profile saved",
            f"Saved to:\n{path}\n\nLaunch the demo with:\n"
            "uv run --package flashdreams-omnidreams interactive-drive",
        )

    # -- live loop + teardown -------------------------------------------

    def _tick(self) -> None:
        if self.session is not None and self._button_listening is not None:
            pressed = self.session.pressed_buttons()
            if pressed:
                code = min(pressed)
                key = self._button_listening
                self.state[f"{key}_buttons"] = (code,)
                self._button_listening = None
                var = self._button_result_vars.get(key)
                if var is not None:
                    var.set(f"Bound to button code {code}")
        if self.session is not None and self._recording_key is not None:
            axis = select_axis_by_span(
                self.session.observed_ranges(),
                self.session.axis_ranges,
                min_fraction=_DETECT_FRACTION,
            )
            if axis is not None:
                key = self._recording_key
                self._recording_key = None
                button = self._record_buttons.get(key)
                if button is not None:
                    button.config(text="Start listening")
                observed = self.session.observed_ranges()
                base = self.session.baseline().get(
                    axis, int(self.session.axis_ranges[axis].center)
                )
                peak = peak_from_observed(observed[axis], base)
                callback = self._detect_callbacks.get(key)
                if callback is not None:
                    callback(axis, base, peak)
        self._draw_live()
        self.root.after(60, self._tick)

    # -- live wheel / pedal / axis visualization ------------------------

    def _find_profile_device(self, profile):
        """Best connected device matching *profile* (exact name preferred)."""
        required = {int(axis) for axis in profile.axis_map.values()}
        best = None  # (strength, device)
        for device in scan_evdev_devices():
            if not required.issubset(set(list_device_axes(device.path))):
                continue
            strength = name_match_strength(device.name, profile.detection_patterns)
            if strength > 0 and (best is None or strength > best[0]):
                best = (strength, device)
        return best[1] if best is not None else None

    def _live_config(self) -> dict:
        """Axis map + steering feel source for the live preview.

        In the editor it reads the edit widgets, so dragging the deadzone /
        range sliders updates the preview immediately; otherwise it's the
        new-profile wizard state.
        """
        if self._editing is not None and hasattr(self, "_edit_range"):
            _path, profile = self._editing
            return {
                "steering_axis": profile.axis_map.get("steering"),
                "throttle_axis": profile.axis_map.get("throttle"),
                "brake_axis": profile.axis_map.get("brake"),
                "invert_steering": bool(self._edit_invert_steer.get()),
                "inverted_pedals": bool(self._edit_invert_pedals.get()),
                "steering_range": float(self._edit_range.get()),
                "steering_deadzone": float(self._edit_deadzone.get()),
            }
        return self.state

    def _draw_live(self) -> None:
        canvas = self.live_canvas
        canvas.delete("all")
        if self.session is None:
            self.activity_var.set("")
            return
        editing = self._editing is not None
        step = self._current_step()
        if not editing and step in ("welcome", "review"):
            self.activity_var.set("")
            return
        if editing:
            self.activity_var.set("Move the stick / wheel to preview the steering feel")
        elif step == "buttons":
            self.activity_var.set(
                "Press the button on your device..." if self._button_listening else ""
            )
        elif step in _AXIS_LIVE_STEPS:
            if self._recording_key is not None:
                self.activity_var.set("Listening -- move the control...")
            elif self.session.is_active():
                self.activity_var.set("Activity detected")
            else:
                self.activity_var.set("Operate a control to see it move")
        else:
            self.activity_var.set("")

        axes = self.session.axes()
        ranges = self.session.axis_ranges
        steer, throttle, brake = self._sim_values(axes, ranges)
        self._draw_wheel(canvas, 78, 70, 56, steer)
        self._draw_pedal(canvas, 168, throttle, "Throttle", "#76b900")
        self._draw_pedal(canvas, 226, brake, "Brake", "#d05a5a")
        self._draw_axis_strip(canvas, 300, axes, ranges)
        if not editing and step == "buttons":
            held = sorted(c for c, v in self.session.buttons().items() if v == 1)
            canvas.create_text(
                10,
                _CANVAS_H - 8,
                anchor="w",
                fill="#666",
                font=("TkFixedFont", 8),
                text="Buttons held: "
                + (", ".join(str(c) for c in held) if held else "(none)"),
            )

    def _sim_values(self, axes: dict, ranges: dict) -> tuple[float, float, float]:
        """Normalized (steer, throttle, brake) from the active config source.

        Unmapped controls read as neutral; the steering curve (deadzone +
        sensitivity) is applied so the preview matches what the runtime does.
        """
        cfg = self._live_config()
        steer = throttle = brake = 0.0
        steer_axis = cfg.get("steering_axis")
        if steer_axis is not None and steer_axis in ranges and steer_axis in axes:
            rng = ranges[steer_axis]
            value = (axes[steer_axis] - rng.center) / (rng.span / 2.0)
            if cfg.get("invert_steering"):
                value = -value
            steer = apply_steering_curve(
                value,
                deadzone=float(cfg.get("steering_deadzone", 0.0) or 0.0),
                scale=float(cfg.get("steering_range", 1.0) or 1.0),
            )
        for key in ("throttle", "brake"):
            axis = cfg.get(f"{key}_axis")
            if axis is None or axis not in ranges or axis not in axes:
                continue
            rng = ranges[axis]
            if cfg.get("inverted_pedals", True):
                value = (rng.maximum - axes[axis]) / rng.span
            else:
                value = (axes[axis] - rng.minimum) / rng.span
            value = max(0.0, min(1.0, value))
            if key == "throttle":
                throttle = value
            else:
                brake = value
        return steer, throttle, brake

    def _draw_wheel(self, canvas, cx: int, cy: int, r: int, steer: float) -> None:
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#999", width=5)
        # Positive steer = left, which reads as a counter-clockwise turn.
        angle = math.radians(-steer * _WHEEL_MAX_DEG)
        for spoke in range(3):
            a = angle + spoke * (2.0 * math.pi / 3.0)
            x = cx + r * math.sin(a)
            y = cy - r * math.cos(a)
            is_top = spoke == 0
            canvas.create_line(
                cx,
                cy,
                x,
                y,
                fill="#76b900" if is_top else "#bbb",
                width=5 if is_top else 3,
            )
        canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="#555", outline="")
        canvas.create_text(
            cx,
            cy + r + 14,
            fill="#666",
            text=f"{int(round(steer * _WHEEL_MAX_DEG)):+d}\u00b0",
        )

    def _draw_pedal(self, canvas, x: int, value: float, label: str, color: str) -> None:
        top, bottom, width = 16, 118, 34
        value = max(0.0, min(1.0, value))
        canvas.create_rectangle(x, top, x + width, bottom, outline="#999")
        fill_h = value * (bottom - top)
        canvas.create_rectangle(
            x, bottom - fill_h, x + width, bottom, fill=color, outline=""
        )
        canvas.create_text(
            x + width / 2,
            top - 8,
            fill="#666",
            font=("TkDefaultFont", 8),
            text=f"{int(value * 100)}%",
        )
        canvas.create_text(
            x + width / 2,
            bottom + 14,
            fill="#666",
            font=("TkDefaultFont", 8),
            text=label,
        )

    def _draw_axis_strip(self, canvas, x0: int, axes: dict, ranges: dict) -> None:
        bar_x, bar_w, row_h = x0 + 52, 150, 16
        canvas.create_text(
            x0,
            6,
            anchor="w",
            fill="#888",
            font=("TkDefaultFont", 8, "bold"),
            text="Axes",
        )
        for i, code in enumerate(sorted(ranges)):
            if i >= 8:
                break
            rng = ranges[code]
            value = axes.get(code, rng.minimum)
            frac = max(0.0, min(1.0, (value - rng.minimum) / rng.span))
            y = 20 + i * row_h
            canvas.create_text(
                x0,
                y,
                anchor="w",
                fill="#666",
                font=("TkFixedFont", 8),
                text=f"0x{code:02x}",
            )
            canvas.create_rectangle(bar_x, y - 5, bar_x + bar_w, y + 5, outline="#aaa")
            canvas.create_rectangle(
                bar_x, y - 5, bar_x + frac * bar_w, y + 5, fill="#5a9bd5", outline=""
            )
            canvas.create_text(
                bar_x + bar_w + 6,
                y,
                anchor="w",
                fill="#666",
                font=("TkFixedFont", 8),
                text=str(value),
            )

    def _on_close(self) -> None:
        self._ffb_stop()
        if self.session is not None:
            self.session.stop()
            self.session = None
        self.root.destroy()


def main() -> None:
    if not sys.platform.startswith("linux") or not Path("/dev/input").exists():
        print(
            "interactive-drive-configuration requires Linux with evdev input "
            "devices under /dev/input.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if tk is None:
        print(
            "Tkinter is not available. Install your platform's Tk package "
            "(e.g. 'sudo apt-get install python3-tk') and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    root = tk.Tk()
    ConfigApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
