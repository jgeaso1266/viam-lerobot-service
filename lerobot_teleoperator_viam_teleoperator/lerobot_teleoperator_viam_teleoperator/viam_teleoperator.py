"""Viam implementation of Teleoperator interface for LeRobot."""
import asyncio
import math
import os
from typing import Any, Dict, List, Optional

from grpclib import GRPCError
from viam.components.arm import Arm
from viam.components.input import Controller, Control, Event
from viam.robot.client import RobotClient
from viam.logging import logging

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_viam_teleoperator import ViamTeleoperatorConfig

LOGGER = logging.getLogger(__name__)

async def viam_connect(
    api_key_value: str,
    api_key_id: str,
    robot_address: str,
):
    """Connect to a Viam robot using API key credentials."""

    opts = RobotClient.Options.with_api_key(api_key_value, api_key_id)
    return await RobotClient.at_address(robot_address, opts)

class ViamTeleoperator(Teleoperator):
    """
    Wraps a Viam Arm or Controller component to provide a LeRobot-compatible teleop interface.

    Supports two modes:
    - "arm" mode: Returns absolute joint positions (mirroring a leader arm)
    - "controller" mode: Returns delta values for end-effector control (gamepad input)
    """

    config_class = ViamTeleoperatorConfig
    name = "viam_teleoperator"

    def __init__(self, config: ViamTeleoperatorConfig):
        super().__init__(config)
        self.id = config.id
        self._is_connected = False
        self._last_positions: Optional[List[float]] = None
        self._teleop_device_name = config.teleop_device_name
        self._num_joints = config.num_joints
        self._teleop_device_type = config.teleop_device_type
        self._machine = None
        self._teleop_device: Arm | Controller | None = None
        self._loop = asyncio.new_event_loop()

        # Controller-specific configuration
        self._use_gripper = config.use_gripper
        self._axis_deadzone = config.axis_deadzone

        # Resolve Control enums from string config
        self._axis_x_control = Control(config.axis_x)
        self._axis_y_control = Control(config.axis_y)
        self._axis_z_control = Control(config.axis_z)
        self._gripper_close_control = Control(config.gripper_close_button)
        self._gripper_open_control = Control(config.gripper_open_button)

        # Resolve scale factors
        base_scale = config.delta_scale
        self._scale_x = config.delta_scale_x if config.delta_scale_x is not None else base_scale
        self._scale_y = config.delta_scale_y if config.delta_scale_y is not None else base_scale
        self._scale_z = config.delta_scale_z if config.delta_scale_z is not None else base_scale

    @property
    def action_features(self) -> Dict[str, type]:
        """Return action features based on teleop device type.

        For arm mode: joint positions (e.g., {"joint_0.pos": float, ...})
        For controller mode: delta values (e.g., {"delta_x": float, ...})
        """
        if self._teleop_device_type == "controller":
            features = {
                "delta_x": float,
                "delta_y": float,
                "delta_z": float,
            }
            if self._use_gripper:
                features["gripper"] = float
            return features
        else:
            return {f"joint_{i}.pos": float for i in range(self._num_joints)}

    @property
    def feedback_features(self) -> Dict[str, type]:
        """No feedback features supported."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the teleoperator device and disable torque for free movement."""
        self._machine = self._loop.run_until_complete(viam_connect(
            api_key_value=os.environ.get("VIAM_API_KEY", ""),
            api_key_id=os.environ.get("VIAM_API_KEY_ID", ""),
            robot_address=os.environ.get("VIAM_MACHINE_FQDN", "")
        ))

        if self._teleop_device_type == "controller":
            self._teleop_device = Controller.from_robot(self._machine, self._teleop_device_name)
            if not self._teleop_device:
                raise ValueError(f"Controller '{self._teleop_device_name}' not found on robot.")

            # Validate controller has expected controls
            available_controls = self._loop.run_until_complete(self._teleop_device.get_controls())
            self._validate_controller_controls(available_controls)
        else:
            self._teleop_device = Arm.from_robot(self._machine, self._teleop_device_name)
            if not self._teleop_device:
                raise ValueError(f"Arm '{self._teleop_device_name}' not found on robot.")

        self._is_connected = True
        LOGGER.info(f"ViamTeleoperator connected ({self._teleop_device_type} mode)")

    @property
    def is_calibrated(self) -> bool:
        """Viam arms handle their own calibration."""
        return True

    def calibrate(self) -> None:
        """No calibration needed - handled by Viam arm component."""

    def configure(self) -> None:
        """Configure the teleoperator device."""
        if self._teleop_device_type == "arm":
            self._loop.run_until_complete(self._set_torque(False))
        # Controller mode requires no special configuration

    def get_action(self) -> Dict[str, float]:
        """
        Get current action from the teleoperator.

        Returns:
            For arm mode: dict mapping joint names to positions in radians
            For controller mode: dict with delta_x, delta_y, delta_z (and optionally gripper)
        """
        if self._teleop_device_type == "controller":
            return self._get_controller_action()
        else:
            return self._get_arm_action()

    def _get_arm_action(self) -> Dict[str, float]:
        """Get action from arm teleoperator (joint positions)."""
        try:
            joint_positions = self._loop.run_until_complete(self._teleop_device.get_joint_positions())
            LOGGER.debug(f"Teleoperator joint positions: {joint_positions.values}")
            self._last_positions = {
                f"joint_{i}.pos": math.radians(pos)
                for i, pos in enumerate(joint_positions.values)
            }
            return self._last_positions
        except asyncio.TimeoutError:
            LOGGER.warning("Failed to get joint positions from teleoperator arm, using last known positions.")
            if self._last_positions is not None:
                return self._last_positions
            raise RuntimeError("Failed to get joint positions and no previous positions available")
        except GRPCError as e:
            LOGGER.warning(f"Failed to get joint positions from teleoperator arm: {e.message}, using last known positions.")
            if self._last_positions is not None:
                return self._last_positions
            raise RuntimeError(f"Failed to get joint positions ({e.message}) and no previous positions available")

    def _get_controller_action(self) -> Dict[str, float]:
        """Get action from controller teleoperator (delta values).

        Reads current state from controller axes and buttons,
        converts to delta values for end-effector control.
        """
        events = self._loop.run_until_complete(self._teleop_device.get_events())

        delta_x = self._get_axis_value(events, self._axis_x_control) * self._scale_x
        delta_y = self._get_axis_value(events, self._axis_y_control) * self._scale_y
        delta_z = self._get_axis_value(events, self._axis_z_control) * self._scale_z

        LOGGER.debug(f"Controller deltas: x={delta_x:.4f}, y={delta_y:.4f}, z={delta_z:.4f}")

        action = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self._use_gripper:
            gripper_action = self._get_gripper_action(events)
            action["gripper"] = gripper_action
            LOGGER.debug(f"Gripper action: {gripper_action}")

        return action

    def _get_axis_value(self, events: Dict[Control, Event], control: Control) -> float:
        """Extract axis value from events, applying deadzone."""
        if control not in events:
            return 0.0

        value = events[control].value

        # Apply deadzone
        if abs(value) < self._axis_deadzone:
            return 0.0

        return value

    def _get_gripper_action(self, events: Dict[Control, Event]) -> float:
        """Determine gripper action from button states.

        Returns:
            0.0 = CLOSE
            1.0 = STAY (no action)
            2.0 = OPEN
        """
        # Check close button
        if self._gripper_close_control in events:
            if events[self._gripper_close_control].value > 0:
                return 0.0  # CLOSE

        # Check open button
        if self._gripper_open_control in events:
            if events[self._gripper_open_control].value > 0:
                return 2.0  # OPEN

        return 1.0  # STAY

    def _validate_controller_controls(self, available_controls: List[Control]) -> None:
        """Warn if configured controls are not available on the controller."""
        required = [self._axis_x_control, self._axis_y_control, self._axis_z_control]
        if self._use_gripper:
            required.extend([self._gripper_close_control, self._gripper_open_control])

        for control in required:
            if control not in available_controls:
                LOGGER.warning(
                    f"Configured control '{control.value}' not available on controller. "
                    f"Available: {[c.value for c in available_controls]}"
                )

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send feedback to the teleoperator (not implemented)."""
        # Force feedback not supported on Viam arms

    def disconnect(self) -> None:
        """Disconnect from the teleoperator."""
        if self._machine:
            self._loop.run_until_complete(self._machine.close())

        # Cancel any remaining tasks on our loop
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()

        self._is_connected = False
        self._teleop_device = None
        LOGGER.info("ViamTeleoperator disconnected")

    async def _set_torque(self, enabled: bool) -> None:
        """Enable or disable torque on the teleoperator arm."""
        await self._teleop_device.do_command({"command": "set_torque", "enable": enabled})
