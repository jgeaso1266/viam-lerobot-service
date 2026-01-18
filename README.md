# Module lerobot

This module provides a Viam generic service that integrates with the [LeRobot](https://github.com/huggingface/lerobot) framework for robot learning and teleoperation.

## Model jalen:lerobot:lerobot

A generic service that wraps LeRobot functionality, enabling teleoperation of robotic arms through Viam's platform.

### Features

- **Teleoperation**: Control a follower arm using a leader arm in real-time
- **Visualization**: Optional Rerun-based visualization of robot state and camera feeds
- **Session Management**: Start/stop teleoperation sessions with unique session IDs

### Configuration

#### Attributes

| Name          | Type     | Inclusion | Description                                      |
|---------------|----------|-----------|--------------------------------------------------|
| `arm`         | string   | Required  | Name of the follower arm component               |
| `teleop`      | string   | Required  | Name of the leader/teleoperator arm component    |
| `cameras`     | []string | Optional  | List of camera component names for observation   |
| `dataset_dir` | string   | Optional  | Directory for storing datasets (default: `./datasets`) |
| `policy_dir`  | string   | Optional  | Directory for storing policies (default: `./policies`) |

#### Environment Variables

The service requires the following environment variables for API authentication:

| Variable            | Description                          |
|---------------------|--------------------------------------|
| `VIAM_API_KEY`      | Viam API key for authentication      |
| `VIAM_API_KEY_ID`   | Viam API key ID                      |
| `VIAM_MACHINE_FQDN` | Fully qualified domain name of the machine |

#### Example Configuration

Minimal configuration:
```json
{
  "arm": "follower-arm",
  "teleop": "leader-arm"
}
```

Full configuration with cameras:
```json
{
  "arm": "follower-arm",
  "teleop": "leader-arm",
  "cameras": ["front-camera", "wrist-camera"],
  "dataset_dir": "/data/datasets",
  "policy_dir": "/data/policies"
}
```

### DoCommand

The service exposes functionality through `DoCommand`:

#### start_teleoperation

Starts a teleoperation session where the leader arm controls the follower arm.

**Request:**
```json
{
  "command": "start_teleoperation",
  "fps": 30,
  "teleop_time_s": 60,
  "display_data": false,
  "display_ip": null,
  "display_port": 0,
  "display_compressed_images": false
}
```

| Parameter                  | Type   | Default | Description                                    |
|----------------------------|--------|---------|------------------------------------------------|
| `fps`                      | int    | 30      | Control loop frequency in Hz                   |
| `teleop_time_s`            | int    | 60      | Maximum teleoperation duration in seconds      |
| `display_data`             | bool   | false   | Enable Rerun visualization                     |
| `display_ip`               | string | null    | IP address for remote Rerun viewer             |
| `display_port`             | int    | 0       | Port for remote Rerun viewer                   |
| `display_compressed_images`| bool   | false   | Compress images for visualization              |

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "active_sessions": ["550e8400-e29b-41d4-a716-446655440000"]
}
```

#### stop_teleoperation

Stops an active teleoperation session.

**Request:**
```json
{
  "command": "stop_teleoperation",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "duration_s": 45.2,
  "active_sessions": []
}
```

### Usage Example (Python)

```python
from viam.robot.client import RobotClient
from viam.services.generic import Generic

async def main():
    robot = await RobotClient.at_address("your-robot-address", ...)

    lerobot_service = Generic.from_robot(robot, "lerobot-service")

    # Start teleoperation
    result = await lerobot_service.do_command({
        "command": "start_teleoperation",
        "fps": 30,
        "teleop_time_s": 120
    })
    session_id = result["session_id"]
    print(f"Started session: {session_id}")

    # ... perform teleoperation ...

    # Stop teleoperation
    result = await lerobot_service.do_command({
        "command": "stop_teleoperation",
        "session_id": session_id
    })
    print(f"Session duration: {result['duration_s']}s")
```

## LeRobot Plugin Packages

This module includes three LeRobot plugin packages that provide Viam-compatible implementations:

### lerobot_robot_viam_robot

Implements the LeRobot `Robot` interface for Viam arm components.

### lerobot_teleoperator_viam_teleoperator

Implements the LeRobot `Teleoperator` interface for Viam arm components used as leader devices.

### lerobot_camera_viam_camera

Implements the LeRobot `Camera` interface for Viam camera components.

## Requirements

- Python 3.10+
- [LeRobot](https://github.com/huggingface/lerobot) v0.3.2+
- [Viam SDK](https://python.viam.dev/)
