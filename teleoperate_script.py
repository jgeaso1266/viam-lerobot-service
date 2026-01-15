"""Teleoperate a Viam robot arm using another Viam arm as the input device."""
import asyncio

from viam.robot.client import RobotClient
from viam.components.arm import Arm

from lerobot_robot_viam_robot import ViamRobotConfig, ViamRobotWrapper
from lerobot_teleoperator_viam_teleoperator import ViamTeleoperatorConfig, ViamTeleopWrapper

async def connect():
    """Connect to the Viam robot using API credentials."""
    opts = RobotClient.Options.with_api_key(

        api_key='12j96ln186dx8cz0iuijfhc4816qfas0',

        api_key_id='b3da81c3-2aa9-49c5-a053-f02ff5925813'
    )
    return await RobotClient.at_address('test-arm-main.0ddc34cpra.viam.cloud', opts)

async def main():
    """Main teleoperation loop."""
    async with await connect() as machine:
        print('Resources:')
        print(machine.resource_names)

        follower = Arm.from_robot(machine, "follower")
        leader = Arm.from_robot(machine, "leader")

        robot_config = ViamRobotConfig(
            arm=follower.name
        )

        teleop_config = ViamTeleoperatorConfig(
            teleop_device=leader.name,
        )

        robot = ViamRobotWrapper(robot_config, dependencies={
                Arm.get_resource_name("follower"): follower,
                Arm.get_resource_name("leader"): leader
            }
        )
        teleop_device = ViamTeleopWrapper(teleop_config, dependencies={
                Arm.get_resource_name("follower"): follower,
                Arm.get_resource_name("leader"): leader
            }
        )
        await robot.connect_async()  # Set speed/acceleration
        await teleop_device.connect_async()  # Disable torque on leader

        print("\nStarting teleoperation loop...")
        print("Press Ctrl+C to stop\n")

        while True:
            action = await teleop_device.get_action_async()
            await robot.send_action_async(action)

if __name__ == "__main__":
    asyncio.run(main())
