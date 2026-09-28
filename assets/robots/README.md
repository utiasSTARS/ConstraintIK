# Robot assets (viewer only)

Visual URDFs and meshes for the viser viewer and the README media; the joint vector maps 1:1 onto the actuated joints and the end-effector links match IKFast (`panda_link8`, `ee_link`). Collision geometry was removed and meshes were converted to GLB.

- `panda/`: Franka Emika Panda with hand, from `frankaemika/franka_ros` `franka_description` (Apache-2.0, Copyright 2017 Franka Emika GmbH; see `panda/LICENSE`, `panda/NOTICE`).
- `ur5/`: UR5 with a Robotiq 2F-85. The URDF is expanded from `ur5_robotiq_robot_limited.urdf.xacro` of KavrakiLab `robowflex_resources` (MIT, `ur5/LICENSE.robowflex_resources`) with the 0.9144 m pedestal offset removed. UR5 meshes come from `ros-industrial/universal_robot` `ur_description` (BSD, as declared in its package manifest). Robotiq 2F-85 meshes come from `StanleyInnovation/robotiq_85_gripper` (BSD, as declared in its package manifest; the repository is no longer public).
