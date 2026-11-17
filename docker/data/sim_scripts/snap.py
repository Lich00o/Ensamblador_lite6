import rclpy, numpy as np, cv2, time
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
rclpy.init(); n = Node("snap"); d = {}
n.create_subscription(Image, "/sim_camera/image", lambda m: d.setdefault("img", m), 1)
n.create_subscription(Image, "/charuco_detector/debug_image", lambda m: d.setdefault("dbg", m), 1)
n.create_subscription(PoseStamped, "/charuco_detector/board_pose", lambda m: d.setdefault("pose", m), 1)
t = time.time()
while time.time() - t < 25 and not ("dbg" in d and "pose" in d): rclpy.spin_once(n, timeout_sec=0.2)
for k in ("img", "dbg"):
    if k in d:
        m = d[k]; a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding == "rgb8": a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"/home/ubuntu/.ros/lite6_handeye/sim_{k}.png", a); print(k, m.encoding, m.width, m.height)
p = d.get("pose"); print("pose", None if p is None else (p.header.frame_id, round(p.pose.position.x,3), round(p.pose.position.y,3), round(p.pose.position.z,3)))
