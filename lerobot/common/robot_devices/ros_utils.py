import rclpy


def ensure_rclpy_init():
    """Ensure rclpy.init() is only called once per process."""
    if not getattr(ensure_rclpy_init, '_initialized', False):
        rclpy.init()
        ensure_rclpy_init._initialized = True
