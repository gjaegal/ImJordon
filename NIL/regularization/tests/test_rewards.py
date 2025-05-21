import numpy as np
from regularization.torque_penalty import TorquePenalty
from regularization.action_smoothing import ActionSmoother
from regularization.physics_constraints import PhysicsConstraints

def test_torque_penalty():
    tp = TorquePenalty(weight=0.1)
    torques = np.array([1.0, -2.0, 0.5])
    reward = tp.compute(torques)
    expected = 0.1 * (1.0**2 + 2.0**2 + 0.5**2)
    assert np.isclose(reward, expected), f"Expected {expected}, got {reward}"

def test_action_smoothing():
    sm = ActionSmoother(action_dim=3, weight=0.05)
    a1 = np.array([0.0, 0.0, 0.0])
    sm.compute(a1)  # 첫 액션
    a2 = np.array([1.0, 0.0, -1.0])
    reward = sm.compute(a2)
    expected = 0.05 * np.linalg.norm(a2 - a1)
    assert np.isclose(reward, expected), f"Expected {expected}, got {reward}"

def test_physics_constraints():
    pc = PhysicsConstraints(vel_limit=1.0, torque_limit=2.0, weight=0.2)
    vels = np.array([0.5, 1.5, 3.0])     # 하나는 초과
    torques = np.array([1.0, 3.0, 2.5])  # 둘은 초과
    penalty = pc.compute(vels, torques)

    expected_vel = ((1.5 - 1.0)**2 + (3.0 - 1.0)**2)
    expected_torque = ((3.0 - 2.0)**2 + (2.5 - 2.0)**2)
    expected = 0.2 * (expected_vel + expected_torque)

    assert np.isclose(penalty, expected), f"Expected {expected}, got {penalty}"

if __name__ == "__main__":
    test_torque_penalty()
    test_action_smoothing()
    test_physics_constraints()
    print("✅ All tests passed.")
