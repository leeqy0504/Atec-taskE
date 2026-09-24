from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_contact_with_logging(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_norm = net_contact_forces.norm(dim=-1)
    max_force_per_body = force_norm.max(dim=1)[0]
    terminated = max_force_per_body.max(dim=1)[0] > threshold

    body_names = list(getattr(contact_sensor, "body_names", []))
    selected_body_names = [body_names[idx] for idx in sensor_cfg.body_ids if idx < len(body_names)]

    debug_entries = []
    max_force_per_body_cpu = max_force_per_body.detach().cpu()
    terminated_cpu = terminated.detach().cpu()
    for env_index in range(max_force_per_body_cpu.shape[0]):
        body_force_pairs = []
        for body_name, force_value in zip(selected_body_names, max_force_per_body_cpu[env_index].tolist()):
            body_force_pairs.append((body_name, float(force_value)))
        body_force_pairs.sort(key=lambda item: item[1], reverse=True)
        debug_entries.append(
            {
                "terminated": bool(terminated_cpu[env_index].item()),
                "threshold": float(threshold),
                "body_force_pairs": body_force_pairs,
            }
        )

    env._last_illegal_contact_debug = debug_entries
    return terminated
