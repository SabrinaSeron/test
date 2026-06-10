"""
observation_buffer.py — dimensions et assemblage des observations actor/critic.

ACTOR (58 valeurs) :
  joint_angles      (16)
  prev_action       (16)
  joint_vel         (16)
  contact_fingers    (4)  — 0.0 ou 1.0 par doigt
  pressure_fingers   (4)  — pression moyenne par doigt (N)
  contact_palm       (1)  — 0.0 ou 1.0
  pressure_palm      (1)  — pression paume (N)

CRITIC (100 valeurs = 58 actor + 42) :
  motor_torques     (16)
  fingertip_pos     (12)
  obj_pos            (3)
  obj_rotvec         (3)
  obj_linvel         (3)
  obj_angvel         (3)
  fall_flag          (1)
  palm_to_obj_dist   (1)
"""

import jax
import jax.numpy as jnp

from tactile import TACTILE_DIM

# indices des grilles dans le vecteur tactile brut (13 valeurs)
_IDX_IF   = jnp.array([0, 1, 2])   # index  : md, px, tip
_IDX_MF   = jnp.array([3, 4, 5])   # majeur : md, px, tip
_IDX_RF   = jnp.array([6, 7, 8])   # annulaire : md, px, tip
_IDX_TH   = jnp.array([9, 10, 11]) # pouce  : px, ds, tip
_IDX_PALM = 12                      # paume

CONTACT_THRESHOLD = 0.15  # N

# dimensions actor
_DIM_JOINTS     = 16
_DIM_PREV_ACT   = 16
_DIM_JOINT_VEL  = 16
_DIM_CONTACT_F  = 4
_DIM_PRESSURE_F = 4
_DIM_CONTACT_P  = 1
_DIM_PRESSURE_P = 1

ACTOR_OBS_DIM = (
    _DIM_JOINTS + _DIM_PREV_ACT + _DIM_JOINT_VEL
    + _DIM_CONTACT_F + _DIM_PRESSURE_F
    + _DIM_CONTACT_P + _DIM_PRESSURE_P
)  # 58

# dimensions critic (en plus des obs actor)
_DIM_TORQUES   = 16
_DIM_FT_POS    = 12
_DIM_OBJ_POS   = 3
_DIM_OBJ_ROT   = 3
_DIM_OBJ_LVEL  = 3
_DIM_OBJ_AVEL  = 3
_DIM_FALL      = 1
_DIM_PALM_DIST = 1

CRITIC_OBS_DIM = ACTOR_OBS_DIM + (
    _DIM_TORQUES + _DIM_FT_POS
    + _DIM_OBJ_POS + _DIM_OBJ_ROT + _DIM_OBJ_LVEL + _DIM_OBJ_AVEL
    + _DIM_FALL + _DIM_PALM_DIST
)  # 100


def _quat_to_rotvec(quat: jax.Array) -> jax.Array:
    """(N, 4) wxyz -> (N, 3) vecteur de rotation."""
    w = jnp.clip(quat[..., 0], -1.0, 1.0)
    xyz = quat[..., 1:]
    sign = jnp.where(w >= 0, 1.0, -1.0)
    w = w * sign
    xyz = xyz * sign[..., None]
    angle = 2.0 * jnp.arccos(w)
    sin_half = jnp.sqrt(jnp.maximum(1.0 - w ** 2, 1e-12))
    axis = xyz / sin_half[..., None]
    return axis * angle[..., None]


def _extract_tactile_features(tactile: jax.Array):
    """
    Depuis le vecteur tactile brut (N, 13), extrait :
      contact_fingers  : (N, 4) — 1.0 si doigt en contact, 0.0 sinon
      pressure_fingers : (N, 4) — pression moyenne par doigt (N)
      contact_palm     : (N, 1) — 1.0 si paume en contact
      pressure_palm    : (N, 1) — pression paume (N)

    APPELLEE UNE SEULE FOIS par step dans env_wrapper.step().
    Le resultat est passe a la fois a build_asymmetric_observation
    et a compute_catching_reward via env.current_*.
    """
    p_if   = tactile[:, _IDX_IF].mean(axis=-1, keepdims=True)
    p_mf   = tactile[:, _IDX_MF].mean(axis=-1, keepdims=True)
    p_rf   = tactile[:, _IDX_RF].mean(axis=-1, keepdims=True)
    p_th   = tactile[:, _IDX_TH].mean(axis=-1, keepdims=True)
    p_palm = tactile[:, _IDX_PALM:_IDX_PALM + 1]

    pressure_fingers = jnp.concatenate([p_if, p_mf, p_rf, p_th], axis=-1)  # (N, 4)
    pressure_palm    = p_palm                                                  # (N, 1)

    contact_fingers = (pressure_fingers > CONTACT_THRESHOLD).astype(jnp.float32)
    contact_palm    = (pressure_palm    > CONTACT_THRESHOLD).astype(jnp.float32)

    return contact_fingers, pressure_fingers, contact_palm, pressure_palm


@jax.jit
def build_asymmetric_observation(
    joint_angles:     jax.Array,   # (N, 16)
    prev_action:      jax.Array,   # (N, 16)
    joint_vel:        jax.Array,   # (N, 16)
    contact_fingers:  jax.Array,   # (N, 4)
    pressure_fingers: jax.Array,   # (N, 4)
    contact_palm:     jax.Array,   # (N, 1)
    pressure_palm:    jax.Array,   # (N, 1)
    motor_torques:    jax.Array,   # (N, 16)
    fingertip_pos:    jax.Array,   # (N, 12)
    obj_pos:          jax.Array,   # (N, 3)
    obj_quat:         jax.Array,   # (N, 4)
    obj_linvel:       jax.Array,   # (N, 3)
    obj_angvel:       jax.Array,   # (N, 3)
    fall_flag:        jax.Array,   # (N,)
    palm_pos:         jax.Array,   # (N, 3)
) -> jax.Array:                    # (N, 100)
    """
    Assemble l observation complete (N, 100).
    Les premiers ACTOR_OBS_DIM=58 elements = obs acteur.
    Les 42 suivants = infos supplementaires pour le critic.
    env_wrapper.py decoupera obs[:, :ACTOR_OBS_DIM] pour l acteur.
    """
    # obs acteur (58)
    actor_obs = jnp.concatenate([
        joint_angles,    # 16
        prev_action,     # 16
        joint_vel,       # 16
        contact_fingers, #  4
        pressure_fingers,#  4
        contact_palm,    #  1
        pressure_palm,   #  1
    ], axis=-1)          # 58

    # infos supplementaires critic (42)
    obj_rotvec  = _quat_to_rotvec(obj_quat)
    fall        = fall_flag[:, None].astype(jnp.float32)
    palm_to_obj = jnp.linalg.norm(obj_pos - palm_pos, axis=-1, keepdims=True)

    return jnp.concatenate([
        actor_obs,       # 58
        motor_torques,   # 16
        fingertip_pos,   # 12
        obj_pos,         #  3
        obj_rotvec,      #  3
        obj_linvel,      #  3
        obj_angvel,      #  3
        fall,            #  1
        palm_to_obj,     #  1
    ], axis=-1)           # 100
