"""
rewards.py — fonctions de recompense pour la LEAP Hand.

IMPORTANT : les features tactiles (contact_fingers, pressure_fingers,
contact_palm, pressure_palm) sont calculees UNE SEULE FOIS dans
observation_buffer._extract_tactile_features() et passees directement ici.
On ne recalcule rien depuis les 13 valeurs brutes.

Structure de la recompense :

BONUS :
  alive_bonus          : +alive_bonus  a chaque step ou le cube n est pas tombe
  contact_bonus        : +(n_doigts_contact - 1) * scale  (encourage plusieurs doigts)
  pressure_bonus       : +scale par doigt en contact ET dans [P_LOW, P_HIGH]
  palm_bonus           : +scale si la paume touche (bonus doux)

PENALITES :
  slip_penalty         : proportionnel a la vitesse du cube (glissement)
  torque_penalty       : proportionnel aux couples moteurs^2 (effort inutile)
  low_pressure_penalty : par doigt en contact mais pression trop faible
  high_pressure_penalty: proportionnel a l exces au dessus de P_HIGH
  drop_term            : grosse penalite si le cube tombe

SEUILS :
  P_LOW  = 0.20 N  -- pression minimale pour une saisie stable
  P_HIGH = 2.0  N  -- pression maximale acceptable
"""

import jax
import jax.numpy as jnp

P_LOW  = 0.20   # N -- pression minimale acceptable
P_HIGH = 2.0    # N -- pression maximale acceptable


def compute_tactile_reward(
    contact_fingers:   jax.Array,   # (N, 4) -- 0.0 ou 1.0 par doigt
    pressure_fingers:  jax.Array,   # (N, 4) -- pression moyenne par doigt en N
    contact_palm:      jax.Array,   # (N, 1) -- 0.0 ou 1.0
    pressure_palm:     jax.Array,   # (N, 1) -- pression paume en N
    contact_bonus_scale:    float = 1.0,
    pressure_bonus_scale:   float = 0.5,
    palm_bonus_scale:       float = 0.3,
    low_pressure_scale:     float = 0.5,
    high_pressure_scale:    float = 0.2,
) -> tuple:
    """
    Calcule les termes de recompense tactiles.

    Les 4 arguments d entree viennent directement de
    observation_buffer._extract_tactile_features() -- pas de recalcul.

    Retourne
    --------
    reward_tactile : (N,)
    info           : dict des termes individuels pour le logging
    """

    # ── nombre de doigts en contact ──────────────────────────────────────────
    # contact_fingers est deja (N, 4) de 0.0/1.0
    # .sum(axis=-1) additionne les 4 colonnes → (N,) = nb de doigts en contact
    # Exemple : [1.0, 0.0, 1.0, 1.0] → 3.0

    n_contacts = contact_fingers.sum(axis=-1)   # (N,)

    # ── BONUS plusieurs doigts en contact ────────────────────────────────────
    # On soustrait 1 pour ne recompenser qu a partir de 2 doigts.
    # jnp.maximum(..., 0) evite les valeurs negatives si 0 ou 1 doigt.
    # Exemples :
    #   0 doigt  → max(0-1, 0) = 0    → pas de bonus
    #   1 doigt  → max(1-1, 0) = 0    → pas de bonus
    #   2 doigts → max(2-1, 0) = 1.0  → +1.0 * scale
    #   3 doigts → max(3-1, 0) = 2.0  → +2.0 * scale
    #   4 doigts → max(4-1, 0) = 3.0  → +3.0 * scale

    contact_bonus = contact_bonus_scale * jnp.maximum(n_contacts - 1.0, 0.0)   # (N,)

    # ── BONUS pression dans la zone cible ────────────────────────────────────
    # Pour chaque doigt, on verifie deux conditions :
    #   1. la pression est >= P_LOW (0.20 N) -- assez forte
    #   2. la pression est <= P_HIGH (2.0 N) -- pas trop forte
    # jnp.logical_and combine les deux conditions.
    # On multiplie par contact_fingers pour ne compter QUE les doigts en contact.
    # Un doigt pas en contact avec pression=0 ne doit pas compter.

    in_range = jnp.logical_and(
        pressure_fingers >= P_LOW,
        pressure_fingers <= P_HIGH,
    ).astype(jnp.float32)               # (N, 4) de 0.0/1.0

    in_range_and_contact = in_range * contact_fingers   # (N, 4)
    pressure_bonus = pressure_bonus_scale * in_range_and_contact.sum(axis=-1)   # (N,)

    # ── BONUS paume ──────────────────────────────────────────────────────────
    # contact_palm est (N, 1), on squeeze pour obtenir (N,)
    # Le coefficient est petit (0.3) pour ne pas forcer la paume a toujours toucher.

    palm_bonus = palm_bonus_scale * contact_palm.squeeze(axis=-1)   # (N,)

    # ── PENALITE pression trop faible ────────────────────────────────────────
    # Condition : le doigt est en contact (contact_fingers=1.0)
    #             MAIS sa pression est < P_LOW (0.20 N)
    # = contact detecte mais instable, risque de glissement.
    # jnp.logical_and puis .astype(float32) → 0.0 ou 1.0 par doigt.

    too_low = jnp.logical_and(
        contact_fingers > 0.5,           # est en contact
        pressure_fingers < P_LOW,        # mais pression insuffisante
    ).astype(jnp.float32)                # (N, 4)

    low_pressure_penalty = -low_pressure_scale * too_low.sum(axis=-1)   # (N,)

    # ── PENALITE pression trop forte ─────────────────────────────────────────
    # jnp.maximum(p - P_HIGH, 0) = l exces au dessus de 2.0 N, ou 0 si en dessous.
    # Exemple : pression = 3.0 N → exces = 3.0 - 2.0 = 1.0 N → penalite = -0.2

    excess = jnp.maximum(pressure_fingers - P_HIGH, 0.0)               # (N, 4)
    high_pressure_penalty = -high_pressure_scale * excess.sum(axis=-1)  # (N,)

    # ── total tactile ─────────────────────────────────────────────────────────
    reward_tactile = (
        contact_bonus
        + pressure_bonus
        + palm_bonus
        + low_pressure_penalty
        + high_pressure_penalty
    )

    info = {
        'contact_bonus':          contact_bonus,
        'pressure_bonus':         pressure_bonus,
        'palm_bonus':             palm_bonus,
        'low_pressure_penalty':   low_pressure_penalty,
        'high_pressure_penalty':  high_pressure_penalty,
        'n_fingers_contact':      n_contacts,
    }

    return reward_tactile, info


def compute_catching_reward(
    obj_linvel:        jax.Array,   # (N, 3)
    obj_angvel:        jax.Array,   # (N, 3)
    obj_pos:           jax.Array,   # (N, 3)
    torques:           jax.Array,   # (N, 16)
    contact_fingers:   jax.Array,   # (N, 4) -- depuis _extract_tactile_features
    pressure_fingers:  jax.Array,   # (N, 4)
    contact_palm:      jax.Array,   # (N, 1)
    pressure_palm:     jax.Array,   # (N, 1)
    reset_height_threshold: float = -0.05,
    slip_vel_scale:    float = 0.1,
    torque_scale:      float = 0.0001,
    alive_bonus:       float = 2.0,
    drop_penalty:      float = -1000.0,
    contact_bonus_scale:   float = 1.0,
    pressure_bonus_scale:  float = 0.5,
    palm_bonus_scale:      float = 0.3,
    low_pressure_scale:    float = 0.5,
    high_pressure_scale:   float = 0.2,
) -> tuple:

    # ── termes existants d Eduardo (inchanges) ────────────────────────────────

    linvel_mag = jnp.linalg.norm(obj_linvel, axis=-1)
    angvel_mag = jnp.minimum(jnp.linalg.norm(obj_angvel, axis=-1), 10.0)
    slip_penalty = -slip_vel_scale * (linvel_mag + 0.5 * angvel_mag)

    torque_penalty = -torque_scale * jnp.sum(torques ** 2, axis=-1)

    alive = (obj_pos[:, 2] >= reset_height_threshold).astype(jnp.float32)
    alive_reward = alive_bonus * alive
    drop_term = (1.0 - alive) * drop_penalty

    # ── termes tactiles (nouveaux) ────────────────────────────────────────────

    reward_tactile, tactile_info = compute_tactile_reward(
        contact_fingers  = contact_fingers,
        pressure_fingers = pressure_fingers,
        contact_palm     = contact_palm,
        pressure_palm    = pressure_palm,
        contact_bonus_scale   = contact_bonus_scale,
        pressure_bonus_scale  = pressure_bonus_scale,
        palm_bonus_scale      = palm_bonus_scale,
        low_pressure_scale    = low_pressure_scale,
        high_pressure_scale   = high_pressure_scale,
    )

    # ── total ─────────────────────────────────────────────────────────────────
    total = alive_reward + slip_penalty + torque_penalty + drop_term + reward_tactile

    info = {
        'alive_reward':   alive_reward,
        'slip_penalty':   slip_penalty,
        'torque_penalty': torque_penalty,
        'drop_penalty':   drop_term,
        'obj_linvel_mag': linvel_mag,
        'obj_angvel_mag': angvel_mag,
        **tactile_info,
    }
    return total, info


def check_termination(
    object_pos:         jax.Array,
    progress_buf:       jax.Array,
    max_episode_length: int = 500,
    reset_height_threshold: float = -0.05,
) -> tuple:
    """Identique a Eduardo."""
    object_fallen = object_pos[:, 2] < reset_height_threshold
    has_nan = jnp.any(jnp.isnan(object_pos), axis=-1)
    has_inf = jnp.any(jnp.isinf(object_pos), axis=-1)
    extreme = jnp.any(jnp.abs(object_pos) > 10.0, axis=-1)
    physics_explosion = has_nan | has_inf | extreme
    termination = object_fallen | physics_explosion
    reset_mask = termination | (progress_buf >= max_episode_length)
    return reset_mask, termination
