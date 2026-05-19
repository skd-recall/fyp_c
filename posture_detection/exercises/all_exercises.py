"""
All Exercise Implementations
Complete implementations of Squat, Pushup, Plank, Lunge, and Bicep Curl
"""

from typing import Dict, Optional
import logging
from exercises.base_exercise import BaseExercise, ExerciseMeasurements, FormFeedback
from core.angle_calculator import (
    AngleCalculator, calculate_knee_angle, calculate_hip_angle,
    calculate_elbow_angle, calculate_shoulder_angle
)

logger = logging.getLogger(__name__)


class SquatExercise(BaseExercise):
    """
    Squat Exercise - Front-facing camera with 30-45 degree side turn recommended.

    Signals used (priority order):
    1. Knee angle            - primary rep counting signal
    2. Direction tracking    - solves up/down phase ambiguity
    3. Hip Y vs Knee Y       - depth (normalized to body height)
    4. Hip-knee-vertical     - thigh tilt / hip hinge proxy
    5. Back angle            - forward lean check (ALL phases)
    6. Knee valgus           - normalized to body width
    7. Tempo via time delta  - device-independent speed check
    """

    # ── Thresholds (beginner-friendly, inspired by reference code) ────────
    STANDING_THRESH    = 160   # knee_angle above this → standing
    BOTTOM_THRESH      =  95   # knee_angle below this → bottom
    DIRECTION_DEADBAND =   2.0 # degrees; smaller changes ignored for direction

    # Back lean: some forward lean is NORMAL. Only flag excessive.
    BACK_ANGLE_WARN    =  45   # degrees from vertical → yellow warning
    BACK_ANGLE_ERROR   =  60   # degrees from vertical → red error

    # Hip depth (normalized): positive = hip below knee (good)
    DEPTH_THRESH       = -0.02  # hip must be at least this close to knee level

    # Knee valgus normalized to hip width: flag if knee moves inward > 15% of hip width
    VALGUS_WARN        =  0.10
    VALGUS_ERROR       =  0.20

    # Tempo: max degrees per SECOND (device-independent)
    TEMPO_MAX_DEG_SEC  = 120.0  # faster than this = swinging

    def __init__(self):
        super().__init__('squat')

        # Direction tracking
        self.prev_knee_angle  = None
        self.direction        = 'down'   # 'down' = descending, 'up' = ascending

        # Tempo tracking (time-based, not frame-based)
        self._last_timestamp  = None
        self._last_angle      = None

        # Cold-start guard: ignore direction until we have seen a standing frame
        self._seen_standing   = False

        # Body proportion cache (filled from calibrator if available)
        self._body_height_px  = None   # total height in pixels from calibration
        self._hip_width_px    = None   # hip width in pixels from calibration

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_best_side(self, pts: dict) -> str:
        """Pick the side with better average landmark visibility."""
        left_vis  = sum(pts[n].visibility for n in
                        ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'] if n in pts)
        right_vis = sum(pts[n].visibility for n in
                        ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'] if n in pts)
        return 'LEFT' if left_vis >= right_vis else 'RIGHT'

    def _hip_knee_vertical(self, hip, knee) -> float:
        """Angle of thigh (hip→knee) from vertical. 0°=standing, ~80°=full squat."""
        return AngleCalculator.calculate_angle_from_vertical(hip, knee) or 0.0

    def _update_direction(self, knee_angle: float) -> None:
        """Update movement direction with deadband to avoid noise flicker."""
        if self.prev_knee_angle is not None:
            delta = knee_angle - self.prev_knee_angle
            if delta < -self.DIRECTION_DEADBAND:
                self.direction = 'down'   # knee angle decreasing = going down
            elif delta > self.DIRECTION_DEADBAND:
                self.direction = 'up'     # knee angle increasing = coming up
        self.prev_knee_angle = knee_angle

    def _calc_tempo(self, knee_angle: float, timestamp: float) -> float:
        """
        Returns angular velocity in degrees/second.
        Device-independent because we use real time, not frame count.
        """
        velocity = 0.0
        if self._last_timestamp is not None and self._last_angle is not None:
            dt = timestamp - self._last_timestamp
            if dt > 0.001:   # avoid division by near-zero
                velocity = abs(knee_angle - self._last_angle) / dt
        self._last_timestamp = timestamp
        self._last_angle     = knee_angle
        return velocity

    def _normalize_valgus(self, raw_valgus: float, hip_width: float) -> float:
        """Normalize valgus distance to hip width so it is resolution-independent."""
        if hip_width and hip_width > 0:
            return raw_valgus / hip_width
        return raw_valgus / 50.0   # fallback: assume 50px hip width

    def _normalize_depth(self, hip_depth_diff: float, body_height: float) -> float:
        """
        Normalize hip-depth difference to body height.
        Positive = hip below knee (good depth).
        """
        if body_height and body_height > 0:
            return hip_depth_diff / body_height
        return hip_depth_diff / 400.0  # fallback: assume 400px body height

    # ── Core methods ──────────────────────────────────────────────────────

    def extract_measurements(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> Optional[ExerciseMeasurements]:

        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None

        import time
        timestamp = time.time()

        try:
            side = self._get_best_side(landmark_points)
            opp  = 'RIGHT' if side == 'LEFT' else 'LEFT'

            shoulder = (landmark_points.get(f'{side}_SHOULDER') or
                        landmark_points.get(f'{opp}_SHOULDER'))
            hip      = landmark_points.get(f'{side}_HIP')
            knee     = landmark_points.get(f'{side}_KNEE')
            ankle    = landmark_points.get(f'{side}_ANKLE')

            if not all([hip, knee, ankle]):
                return None

            # ── 1. Primary angles ──────────────────────────────────────
            knee_angle = calculate_knee_angle(hip, knee, ankle)
            if knee_angle is None:
                return None

            hip_angle  = calculate_hip_angle(shoulder, hip, knee) if shoulder else 0.0
            back_angle = (AngleCalculator.calculate_angle_from_vertical(shoulder, hip)
                          if shoulder else 0.0) or 0.0

            # ── 2. Thigh tilt (hip hinge proxy) ───────────────────────
            hip_knee_vert = self._hip_knee_vertical(hip, knee)

            # ── 3. Hip depth vs knee depth (normalized) ───────────────
            # In image coords Y increases downward → hip.y > knee.y = hips below knees
            raw_depth_diff = hip.y - knee.y

            # Try to get body height from calibration, else estimate from frame
            body_height = self._body_height_px
            if body_height is None:
                # Estimate: shoulder-to-ankle distance
                if shoulder and ankle:
                    body_height = abs(ankle.y - shoulder.y)
                else:
                    body_height = image_height * 0.75

            norm_depth = self._normalize_depth(raw_depth_diff, body_height)

            # ── 4. Knee valgus (both sides for reliability) ───────────
            l_hip   = landmark_points.get('LEFT_HIP')
            l_knee  = landmark_points.get('LEFT_KNEE')
            l_ankle = landmark_points.get('LEFT_ANKLE')
            r_hip   = landmark_points.get('RIGHT_HIP')
            r_knee  = landmark_points.get('RIGHT_KNEE')
            r_ankle = landmark_points.get('RIGHT_ANKLE')

            # Hip width for normalization
            hip_width = self._hip_width_px
            if hip_width is None and l_hip and r_hip:
                hip_width = abs(l_hip.x - r_hip.x)
                if hip_width < 10:
                    hip_width = 50.0

            raw_valgus_l = 0.0
            raw_valgus_r = 0.0
            if l_hip and l_knee and l_ankle:
                raw_valgus_l = abs(
                    AngleCalculator.calculate_knee_valgus(l_hip, l_knee, l_ankle) or 0.0
                )
            if r_hip and r_knee and r_ankle:
                raw_valgus_r = abs(
                    AngleCalculator.calculate_knee_valgus(r_hip, r_knee, r_ankle) or 0.0
                )

            # Use the worse side, normalized
            raw_valgus   = max(raw_valgus_l, raw_valgus_r)
            norm_valgus  = self._normalize_valgus(raw_valgus, hip_width)

            # ── 5. Tempo (degrees/second, device-independent) ─────────
            tempo = self._calc_tempo(knee_angle, timestamp)

            # ── 6. Direction + phase ───────────────────────────────────
            self._update_direction(knee_angle)

            angles = {
                'knee_angle':      knee_angle,
                'hip_angle':       hip_angle,
                'back_angle':      back_angle,
                'hip_knee_vert':   hip_knee_vert,
                'norm_depth':      norm_depth,
                'norm_valgus':     norm_valgus,
                'tempo':           tempo,
            }

            phase = self.determine_phase(
                ExerciseMeasurements(
                    angles=angles, alignments={},
                    distances={}, phase='', is_valid=True
                )
            )

            return ExerciseMeasurements(
                angles=angles,
                alignments={},
                distances={'norm_depth': norm_depth},
                phase=phase,
                is_valid=True
            )

        except Exception as e:
            logger.error(f"Squat extract error: {e}")
            return None

    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        """
        Direction-aware phase detection.
        Cold-start guard ensures we never start in a mid-squat phase.
        """
        knee_angle = measurements.angles.get('knee_angle', 170)

        # Mark when user has been seen standing (cold-start fix)
        if knee_angle >= self.STANDING_THRESH:
            self._seen_standing = True

        # If never seen standing yet, always return standing
        # (prevents wrong phase on first detection if user is already squatting)
        if not self._seen_standing:
            return 'standing'

        # Clear zones
        if knee_angle >= self.STANDING_THRESH:
            return 'standing'
        if knee_angle <= self.BOTTOM_THRESH:
            return 'bottom'

        # Mid-range: use direction
        return 'descent' if self.direction == 'down' else 'ascent'

    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
            corrections = []
            messages    = []

            knee_angle    = measurements.angles.get('knee_angle',    170)
            back_angle    = measurements.angles.get('back_angle',      0)
            hip_knee_vert = measurements.angles.get('hip_knee_vert',   0)
            norm_depth    = measurements.angles.get('norm_depth',      0)
            norm_valgus   = measurements.angles.get('norm_valgus',     0)
            tempo         = measurements.angles.get('tempo',           0)
            phase         = measurements.phase

            # ── 1. Depth check — BOTTOM ONLY ────────────────────────────────
            # Only check depth when fully at bottom, not during descent
            if phase == 'bottom':
                if norm_depth < self.DEPTH_THRESH:
                    corrections.append(
                        "Go a little deeper — try to bring hips to knee level"
                    )

            # ── 2. Hip hinge — BOTTOM ONLY ──────────────────────────────────
            # Do NOT check during descent — thigh starts vertical and that is correct
            # Only flag at bottom if the person has not hinged enough
            if phase == 'bottom':
                if hip_knee_vert < 55:
                    corrections.append(
                        "Sit back more — push hips backward as you squat"
                    )

            # ── 3. Forward lean — phase-aware thresholds ─────────────────────
            # During STANDING: flag any lean (person should be upright)
            # During DESCENT: only flag severe lean (some is natural early on)
            # During BOTTOM: standard thresholds apply
            # During ASCENT: only flag severe (hips shooting back is common)
            if phase == 'standing':
                if back_angle > 20:
                    corrections.append(
                        f"Stand up straight — reduce forward lean ({back_angle:.0f}°)"
                    )
            elif phase == 'descent':
                # Very lenient during descent — only flag truly excessive lean
                if back_angle > self.BACK_ANGLE_ERROR + 15:   # 75° threshold
                    corrections.append(
                        f"Too much forward lean while going down ({back_angle:.0f}°) — "
                        f"chest up"
                    )
            elif phase == 'bottom':
                if back_angle > self.BACK_ANGLE_ERROR:         # 60°
                    corrections.append(
                        f"Excessive forward lean at bottom ({back_angle:.0f}°) — "
                        f"chest up, keep torso more upright"
                    )
                elif back_angle > self.BACK_ANGLE_WARN:        # 45°
                    corrections.append(
                        f"Slight forward lean ({back_angle:.0f}°) — "
                        f"try to keep chest higher"
                    )
            elif phase == 'ascent':
                # Flag hips shooting back during ascent
                if back_angle > self.BACK_ANGLE_ERROR + 10:   # 70°
                    corrections.append(
                        f"Hips shooting back on the way up ({back_angle:.0f}°) — "
                        f"drive chest up as you stand"
                    )

            # ── 4. Knee valgus — DESCENT, BOTTOM, ASCENT only ───────────────
            # Never check when standing — natural variation at rest
            if phase in ('descent', 'bottom', 'ascent'):
                if norm_valgus > self.VALGUS_ERROR:
                    corrections.append(
                        "Knees collapsing inward — push knees out over your toes"
                    )
                elif norm_valgus > self.VALGUS_WARN:
                    corrections.append(
                        "Knees drifting slightly inward — push them out"
                    )

            # ── 5. Full lockout — STANDING only ──────────────────────────────
            if phase == 'standing' and knee_angle < 150:
                corrections.append(
                    "Stand up fully between reps — extend your legs completely"
                )

            # ── 6. Tempo ─────────────────────────────────────────────────────
            if tempo > self.TEMPO_MAX_DEG_SEC:
                corrections.append(
                    "Slow down — take 2 to 3 seconds going down"
                )

            # ── Result ───────────────────────────────────────────────────────
            is_correct = len(corrections) == 0
            severity   = (
                'good'    if is_correct else
                'warning' if len(corrections) == 1 else
                'error'
            )

            if is_correct:
                if phase == 'bottom':
                    messages.append("Good depth! Drive through heels to stand up.")
                elif phase == 'standing':
                    messages.append("Good — ready for next rep!")
                elif phase == 'descent':
                    messages.append("Good descent — sit back and down.")
                else:
                    messages.append("Good — drive up through your heels!")

            return FormFeedback(
                is_correct=is_correct,
                messages=messages,
                corrections=corrections,
                severity=severity
            )

    def set_body_proportions(self, body_height_px: float, hip_width_px: float):
        """
        Called after calibration to give the exercise class
        body proportion data for threshold normalization.
        """
        self._body_height_px = body_height_px
        self._hip_width_px   = hip_width_px
        logger.info(
            f"Squat proportions set: height={body_height_px:.0f}px "
            f"hip_width={hip_width_px:.0f}px"
        )

    def _compute_measurements_from_points(self, landmark_points, width, height):
        try:
            side  = self._get_best_side(landmark_points)
            hip   = landmark_points.get(f'{side}_HIP')
            knee  = landmark_points.get(f'{side}_KNEE')
            ankle = landmark_points.get(f'{side}_ANKLE')

            if not all([hip, knee, ankle]):
                return None

            knee_angle = calculate_knee_angle(hip, knee, ankle)
            if knee_angle is None:
                return None

            angles = {
                'knee_angle': knee_angle,
                'hip_angle': 0.0, 'back_angle': 0.0,
                'hip_knee_vert': 0.0, 'norm_depth': 0.0,
                'norm_valgus': 0.0, 'tempo': 0.0
            }
            temp  = ExerciseMeasurements(
                angles=angles, alignments={}, distances={},
                phase='', is_valid=True
            )
            phase = self.determine_phase(temp)
            return ExerciseMeasurements(
                angles=angles, alignments={}, distances={},
                phase=phase, is_valid=True
            )
        except Exception as e:
            logger.error(f"Squat compute error: {e}")
            return None

class PushupExercise(BaseExercise):
    """
    Push-up Exercise — requires SIDE VIEW camera.

    Signals (priority order):
    1. Elbow angle              - primary rep signal
    2. Direction tracking       - solves up/down phase ambiguity
    3. Confidence filtering     - rejects low-visibility frames silently
    4. Body sag vs pike         - two separate checks, distinct feedback
    5. Elbow flare              - injury risk detection
    6. Shoulder depth           - catches partial reps
    7. Tempo                    - device-independent (degrees/second)
    8. Temporal persistence     - corrections only after 0.4s of bad form
    """

    # ── Rep thresholds ────────────────────────────────────────────────────
    UP_THRESH          = 155
    DOWN_THRESH        =  85
    DIRECTION_DEADBAND =   2.0

    # ── Body alignment ────────────────────────────────────────────────────
    SAG_WARN           =  12
    SAG_ERROR          =  20
    PIKE_WARN          =  12
    PIKE_ERROR         =  20

    # ── Elbow flare ───────────────────────────────────────────────────────
    FLARE_WARN         =  55
    FLARE_ERROR        =  70

    # ── Shoulder depth ────────────────────────────────────────────────────
    SHOULDER_DEPTH_TOL =  10   # pixels

    # ── Tempo ─────────────────────────────────────────────────────────────
    TEMPO_MAX          = 130.0  # degrees/second

    # ── Confidence filtering ──────────────────────────────────────────────
    MIN_LANDMARK_VISIBILITY = 0.5

    # ── Temporal persistence (seconds bad form must persist before flagging)
    CORRECTION_PERSIST_SEC  = 0.4

    def __init__(self):
        super().__init__('pushup')

        # Direction tracking
        self.prev_elbow_angle = None
        self.direction        = 'down'
        self._seen_up         = False

        # Tempo tracking
        self._last_timestamp  = None
        self._last_angle      = None

        # Body proportion cache
        self._body_height_px  = None

        # Temporal persistence: track when each issue first appeared
        # key = issue name, value = timestamp when it first appeared
        self._issue_first_seen: dict = {}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_best_side(self, pts: dict) -> str:
        lv = sum(pts[n].visibility for n in
                 ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
                  'LEFT_HIP', 'LEFT_ANKLE'] if n in pts)
        rv = sum(pts[n].visibility for n in
                 ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
                  'RIGHT_HIP', 'RIGHT_ANKLE'] if n in pts)
        return 'LEFT' if lv >= rv else 'RIGHT'

    def _check_landmark_confidence(self, pts: dict, side: str) -> bool:
        """
        Reject frame if any key landmark on chosen side is below threshold.
        Returns True if frame is usable, False if it should be skipped.
        """
        key_landmarks = [
            f'{side}_SHOULDER', f'{side}_ELBOW', f'{side}_WRIST'
        ]
        for name in key_landmarks:
            pt = pts.get(name)
            if pt is None or pt.visibility < self.MIN_LANDMARK_VISIBILITY:
                logger.debug(f"Low confidence landmark rejected: {name}")
                return False
        return True

    def _update_direction(self, elbow_angle: float) -> None:
        if self.prev_elbow_angle is not None:
            delta = elbow_angle - self.prev_elbow_angle
            if delta < -self.DIRECTION_DEADBAND:
                self.direction = 'down'
            elif delta > self.DIRECTION_DEADBAND:
                self.direction = 'up'
        self.prev_elbow_angle = elbow_angle

    def _calc_tempo(self, elbow_angle: float, timestamp: float) -> float:
        velocity = 0.0
        if self._last_timestamp is not None and self._last_angle is not None:
            dt = timestamp - self._last_timestamp
            if dt > 0.001:
                velocity = abs(elbow_angle - self._last_angle) / dt
        self._last_timestamp = timestamp
        self._last_angle     = elbow_angle
        return velocity

    def _check_sag_pike(self, shoulder, hip, ankle) -> tuple:
        """
        Measure hip deviation from the shoulder-ankle straight line.
        Returns (sag_deg, pike_deg) — only one will be non-zero.
        sag  = hip drops below the line (weak core)
        pike = hip rises above the line (compensating)
        """
        if not all([shoulder, hip, ankle]):
            return 0.0, 0.0

        line_dx  = ankle.x - shoulder.x
        line_dy  = ankle.y - shoulder.y
        line_len = ((line_dx ** 2) + (line_dy ** 2)) ** 0.5

        if line_len < 1:
            return 0.0, 0.0

        cross       = (line_dx * (shoulder.y - hip.y)) - (line_dy * (shoulder.x - hip.x))
        perp_dist   = cross / line_len
        body_h      = self._body_height_px or line_len
        deviation   = (perp_dist / body_h) * 90.0

        return (deviation, 0.0) if deviation > 0 else (0.0, abs(deviation))

    def _is_persistent(self, issue_key: str, is_present: bool, now: float) -> bool:
        """
        Returns True only if the issue has been continuously present
        for at least CORRECTION_PERSIST_SEC seconds.
        Clears the timer when the issue disappears.
        """
        if is_present:
            if issue_key not in self._issue_first_seen:
                self._issue_first_seen[issue_key] = now
            return (now - self._issue_first_seen[issue_key]) >= self.CORRECTION_PERSIST_SEC
        else:
            # Issue gone — reset its timer
            self._issue_first_seen.pop(issue_key, None)
            return False

    # ── Core methods ──────────────────────────────────────────────────────

    def extract_measurements(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> Optional[ExerciseMeasurements]:

        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None

        import time
        timestamp = time.time()

        try:
            side = self._get_best_side(landmark_points)
            opp  = 'RIGHT' if side == 'LEFT' else 'LEFT'

            # ── Confidence filter: reject bad frames early ─────────────
            if not self._check_landmark_confidence(landmark_points, side):
                return None   # frame silently skipped

            shoulder = landmark_points.get(f'{side}_SHOULDER')
            elbow    = landmark_points.get(f'{side}_ELBOW')
            wrist    = landmark_points.get(f'{side}_WRIST')
            hip      = (landmark_points.get(f'{side}_HIP') or
                        landmark_points.get(f'{opp}_HIP'))
            ankle    = (landmark_points.get(f'{side}_ANKLE') or
                        landmark_points.get(f'{opp}_ANKLE'))

            if not all([shoulder, elbow, wrist]):
                return None

            # ── 1. Primary elbow angle ─────────────────────────────────
            elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
            if elbow_angle is None:
                return None

            # ── 2. Body sag vs pike ────────────────────────────────────
            sag_deg, pike_deg = (0.0, 0.0)
            if hip and ankle:
                sag_deg, pike_deg = self._check_sag_pike(shoulder, hip, ankle)

            # ── 3. Elbow flare ─────────────────────────────────────────
            elbow_flare = 0.0
            if hip:
                elbow_flare = (
                    AngleCalculator.calculate_elbow_to_body_angle(
                        shoulder, elbow, hip
                    ) or 0.0
                )

            # ── 4. Shoulder depth ──────────────────────────────────────
            shoulder_depth_ok = True
            if elbow_angle < self.DOWN_THRESH + 15:
                shoulder_depth_ok = (
                    shoulder.y >= elbow.y - self.SHOULDER_DEPTH_TOL
                )

            # ── 5. Tempo ───────────────────────────────────────────────
            tempo = self._calc_tempo(elbow_angle, timestamp)

            # ── 6. Direction ───────────────────────────────────────────
            self._update_direction(elbow_angle)

            # Cache body height for sag/pike normalization
            if self._body_height_px is None and ankle:
                self._body_height_px = abs(ankle.y - shoulder.y)

            angles = {
                'elbow_angle':       elbow_angle,
                'sag_deg':           sag_deg,
                'pike_deg':          pike_deg,
                'elbow_flare':       elbow_flare,
                'shoulder_depth_ok': 1.0 if shoulder_depth_ok else 0.0,
                'tempo':             tempo,
                'timestamp':         timestamp,
            }

            phase = self.determine_phase(
                ExerciseMeasurements(
                    angles=angles, alignments={},
                    distances={}, phase='', is_valid=True
                )
            )

            return ExerciseMeasurements(
                angles=angles,
                alignments={'sag': sag_deg, 'pike': pike_deg},
                distances={},
                phase=phase,
                is_valid=True
            )

        except Exception as e:
            logger.error(f"Pushup extract error: {e}")
            return None

    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        elbow_angle = measurements.angles.get('elbow_angle', 170)

        if elbow_angle >= self.UP_THRESH:
            self._seen_up = True

        if not self._seen_up:
            return 'up'

        if elbow_angle >= self.UP_THRESH:
            return 'up'
        if elbow_angle <= self.DOWN_THRESH:
            return 'down'

        return 'descent' if self.direction == 'down' else 'ascent'

    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
        corrections = []
        messages    = []

        elbow_angle       = measurements.angles.get('elbow_angle',       170)
        sag_deg           = measurements.angles.get('sag_deg',             0)
        pike_deg          = measurements.angles.get('pike_deg',            0)
        elbow_flare       = measurements.angles.get('elbow_flare',         0)
        shoulder_depth_ok = measurements.angles.get('shoulder_depth_ok',   1)
        tempo             = measurements.angles.get('tempo',               0)
        now               = measurements.angles.get('timestamp', 0)
        phase             = measurements.phase

        # ── 1. Body SAG — persistent check ──────────────────────────────
        if self._is_persistent('sag_error', sag_deg > self.SAG_ERROR, now):
            corrections.append(
                f"Hips sagging badly — squeeze core and glutes "
                f"to keep body straight ({sag_deg:.0f}°)"
            )
        elif self._is_persistent('sag_warn', sag_deg > self.SAG_WARN, now):
            corrections.append(
                f"Hips dropping slightly — tighten your core "
                f"({sag_deg:.0f}°)"
            )

        # ── 2. Body PIKE — persistent check ─────────────────────────────
        if self._is_persistent('pike_error', pike_deg > self.PIKE_ERROR, now):
            corrections.append(
                f"Hips too high — lower them to form a straight line "
                f"shoulder to ankle ({pike_deg:.0f}°)"
            )
        elif self._is_persistent('pike_warn', pike_deg > self.PIKE_WARN, now):
            corrections.append(
                f"Hips slightly elevated — bring them down a little "
                f"({pike_deg:.0f}°)"
            )

        # ── 3. Elbow flare — persistent check ───────────────────────────
        if self._is_persistent('flare_error', elbow_flare > self.FLARE_ERROR, now):
            corrections.append(
                f"Elbows flaring too wide ({elbow_flare:.0f}°) — "
                f"tuck elbows closer to protect shoulders"
            )
        elif self._is_persistent('flare_warn', elbow_flare > self.FLARE_WARN, now):
            corrections.append(
                f"Elbows slightly wide ({elbow_flare:.0f}°) — "
                f"try tucking them a little closer"
            )

        # ── 4. Depth — at bottom position only ──────────────────────────
        if phase == 'down':
            if elbow_angle > self.DOWN_THRESH + 10:
                corrections.append(
                    f"Go lower — chest closer to ground "
                    f"(elbow: {elbow_angle:.0f}°, target: below {self.DOWN_THRESH}°)"
                )
            if shoulder_depth_ok < 1.0:
                corrections.append(
                    "Go lower — shoulder should reach elbow level at the bottom"
                )

        # ── 5. Full lockout — at top only ────────────────────────────────
        if phase == 'up' and elbow_angle < self.UP_THRESH - 10:
            corrections.append(
                f"Fully extend arms at the top "
                f"(elbow: {elbow_angle:.0f}°, target: above {self.UP_THRESH}°)"
            )

        # ── 6. Tempo ─────────────────────────────────────────────────────
        if self._is_persistent('tempo', tempo > self.TEMPO_MAX, now):
            corrections.append(
                "Slow down — control the movement. "
                "2 seconds down, 1 second up."
            )

        # ── Build result ──────────────────────────────────────────────────
        is_correct = len(corrections) == 0
        severity   = (
            'good'    if is_correct else
            'warning' if len(corrections) == 1 else
            'error'
        )

        if is_correct:
            if phase == 'down':
                messages.append("Good depth! Push up through your palms.")
            elif phase == 'up':
                messages.append("Arms extended — ready for next rep!")
            elif phase == 'descent':
                messages.append("Good descent — controlled and steady.")
            else:
                messages.append("Good — keep pushing!")

        return FormFeedback(
            is_correct=is_correct,
            messages=messages,
            corrections=corrections,
            severity=severity
        )

    def set_body_proportions(self, body_height_px: float):
        self._body_height_px = body_height_px
        logger.info(f"Pushup body height set: {body_height_px:.0f}px")

    def _compute_measurements_from_points(self, landmark_points, width, height):
        try:
            side     = self._get_best_side(landmark_points)
            shoulder = landmark_points.get(f'{side}_SHOULDER')
            elbow    = landmark_points.get(f'{side}_ELBOW')
            wrist    = landmark_points.get(f'{side}_WRIST')

            if not all([shoulder, elbow, wrist]):
                return None

            elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
            if elbow_angle is None:
                return None

            angles = {
                'elbow_angle': elbow_angle,
                'sag_deg': 0.0, 'pike_deg': 0.0,
                'elbow_flare': 0.0, 'shoulder_depth_ok': 1.0,
                'tempo': 0.0, 'timestamp': 0.0
            }
            temp  = ExerciseMeasurements(
                angles=angles, alignments={}, distances={},
                phase='', is_valid=True
            )
            phase = self.determine_phase(temp)
            return ExerciseMeasurements(
                angles=angles, alignments={}, distances={},
                phase=phase, is_valid=True
            )
        except Exception as e:
            logger.error(f"Pushup compute error: {e}")
            return None
        
class PlankExercise(BaseExercise):
    """Plank exercise implementation"""
    
    def __init__(self):
        super().__init__('plank')
    
    def extract_measurements(self, landmarks, image_width: int, image_height: int) -> Optional[ExerciseMeasurements]:
        result = self.extract_measurements_both_sides(landmarks, image_width, image_height)
        if result:
            return result
        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None
        
        try:
            shoulder = landmark_points['LEFT_SHOULDER']
            hip = landmark_points['LEFT_HIP']
            ankle = landmark_points['LEFT_ANKLE']
            elbow = landmark_points['LEFT_ELBOW']
            
            body_alignment = AngleCalculator.calculate_body_alignment(shoulder, hip, ankle)
            hip_angle = calculate_hip_angle(shoulder, hip, ankle)
            
            if None in [body_alignment, hip_angle]:
                return None
            
            angles = {
                'hip_angle': hip_angle,
                'body_alignment': body_alignment
            }
            
            return ExerciseMeasurements(
                angles=angles,
                alignments={'body': body_alignment},
                distances={},
                phase='hold',
                is_valid=True
            )
        except Exception as e:
            logger.error(f"Error extracting plank measurements: {e}")
            return None
    
    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        return 'hold'
    
    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
        messages = []
        corrections = []
    
        hip_angle = measurements.angles.get('hip_angle')
        body_alignment = measurements.alignments.get('body')
    
        # Adjust hip angle range
        hip_base_range = self.get_angle_range('hold', 'hip_angle')
        hip_adjusted = self.adjust_threshold_for_user(
            hip_base_range,
            getattr(self, 'calibrator', None)
        )
    
        if hip_angle:
            if hip_angle < hip_adjusted[0]:
                corrections.append("Hips sagging - engage core and lift")
            elif hip_angle > hip_adjusted[1]:
                corrections.append("Hips too high - lower to straight line")
        
        if body_alignment and body_alignment > 15:
            corrections.append("Maintain straight line from head to heels")
        
        is_correct = len(corrections) == 0
        severity = 'good' if is_correct else 'error'
        
        if is_correct:
            messages.append("Great plank hold!")
        
        return FormFeedback(
            is_correct=is_correct,
            messages=messages,
            corrections=corrections,
            severity=severity
        )
    def _compute_measurements_from_points(self, landmark_points, width, height):
        from ..core.angle_calculator import AngleCalculator, calculate_elbow_angle
        try:
            shoulder = landmark_points.get('LEFT_SHOULDER') or landmark_points.get('RIGHT_SHOULDER')
            elbow = landmark_points.get('LEFT_ELBOW') or landmark_points.get('RIGHT_ELBOW')
            hip = landmark_points.get('LEFT_HIP') or landmark_points.get('RIGHT_HIP')
            knee = landmark_points.get('LEFT_KNEE') or landmark_points.get('RIGHT_KNEE')
            ankle = landmark_points.get('LEFT_ANKLE') or landmark_points.get('RIGHT_ANKLE')

            if not all([shoulder, hip, ankle]):
                return None

            body_alignment = AngleCalculator.calculate_body_alignment(shoulder, hip, ankle)
            if body_alignment is None:
                return None

            angles = {'body_alignment': body_alignment}
            from exercises.base_exercise import ExerciseMeasurements
            temp = ExerciseMeasurements(angles=angles, alignments={}, distances={}, phase='', is_valid=True)
            phase = self.determine_phase(temp)
            return ExerciseMeasurements(angles=angles, alignments={}, distances={}, phase=phase, is_valid=True)
        except Exception as e:
            logger.error(f"Plank compute error: {e}")
            return None
    
    
class ShoulderPressExercise(BaseExercise):
    """Shoulder press (overhead press) exercise implementation"""
    
    def __init__(self):
        super().__init__('shoulder_press')
    
    def extract_measurements(self, landmarks, image_width: int, image_height: int) -> Optional[ExerciseMeasurements]:
        result = self.extract_measurements_both_sides(landmarks, image_width, image_height)
        if result:
            return result
        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None
        
        try:
            shoulder = landmark_points['LEFT_SHOULDER']
            elbow = landmark_points['LEFT_ELBOW']
            wrist = landmark_points['LEFT_WRIST']
            hip = landmark_points['LEFT_HIP']
            
            # Calculate angles
            elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
            shoulder_flexion = calculate_shoulder_angle(hip, shoulder, elbow)
            elbow_to_body = AngleCalculator.calculate_elbow_to_body_angle(shoulder, elbow, hip)
            
            if elbow_angle is None:
                return None
            
            angles = {
                'elbow_angle': elbow_angle,
                'shoulder_flexion': shoulder_flexion if shoulder_flexion else 0,
                'elbow_to_body_angle': elbow_to_body if elbow_to_body else 30
            }
            
            phase = self.determine_phase(ExerciseMeasurements(
                angles=angles, alignments={}, distances={}, phase='', is_valid=True
            ))
            
            return ExerciseMeasurements(
                angles=angles,
                alignments={},
                distances={},
                phase=phase,
                is_valid=True
            )
        except Exception as e:
            logger.error(f"Error extracting shoulder press measurements: {e}")
            return None
    
    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        elbow_angle = measurements.angles.get('elbow_angle', 90)
        
        if elbow_angle <= 105:
            return 'starting'
        elif elbow_angle <= 165:
            return 'pressing'
        elif elbow_angle >= 165:
            return 'lockout'
        else:
            return 'lowering'
    
    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
        messages = []
        corrections = []
    
        elbow_angle = measurements.angles.get('elbow_angle')
        elbow_to_body = measurements.angles.get('elbow_to_body_angle')
        phase = measurements.phase
    
        if phase == 'lockout':
            # Adjust lockout range
            lockout_base_range = self.get_angle_range('lockout', 'elbow_angle')
            lockout_adjusted = self.adjust_threshold_for_user(
                lockout_base_range,
                getattr(self, 'calibrator', None)
            )
        
            if elbow_angle and elbow_angle < lockout_adjusted[0]:
                corrections.append(f"Fully extend arms overhead - reach {int(lockout_adjusted[0])}°+")
    
        elif phase == 'starting':
            # Adjust starting position range
            start_base_range = self.get_angle_range('starting', 'elbow_angle')
            start_adjusted = self.adjust_threshold_for_user(
                start_base_range,
                getattr(self, 'calibrator', None)
            )
        
            if elbow_angle and not (start_adjusted[0] <= elbow_angle <= start_adjusted[1]):
                corrections.append(f"Start with elbows at shoulder level ({int(start_adjusted[0])}-{int(start_adjusted[1])}°)")
        # elif phase == 'starting':
        #     if elbow_angle and not self.is_angle_in_range(elbow_angle, 'starting', 'elbow_angle'):
        #         corrections.append("Start with elbows at shoulder level (90°)")
        
        if elbow_to_body and (elbow_to_body < 20 or elbow_to_body > 55):
            corrections.append("Keep elbows at 30-45° angle - don't flare wide")
        
        is_correct = len(corrections) == 0
        severity = 'good' if is_correct else 'warning'
        
        if is_correct:
            messages.append("Excellent shoulder press form!")
        
        return FormFeedback(
            is_correct=is_correct,
            messages=messages,
            corrections=corrections,
            severity=severity
        )
    def _compute_measurements_from_points(self, landmark_points, width, height):
        from ..core.angle_calculator import AngleCalculator, calculate_elbow_angle, calculate_shoulder_angle
        try:
            shoulder = landmark_points.get('LEFT_SHOULDER') or landmark_points.get('RIGHT_SHOULDER')
            elbow = landmark_points.get('LEFT_ELBOW') or landmark_points.get('RIGHT_ELBOW')
            wrist = landmark_points.get('LEFT_WRIST') or landmark_points.get('RIGHT_WRIST')
            hip = landmark_points.get('LEFT_HIP') or landmark_points.get('RIGHT_HIP')

            if not all([shoulder, elbow, wrist]):
                return None

            elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
            shoulder_flexion = calculate_shoulder_angle(hip, shoulder, elbow) if hip else 0
            elbow_to_body = AngleCalculator.calculate_elbow_to_body_angle(shoulder, elbow, hip) if hip else 30

            if elbow_angle is None:
                return None

            angles = {
                'elbow_angle': elbow_angle,
                'shoulder_flexion': shoulder_flexion if shoulder_flexion else 0,
                'elbow_to_body_angle': elbow_to_body if elbow_to_body else 30
            }
            from exercises.base_exercise import ExerciseMeasurements
            temp = ExerciseMeasurements(angles=angles, alignments={}, distances={}, phase='', is_valid=True)
            phase = self.determine_phase(temp)
            return ExerciseMeasurements(angles=angles, alignments={}, distances={}, phase=phase, is_valid=True)
        except Exception as e:
            logger.error(f"ShoulderPress compute error: {e}")
            return None




class BicepCurlExercise(BaseExercise):
    """
    Bicep Curl Exercise
    Camera: FRONT-FACING
    Key signals: elbow angle, upper arm vertical angle, shoulder stability, tempo
    """

    def __init__(self):
        super().__init__('bicep_curl')
        self._angle_history = []   # for tempo detection
        self._history_maxlen = 5

    def _get_best_side(self, landmark_points: dict) -> str:
        """Pick the side with better average visibility"""
        left_names = ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST']
        right_names = ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']

        left_vis = sum(
            landmark_points[n].visibility
            for n in left_names if n in landmark_points
        )
        right_vis = sum(
            landmark_points[n].visibility
            for n in right_names if n in landmark_points
        )
        return 'LEFT' if left_vis >= right_vis else 'RIGHT'

    def extract_measurements(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> Optional[ExerciseMeasurements]:

        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None

        try:
            # Pick the more visible/stable side
            side = self._get_best_side(landmark_points)

            shoulder = landmark_points.get(f'{side}_SHOULDER')
            elbow    = landmark_points.get(f'{side}_ELBOW')
            wrist    = landmark_points.get(f'{side}_WRIST')
            hip      = landmark_points.get(f'{side}_HIP') or \
                       landmark_points.get('LEFT_HIP') or \
                       landmark_points.get('RIGHT_HIP')

            if not all([shoulder, elbow, wrist]):
                return None

            # --- Primary angle: elbow flexion ---
            elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
            if elbow_angle is None:
                return None

            # --- Upper arm angle from vertical ---
            # When arm hangs at side correctly: ~0-20°
            # When elbow drifts forward: >40°
            upper_arm_angle = AngleCalculator.calculate_angle_from_vertical(
                shoulder, elbow
            ) or 0.0

            # --- Shoulder stability ---
            # hip->shoulder->elbow angle. Low = arm at side (good), high = swinging
            shoulder_angle = 0.0
            if hip:
                shoulder_angle = calculate_shoulder_angle(
                    hip, shoulder, elbow
                ) or 0.0

            # --- Tempo: track angle velocity ---
            self._angle_history.append(elbow_angle)
            if len(self._angle_history) > self._history_maxlen:
                self._angle_history.pop(0)

            # Degrees per frame (approximate velocity)
            angle_velocity = 0.0
            if len(self._angle_history) >= 2:
                # Average change per frame over last N frames
                changes = [
                    abs(self._angle_history[i] - self._angle_history[i-1])
                    for i in range(1, len(self._angle_history))
                ]
                angle_velocity = sum(changes) / len(changes)

            angles = {
                'elbow_angle':     elbow_angle,
                'upper_arm_angle': upper_arm_angle,
                'shoulder_angle':  shoulder_angle,
                'angle_velocity':  angle_velocity,
            }

            # Determine phase using direction-aware logic
            phase = self.determine_phase(
                ExerciseMeasurements(
                    angles=angles, alignments={},
                    distances={}, phase='', is_valid=True
                )
            )

            return ExerciseMeasurements(
                angles=angles,
                alignments={},
                distances={},
                phase=phase,
                is_valid=True
            )

        except Exception as e:
            logger.error(f"BicepCurl extract error: {e}")
            return None

    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        """
        Direction-aware phase detection.
        Uses elbow angle thresholds with clear zones:
          extended:   165-180  (arm fully straight at bottom)
          lifting:     60-165  going UP
          contracted:  25-60   (arm fully bent at top)
          lowering:    60-165  going DOWN
        """
        elbow_angle = measurements.angles.get('elbow_angle', 180)

        if elbow_angle >= 160:
            return 'extended'
        elif elbow_angle <= 55:
            return 'contracted'
        else:
            # Mid-range: need direction from rep_counter if available
            # Default to 'lifting' — direction tracking in rep_counter
            # handles the actual rep logic correctly regardless
            return 'lifting'

    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
        corrections = []
        messages = []

        elbow_angle    = measurements.angles.get('elbow_angle', 90)
        upper_arm_angle = measurements.angles.get('upper_arm_angle', 0)
        shoulder_angle  = measurements.angles.get('shoulder_angle', 0)
        angle_velocity  = measurements.angles.get('angle_velocity', 0)
        phase           = measurements.phase

        # ── 1. Elbow drift (upper arm from vertical) ──────────────────────
        # 0-30°  : good (natural armpit opening is fine)
        # 30-45° : soft warning
        # >45°   : clear error
        if upper_arm_angle > 45:
            corrections.append(
                f"Elbows swinging too far forward ({upper_arm_angle:.0f}°) "
                f"- keep upper arms at your sides"
            )
        elif upper_arm_angle > 30:
            corrections.append(
                f"Elbows drifting slightly ({upper_arm_angle:.0f}°) "
                f"- try to keep them closer to your body"
            )

        # ── 2. Shoulder stability (all phases) ────────────────────────────
        # Lowering is more lenient (natural slight movement allowed)
        shoulder_limit = 35 if phase == 'lowering' else 28
        if shoulder_angle > shoulder_limit:
            corrections.append(
                f"Shoulders moving too much ({shoulder_angle:.0f}°) "
                f"- keep shoulders still, isolate the biceps"
            )

        # ── 3. Full contraction at top ────────────────────────────────────
        if phase == 'contracted':
            if elbow_angle > 60:
                corrections.append(
                    f"Curl higher - squeeze at the top "
                    f"(current: {elbow_angle:.0f}°, target: below 55°)"
                )

        # ── 4. Full extension at bottom ───────────────────────────────────
        if phase == 'extended':
            if elbow_angle < 155:
                corrections.append(
                    f"Fully extend arms at bottom "
                    f"(current: {elbow_angle:.0f}°, target: above 160°)"
                )

        # ── 5. Tempo / momentum check ─────────────────────────────────────
        # >15° change per frame at 30fps = ~450°/sec = way too fast
        # Typical good rep: 2-4 seconds = ~50-80°/sec = ~1.5-2.5° per frame
        if angle_velocity > 12:
            corrections.append(
                f"Slow down - you are swinging the weight "
                f"(speed: {angle_velocity:.1f}°/frame). "
                f"Take 2-3 seconds up and down."
            )

        is_correct = len(corrections) == 0
        severity = 'good' if is_correct else 'warning'

        if is_correct:
            if phase == 'contracted':
                messages.append("Perfect! Squeeze the bicep at the top!")
            elif phase == 'extended':
                messages.append("Good starting position - ready to curl!")
            else:
                messages.append("Good form - keep it up!")

        return FormFeedback(
            is_correct=is_correct,
            messages=messages,
            corrections=corrections,
            severity=severity
        )

    def _compute_measurements_from_points(self, landmark_points, width, height):
        try:
            side = self._get_best_side(landmark_points)
            shoulder = landmark_points.get(f'{side}_SHOULDER')
            elbow    = landmark_points.get(f'{side}_ELBOW')
            wrist    = landmark_points.get(f'{side}_WRIST')

            if not all([shoulder, elbow, wrist]):
                return None

            elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
            if elbow_angle is None:
                return None

            angles = {'elbow_angle': elbow_angle, 'upper_arm_angle': 0,
                      'shoulder_angle': 0, 'angle_velocity': 0}
            temp = ExerciseMeasurements(
                angles=angles, alignments={}, distances={},
                phase='', is_valid=True
            )
            phase = self.determine_phase(temp)
            return ExerciseMeasurements(
                angles=angles, alignments={}, distances={},
                phase=phase, is_valid=True
            )
        except Exception as e:
            logger.error(f"BicepCurl compute error: {e}")
            return None