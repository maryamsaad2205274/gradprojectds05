"""
User-facing validation messages only.
Face detection is not used to block uploads — the landmark model decides validity.
"""

MSG_ANALYSIS_FAILED = (
    "Could not analyze this photo. Please upload a clearer face image."
)

# Legacy aliases used by templates / app
MSG_INVALID = MSG_ANALYSIS_FAILED
FRIENDLY_FAIL = MSG_ANALYSIS_FAILED
MSG_NO_FACE = MSG_ANALYSIS_FAILED
MSG_LANDMARKS_FAILED = MSG_ANALYSIS_FAILED
MSG_LOW_QUALITY = MSG_ANALYSIS_FAILED
MSG_MULTIPLE_FACES = MSG_ANALYSIS_FAILED
MSG_WRONG_SIDE_FOR_FRONT = MSG_ANALYSIS_FAILED
MSG_WRONG_FRONT_FOR_SIDE = MSG_ANALYSIS_FAILED
