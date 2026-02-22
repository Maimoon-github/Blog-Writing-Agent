# config/settings.py
# Centralized configuration for paddleaurum.com autonomous blog agent.

# Maximum number of revision iterations in the reflection loop.
MAX_ITERATIONS = 3

# SEO score threshold to consider an article ready for publishing without forced human review.
SEO_THRESHOLD = 85

# Default tone for generated articles. Options: "coach", "expert", "beginner-friendly".
DEFAULT_TONE = "coach"