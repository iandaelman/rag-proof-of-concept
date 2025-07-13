from enum import Enum


class RetrievalMethod(Enum):
    SIMILARITY_SEARCH = "similarity"
    MMR = "mmr"
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"