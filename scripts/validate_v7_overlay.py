"""Static checks for the Tri-Fair v7 overlay."""

from src.config.v7_profiles import TRI_FAIR_V7_CONFIG, V7_STUDY_VERSION
from src.tri_fair_v7 import TRI_FAIR_V7_METHOD_VERSION, _simplex_reference_directions


def main() -> None:
    params = dict(TRI_FAIR_V7_CONFIG.optimizer_params)
    assert TRI_FAIR_V7_CONFIG.name == "Tri-Fair-v7"
    assert params["archive_cap"] >= 12
    assert params["smart_start_count"] >= 6
    assert len(_simplex_reference_directions(4)) == 15
    assert V7_STUDY_VERSION.startswith("7.0-")
    assert TRI_FAIR_V7_METHOD_VERSION.startswith("7.0-")
    print("Tri-Fair v7 static validation passed.")


if __name__ == "__main__":
    main()
