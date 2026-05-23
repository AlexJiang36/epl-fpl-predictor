from app.schemas.model_metadata import ModelMetadataArtifact
from app.utils.model_metadata_store import save_model_metadata_artifact


def main() -> None:
    artifacts = [
        ModelMetadataArtifact(
            model_name="baseline_rollavg_v0",
            task_type="player_points",
            feature_version="v0",
            training_window_start_gw=1,
            training_window_end_gw=27,
            evaluation_start_gw=28,
            evaluation_end_gw=30,
            metrics_summary={
                "overall_mae": 1.0351,
            },
            notes="Baseline rolling-average player model metadata.",
        ),
        ModelMetadataArtifact(
            model_name="baseline_rollavg_v1",
            task_type="player_points",
            feature_version="v1",
            training_window_start_gw=1,
            training_window_end_gw=30,
            evaluation_start_gw=28,
            evaluation_end_gw=30,
            metrics_summary={
                "overall_mae": 1.0,
            },
            notes="Improved rolling-average player model metadata. Replace MAE with exact value if you have it.",
        ),
        ModelMetadataArtifact(
            model_name="ridge_rollform_v1",
            task_type="player_points",
            feature_version="v0",
            training_window_start_gw=1,
            training_window_end_gw=22,
            evaluation_start_gw=23,
            evaluation_end_gw=30,
            metrics_summary={
                "validation_mae": 0.9884,
            },
            notes="Ridge rolling-form player model metadata.",
        ),
        ModelMetadataArtifact(
            model_name="match_baseline_v0",
            task_type="match_result",
            feature_version="match_v0",
            training_window_start_gw=1,
            training_window_end_gw=29,
            evaluation_start_gw=30,
            evaluation_end_gw=30,
            metrics_summary={
                "accuracy": 0.4000,
                "logloss": 1.1762,
            },
            notes="Baseline match prediction model metadata.",
        ),
        ModelMetadataArtifact(
            model_name="match_logreg_v0",
            task_type="match_result",
            feature_version="match_v0",
            training_window_start_gw=1,
            training_window_end_gw=29,
            evaluation_start_gw=30,
            evaluation_end_gw=30,
            metrics_summary={
                "accuracy": 0.2000,
                "logloss": 1.5399,
            },
            notes="Logistic regression match prediction model metadata.",
        ),
    ]

    for artifact in artifacts:
        path = save_model_metadata_artifact(artifact)
        print("saved:", path)


if __name__ == "__main__":
    main()