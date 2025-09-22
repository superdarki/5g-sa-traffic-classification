import joblib
import argparse
import pandas as pd


def display_feature_importances(model_path, top_n=20):
    """
    Loads a saved model bundle and lists the most important features.

    Args:
        model_path (str): The path to the saved .joblib model file.
        top_n (int): The number of top features to display.
    """
    try:
        print(f"--- Loading model bundle from: {model_path} ---")
        model_bundle = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # Extract components from the bundle
    model = model_bundle.get("model")
    feature_names = model_bundle.get("columns")

    if model is None or feature_names is None:
        print("Error: The model bundle is missing the 'model' or 'columns' key.")
        return

    # Get feature importances from the trained model
    importances = model.feature_importances_

    # Create a DataFrame for easy sorting and display
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    )

    # Sort the features by importance in descending order
    sorted_features = feature_importance_df.sort_values(
        by="importance", ascending=False
    ).reset_index(drop=True)

    # Display the results
    print(f"\n--- Top {top_n} Most Important Features for Classification ---")
    print(sorted_features.head(top_n).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List the most important features from a trained traffic classifier model."
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to the saved .joblib model bundle (e.g., 'traffic_mixed_classifier.joblib').",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top features to display (default: 20).",
    )
    args = parser.parse_args()

    display_feature_importances(args.model_file, top_n=args.top)
