import argparse
import joblib
import pandas as pd
import warnings
from pathlib import Path

# Suppress sklearn warnings about feature names when predicting from raw arrays/dicts
warnings.filterwarnings('ignore', category=UserWarning)

MODEL_PATH = Path("outputs/models/best_model_random_forest.joblib")

def load_model():
    """Loads the trained scikit-learn pipeline from disk."""
    if not MODEL_PATH.exists():
        print(f"❌ Error: Could not find the model file at {MODEL_PATH}")
        print("Please ensure you have run Notebook 04 to train and save the model.")
        exit(1)
    
    return joblib.load(MODEL_PATH)

def run_inference(model, sample_data: dict) -> None:
    """Runs inference on a single sample and prints the result to the CLI."""
    
    print("\n📦 --- Incoming Order Data --- 📦")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    print("---------------------------------\n")
    
    # The pipeline expects a DataFrame
    input_df = pd.DataFrame([sample_data])
    
    print("🔮 Running prediction...")
    
    # Predict probabilities (returns an array like [[prob_negative, prob_positive]])
    probabilities = model.predict_proba(input_df)[0]
    prob_positive = probabilities[1]
    
    # Predict absolute class
    predicted_class = model.predict(input_df)[0]
    
    print("\n✅ --- Prediction Results --- ✅")
    print(f"  Probability of a Positive Review: {prob_positive:.1%}")
    
    if predicted_class == 1:
        print("  Verdict: The customer is highly likely to leave a POSITIVE review. 😊")
    else:
        print("  Verdict: WARNING - High risk of a NEGATIVE review! 🚨")
        print("           Consider proactive customer support intervention.")
    print("---------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Olist Customer Review Prediction - CLI Inference Script")
    parser.add_argument('--freight', type=float, help="Freight cost in Brazilian Reais", default=15.50)
    parser.add_argument('--items', type=int, help="Number of items in the order", default=1)
    parser.add_argument('--seller_rating', type=float, help="Seller's historical average review score (1-5)", default=4.8)
    
    args = parser.parse_args()
    
    # Example base order, dynamically updated with CLI arguments
    sample_order = {
        'purchase_month': 11,
        'purchase_weekday': 4,
        'purchase_hour': 14,
        'total_order_value': 120.0,
        'freight_value': args.freight,
        'freight_ratio': args.freight / (120.0 + args.freight),
        'number_of_items': args.items,
        'seller_count': 1,
        'dominant_product_category': 'health_beauty',
        'number_of_categories': 1,
        'total_product_weight_g': 500.0,
        'total_product_volume_cm3': 8000.0,
        'customer_state': 'SP',
        'seller_state': 'SP',
        'same_state': 1,
        'payment_type': 'credit_card',
        'payment_installments': 2,
        'previous_order_count': 0.0,
        'previous_total_spent': 0.0,
        'previous_average_review': float('nan'),  # Handled by the SimpleImputer in the pipeline
        'seller_average_review': args.seller_rating,
        'seller_total_orders': 150.0,
        'seller_late_delivery_rate': 0.02
    }

    print("Loading the Random Forest model pipeline...")
    pipeline = load_model()
    
    run_inference(pipeline, sample_order)
