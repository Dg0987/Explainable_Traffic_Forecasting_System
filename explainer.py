"""
Explainer Module for Traffic Predictions
Provides feature importance, counterfactual explanations, and temporal attention analysis
"""
import numpy as np
from typing import Dict, List, Any


def explain_prediction(model, data: np.ndarray) -> Dict[str, Any]:
    """
    Generate explanation for a traffic speed prediction

    Args:
        model: Trained TrafficLSTM model
        data: Input data array of shape (seq_length, 4)
              Features: [speed, volume, hour, day_of_week]

    Returns:
        Dictionary containing:
        - current_prediction: The predicted speed
        - feature_importance: Impact of each feature
        - explanation: Human-readable explanation
        - counterfactual: What-if scenarios
        - temporal_attention: Attention weights per time step
    """
    feature_names = ['speed', 'volume', 'hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all', 'is_holiday']

    # Ensure data is 2D and float type
    if len(data.shape) == 3:
        data = data[0]  # Remove batch dimension
    data = data.astype(np.float64)

    # Get base prediction
    base_prediction = model.predict(data)

    # Get temporal attention weights
    attention_info = _get_temporal_attention_safe(model, data)

    # Calculate feature importance by perturbation
    importances = calculate_feature_importance(model, data, feature_names)

    # Generate counterfactual explanations
    counterfactuals = generate_counterfactuals(model, data, base_prediction)

    # Create human-readable explanation
    text_explanation = generate_text_explanation(
        importances,
        data,
        base_prediction,
        counterfactuals,
        attention_info
    )

    return {
        "current_prediction": round(base_prediction, 1),
        "feature_importance": importances,
        "explanation": text_explanation,
        "counterfactual": counterfactuals,
        "temporal_attention": attention_info,
        "input_summary": {
            "avg_speed": round(float(np.mean(data[:, 0])), 1),
            "avg_volume": round(float(np.mean(data[:, 1])), 0),
            "current_hour": int(data[-1, 2]),
            "day_of_week": int(data[-1, 3]),
            "is_holiday": int(data[-1, 7])
        }
    }


def _get_temporal_attention_safe(model, data: np.ndarray) -> Dict[str, Any]:
    """
    Safely extract temporal attention weights from the model.
    Returns uniform weights if the model doesn't support attention.
    """
    if hasattr(model, 'get_attention_weights'):
        try:
            attn = model.get_attention_weights(data)
            weights = attn['temporal_attention']
            seq_length = len(weights)

            labels = [f"t-{seq_length - 1 - i}" for i in range(seq_length)]

            max_idx = int(np.argmax(weights))
            peak_label = labels[max_idx]

            return {
                "weights": weights,
                "labels": labels,
                "peak_step": peak_label,
                "peak_weight": round(weights[max_idx], 4),
                "description": _describe_attention_pattern(weights, labels)
            }
        except Exception as e:
            print(f"Warning: Could not extract attention weights: {e}")

    # Fallback: uniform attention
    seq_length = 12
    uniform = [round(1.0 / seq_length, 4)] * seq_length
    labels = [f"t-{seq_length - 1 - i}" for i in range(seq_length)]
    return {
        "weights": uniform,
        "labels": labels,
        "peak_step": "N/A",
        "peak_weight": round(1.0 / seq_length, 4),
        "description": "Attention weights not available for this model."
    }


def _describe_attention_pattern(weights: list, labels: list) -> str:
    """Generate a human-readable description of the temporal attention pattern."""
    weights_arr = np.array(weights)
    max_idx = int(np.argmax(weights_arr))
    peak_weight = weights_arr[max_idx]

    # Describe recency bias
    recent_half = sum(weights[len(weights)//2:])

    if recent_half > 0.65:
        recency = "The model focuses primarily on the most recent hours."
    elif recent_half < 0.35:
        recency = "The model gives significant weight to earlier time steps."
    else:
        recency = "The model distributes attention fairly evenly across the time window."

    hours_ago = len(weights) - 1 - max_idx
    if hours_ago == 0:
        peak_desc = "the most recent hour"
    elif hours_ago == 1:
        peak_desc = "1 hour ago"
    else:
        peak_desc = f"{hours_ago} hours ago"

    return (
        f"The model attended most to {peak_desc} "
        f"(weight: {peak_weight:.1%}). {recency}"
    )


def calculate_feature_importance(model, data: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """
    Calculate feature importance using permutation method

    For each feature, we replace its values with the mean and
    measure how much the prediction changes.
    """
    base_prediction = model.predict(data)
    importances = {}

    for i, feature in enumerate(feature_names):
        # Create modified data with feature set to its mean
        modified_data = data.copy()
        modified_data[:, i] = np.mean(data[:, i])

        # Get new prediction
        new_prediction = model.predict(modified_data)

        # Calculate impact (absolute change in prediction)
        impact = abs(base_prediction - new_prediction)
        importances[feature] = round(float(impact), 2)

    # Normalize to percentages
    total = sum(importances.values()) + 1e-8
    for feature in importances:
        importances[feature] = round(importances[feature] / total * 100, 1)

    return importances


def generate_counterfactuals(model, data: np.ndarray, base_prediction: float) -> Dict[str, Any]:
    """
    Generate counterfactual explanations

    Shows what would happen if certain features changed.
    Uses large, meaningful perturbations so differences are visible.
    """
    counterfactuals = []

    avg_speed = float(np.mean(data[:, 0]))
    avg_volume = float(np.mean(data[:, 1]))

    # Scenario 1: If volume dropped to low-traffic level
    modified = data.copy()
    modified[:, 1] = max(avg_volume * 0.3, 2.0)  # Drop to 30% of current
    new_pred = model.predict(modified)
    counterfactuals.append({
        "scenario": "volume_decrease",
        "description": "If traffic volume dropped significantly (-70%)",
        "original_prediction": round(base_prediction, 2),
        "new_prediction": round(new_pred, 2),
        "change": round(new_pred - base_prediction, 2)
    })

    # Scenario 2: If volume spiked to heavy traffic
    modified = data.copy()
    modified[:, 1] = avg_volume * 3.0  # Triple the volume
    new_pred = model.predict(modified)
    counterfactuals.append({
        "scenario": "volume_increase",
        "description": "If traffic volume tripled (heavy congestion)",
        "original_prediction": round(base_prediction, 2),
        "new_prediction": round(new_pred, 2),
        "change": round(new_pred - base_prediction, 2)
    })

    # Scenario 3: If upstream speed was much higher (free-flow)
    modified = data.copy()
    modified[:, 0] = min(avg_speed + 20, 75.0)  # +20 mph
    new_pred = model.predict(modified)
    counterfactuals.append({
        "scenario": "speed_increase",
        "description": "If upstream speed was +20 mph (free-flow)",
        "original_prediction": round(base_prediction, 2),
        "new_prediction": round(new_pred, 2),
        "change": round(new_pred - base_prediction, 2)
    })

    # Scenario 4: If speed dropped (congestion ahead)
    modified = data.copy()
    modified[:, 0] = max(avg_speed - 25, 10.0)  # -25 mph
    new_pred = model.predict(modified)
    counterfactuals.append({
        "scenario": "speed_decrease",
        "description": "If upstream speed dropped -25 mph (congestion)",
        "original_prediction": round(base_prediction, 2),
        "new_prediction": round(new_pred, 2),
        "change": round(new_pred - base_prediction, 2)
    })

    # Scenario 5: If it was rush hour (hour = 8) vs current
    current_hour = data[-1, 2]
    if abs(current_hour - 8) > 2:
        modified = data.copy()
        modified[:, 2] = 8  # Set to rush hour
        new_pred = model.predict(modified)
        counterfactuals.append({
            "scenario": "rush_hour",
            "description": "If it was morning rush hour (8 AM)",
            "original_prediction": round(base_prediction, 2),
            "new_prediction": round(new_pred, 2),
            "change": round(new_pred - base_prediction, 2)
        })
    else:
        # If already near rush hour, show late-night scenario
        modified = data.copy()
        modified[:, 2] = 3  # 3 AM — empty roads
        new_pred = model.predict(modified)
        counterfactuals.append({
            "scenario": "late_night",
            "description": "If it was late night (3 AM, empty roads)",
            "original_prediction": round(base_prediction, 2),
            "new_prediction": round(new_pred, 2),
            "change": round(new_pred - base_prediction, 2)
        })

    # Scenario 6: Weekend vs weekday
    current_day = data[-1, 3]
    if current_day < 5:
        # Currently weekday → show weekend
        modified = data.copy()
        modified[:, 3] = 6  # Sunday
        new_pred = model.predict(modified)
        counterfactuals.append({
            "scenario": "weekend",
            "description": "If it was Sunday instead",
            "original_prediction": round(base_prediction, 2),
            "new_prediction": round(new_pred, 2),
            "change": round(new_pred - base_prediction, 2)
        })
    else:
        # Currently weekend → show Monday
        modified = data.copy()
        modified[:, 3] = 0  # Monday
        new_pred = model.predict(modified)
        counterfactuals.append({
            "scenario": "weekday",
            "description": "If it was Monday instead",
            "original_prediction": round(base_prediction, 2),
            "new_prediction": round(new_pred, 2),
            "change": round(new_pred - base_prediction, 2)
        })

    # Scenario 7: Heavy Rain
    if data[-1, 5] < 5.0: # If currently not raining heavily
        modified = data.copy()
        modified[:, 5] = 15.0  # heavy rain
        new_pred = model.predict(modified)
        counterfactuals.append({
            "scenario": "heavy_rain",
            "description": "If there was heavy rain (15mm/h)",
            "original_prediction": round(base_prediction, 2),
            "new_prediction": round(new_pred, 2),
            "change": round(new_pred - base_prediction, 2)
        })
        
    # Scenario 8: Holiday
    if data[-1, 7] == 0: # If currently not a holiday
        modified = data.copy()
        modified[:, 7] = 1  # make it a holiday
        new_pred = model.predict(modified)
        counterfactuals.append({
            "scenario": "holiday",
            "description": "If it was a public holiday",
            "original_prediction": round(base_prediction, 2),
            "new_prediction": round(new_pred, 2),
            "change": round(new_pred - base_prediction, 2)
        })

    return {
        "scenarios": counterfactuals,
        "most_impactful": max(counterfactuals, key=lambda x: abs(x['change']))['scenario']
    }


def generate_text_explanation(
    importances: Dict[str, float],
    data: np.ndarray,
    prediction: float,
    counterfactuals: Dict,
    attention_info: Dict = None
) -> str:
    """
    Generate human-readable explanation text
    """
    # Find most important feature
    top_feature = max(importances, key=importances.get)

    # Feature name mapping for better readability
    feature_labels = {
        'speed': 'upstream traffic speed',
        'volume': 'traffic volume',
        'hour': 'time of day',
        'day_of_week': 'day of week',
        'temp': 'temperature',
        'rain_1h': 'rainfall',
        'clouds_all': 'cloud cover',
        'is_holiday': 'holiday status'
    }

    # Determine traffic condition
    if prediction >= 55:
        condition = "free-flowing"
        condition_desc = "Traffic is expected to flow smoothly."
    elif prediction >= 40:
        condition = "moderate"
        condition_desc = "Some congestion may be present."
    else:
        condition = "congested"
        condition_desc = "Significant delays are likely."

    # Build explanation
    explanation_parts = []

    # Main prediction explanation
    explanation_parts.append(
        f"The predicted traffic speed is {prediction:.1f} mph, indicating {condition} conditions. "
        f"{condition_desc}"
    )

    # Feature importance explanation
    explanation_parts.append(
        f"This prediction is most influenced by {feature_labels[top_feature]} "
        f"({importances[top_feature]:.1f}% impact)."
    )

    # Add counterfactual insight
    if counterfactuals['scenarios']:
        most_impactful = counterfactuals['most_impactful']
        for scenario in counterfactuals['scenarios']:
            if scenario['scenario'] == most_impactful:
                change = scenario['change']
                direction = "increase" if change > 0 else "decrease"
                explanation_parts.append(
                    f"{scenario['description']}, the speed would {direction} by "
                    f"{abs(change):.1f} mph."
                )
                break

    # Add temporal attention insight
    if attention_info and attention_info.get('description'):
        explanation_parts.append(attention_info['description'])

    return " ".join(explanation_parts)


def get_feature_recommendations(prediction: float, data: np.ndarray) -> List[str]:
    """
    Generate actionable recommendations based on prediction
    """
    recommendations = []

    current_volume = np.mean(data[:, 1])
    current_hour = int(data[-1, 2])

    if prediction < 40:
        recommendations.append("Consider alternative routes to avoid congestion.")

        if 7 <= current_hour <= 9 or 16 <= current_hour <= 18:
            recommendations.append(
                "This is peak traffic time. Delaying travel by 1-2 hours could "
                "significantly improve conditions."
            )

        if current_volume > 1200:
            recommendations.append(
                "High traffic volume detected. Speed is expected to improve as "
                "volume decreases."
            )

    elif prediction < 55:
        recommendations.append("Moderate traffic expected. Allow extra travel time.")

    else:
        recommendations.append("Favorable traffic conditions. Good time to travel.")

    return recommendations
