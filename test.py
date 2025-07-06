import pandas as pd
import json
from datetime import datetime

# --- Assume survey_log_json and prediction_log_json are populated ---
# (Using the JSON data you provided in the previous examples)
survey_log_json = """
[
    { "timestamp": "2025-06-03 08:01:10.425", "action_type": "passive", "event_type": "app_init" },
    { "timestamp": "2025-06-03 08:01:10.490", "action_type": "passive", "event_type": "question_displayed", "details": { "question_index": 1 } },
    { "timestamp": "2025-06-03 08:01:12.253", "action_type": "active", "event_type": "option_selected", "details": { "question_index": 1, "selected_option": "Yes" } },
    { "timestamp": "2025-06-03 08:01:13.426", "action_type": "active", "event_type": "next_clicked", "details": { "from_question": 1 } },
    { "timestamp": "2025-06-03 08:01:13.438", "action_type": "passive", "event_type": "question_displayed", "details": { "question_index": 2 } },
    { "timestamp": "2025-06-03 08:01:14.667", "action_type": "active", "event_type": "option_selected", "details": { "question_index": 2, "selected_option": "No" } },
    { "timestamp": "2025-06-03 08:01:15.844", "action_type": "active", "event_type": "next_clicked", "details": { "from_question": 2 } },
    { "timestamp": "2025-06-03 08:01:15.858", "action_type": "passive", "event_type": "question_displayed", "details": { "question_index": 3 } },
    { "timestamp": "2025-06-03 08:01:21.262", "action_type": "active", "event_type": "option_selected", "details": { "question_index": 3, "selected_option": "Yes" } },
    { "timestamp": "2025-06-03 08:01:22.615", "action_type": "active", "event_type": "next_clicked", "details": { "from_question": 3 } },
    { "timestamp": "2025-06-03 08:01:22.632", "action_type": "passive", "event_type": "question_displayed", "details": { "question_index": 4 } },
    { "timestamp": "2025-06-03 08:01:23.837", "action_type": "active", "event_type": "option_selected", "details": { "question_index": 4, "selected_option": "No" } },
    { "timestamp": "2025-06-03 08:01:25.179", "action_type": "active", "event_type": "next_clicked", "details": { "from_question": 4 } },
    { "timestamp": "2025-06-03 08:01:25.192", "action_type": "passive", "event_type": "question_displayed", "details": { "question_index": 5 } },
    { "timestamp": "2025-06-03 08:01:27.016", "action_type": "active", "event_type": "option_selected", "details": { "question_index": 5, "selected_option": "Yes" } },
    { "timestamp": "2025-06-03 08:01:27.893", "action_type": "active", "event_type": "option_selected", "details": { "question_index": 5, "selected_option": "No" } },
    { "timestamp": "2025-06-03 08:01:28.748", "action_type": "active", "event_type": "option_selected", "details": { "question_index": 5, "selected_option": "Yes" } },
    { "timestamp": "2025-06-03 08:01:29.649", "action_type": "active", "event_type": "option_selected", "details": { "question_index": 5, "selected_option": "No" } },
    { "timestamp": "2025-06-03 08:01:30.936", "action_type": "active", "event_type": "finish_clicked", "details": { "from_question": 5 } },
    { "timestamp": "2025-06-03 08:01:30.953", "action_type": "passive", "event_type": "survey_submitted" },
    { "timestamp": "2025-06-03 08:01:32.590", "action_type": "passive", "event_type": "survey_completed" },
    { "timestamp": "2025-06-03 08:01:32.975", "action_type": "passive", "event_type": "application_closed" }
]
"""

prediction_log_json = """
[
    { "timestamp": "2025-06-03 08:01:14.372", "predicted_label": "contempt", "confidence": 0.8101, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:15.388", "predicted_label": "contempt", "confidence": 0.8615, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:16.423", "predicted_label": "contempt", "confidence": 0.952, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:17.449", "predicted_label": "contempt", "confidence": 0.9429, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:18.476", "predicted_label": "contempt", "confidence": 1.3562, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:19.513", "predicted_label": "contempt", "confidence": 1.1924, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:20.550", "predicted_label": "contempt", "confidence": 0.8322, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:21.599", "predicted_label": "contempt", "confidence": 0.7133, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:22.638", "predicted_label": "contempt", "confidence": 0.5138, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:23.715", "predicted_label": "contempt", "confidence": 0.8264, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:24.820", "predicted_label": "sadness", "confidence": 0.8256, "predicted_index": 8 },
    { "timestamp": "2025-06-03 08:01:25.897", "predicted_label": "contempt", "confidence": 0.7645, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:26.925", "predicted_label": "contempt", "confidence": 0.5599, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:28.018", "predicted_label": "contempt", "confidence": 1.0093, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:29.067", "predicted_label": "contempt", "confidence": 0.7963, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:30.098", "predicted_label": "contempt", "confidence": 0.8754, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:31.127", "predicted_label": "contempt", "confidence": 1.0047, "predicted_index": 1 },
    { "timestamp": "2025-06-03 08:01:32.156", "predicted_label": "contempt", "confidence": 0.8889, "predicted_index": 1 }
]
"""
# --- End of JSON data ---

questions = [
    "Over the last 2 weeks, have you often been bothered by feeling down, depressed, or hopeless?",
    "Over the last 2 weeks, have you often been bothered by little interest or pleasure in doing things?",
    "Over the last 2 weeks, have you often been bothered by feeling nervous, anxious, or on edge?",
    "Over the last 2 weeks, have you often been bothered by not being able to stop or control worrying?",
    "Over the last 2 weeks, have you often been bothered by having trouble relaxing?",
]


def generate_survey_summary(survey_log_json_str, prediction_log_json_str):
    survey_data = json.loads(survey_log_json_str)
    prediction_data = json.loads(prediction_log_json_str)

    interested_events = ["question_displayed", "option_selected", "survey_submitted"]
    current_question_index = None
    survey_summary = {}
    for entry in survey_data:
        if entry["event_type"] not in interested_events:
            continue
        timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S.%f")

        if entry["event_type"] == "question_displayed":
            question_index = entry["details"].get("question_index", 0)
            if current_question_index is None:
                survey_summary[question_index] = {
                    "start_time": [timestamp],
                    "end_time": [timestamp],
                    "answers": [],
                    "emotions": [],
                }
            else:
                survey_summary[current_question_index]["end_time"][-1] = timestamp
                if question_index not in survey_summary:
                    survey_summary[question_index] = {
                        "start_time": [timestamp],
                        "end_time": [timestamp],
                        "answers": [],
                        "emotions": [],
                    }
                else:
                    survey_summary[question_index]["start_time"].append(timestamp)
                    survey_summary[question_index]["end_time"].append(timestamp)
            current_question_index = question_index

        elif entry["event_type"] == "option_selected":
            survey_summary[current_question_index]["answers"].append(
                entry["details"].get("selected_option", "N/A")
            )

        else:
            if current_question_index is not None:
                survey_summary[current_question_index]["end_time"][-1] = timestamp

    for entry in prediction_data:
        timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
        predicted_label = entry.get("predicted_label", "N/A")
        predicted_confidence = entry.get("confidence", 0.0)

        # Find the question that was active at this timestamp
        for q_idx, q_data in survey_summary.items():
            is_inside = [
                (start <= timestamp < end)
                for start, end in zip(q_data["start_time"], q_data["end_time"])
            ]
            if any(is_inside):
                q_data["emotions"].append((predicted_label, predicted_confidence))
                break

    summary_texts = [
        "Based on the survey log and prediction log, here is the summary of the survey responses:"
    ]
    for q_idx, q_data in survey_summary.items():
        question_text = (
            questions[q_idx - 1] if q_idx - 1 < len(questions) else "Unknown Question"
        )
        duration_seconds = sum(
            (end - start).total_seconds()
            for start, end in zip(q_data["start_time"], q_data["end_time"])
        )
        answer_list = q_data["answers"]
        answer_list_str = ", ".join(answer_list) if answer_list else "No answers"
        emotion_list = q_data["emotions"]
        emotion_list_str = (
            ", ".join(
                f"{label} ({confidence:.4f})" for label, confidence in emotion_list
            )
            if emotion_list
            else "No emotions detected"
        )

        summary = []
        summary.append(f"No {q_idx}")
        summary.append(f"Question : {question_text}")
        summary.append(f"Duration : {duration_seconds:.4f} seconds")
        summary.append(f"Attempted Answers : {answer_list_str}")
        summary.append(f"Detected Emotions : {emotion_list_str}")
        summary_texts.append("\n".join(summary))

    return "\n\n".join(summary_texts)


# Generate and print the summary
full_summary = generate_survey_summary(survey_log_json, prediction_log_json)
print(full_summary)
