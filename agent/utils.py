import re
import json
from sklearn import metrics


def extract_float(s):
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(s))
    return float(match.group()) if match else None


def fix_incomplete_json(json_text):
    json_text = json_text.strip()
    if not json_text.startswith('{') and "PERSPECTIVE STRENGTH" in json_text:
        json_text = '{' + json_text
    if "EXPLANATION" in json_text and not json_text.rstrip().endswith('}'):
        if not json_text.rstrip().endswith('"'):
            json_text += '"'
        json_text += '}'
    return json_text


def parse_llm_output_json(output_text):
    try:
        output_text = output_text.strip()
        if output_text.startswith("```"):
            output_text = re.sub(r"^```[a-zA-Z]*\n?", "", output_text)
        if output_text.endswith("```"):
            output_text = output_text[:output_text.rfind("```")]
        first_brace = output_text.find('{')
        last_brace = output_text.rfind('}')
        if first_brace == -1 and last_brace != -1:
            json_text = '{' + output_text[:last_brace+1]
        elif first_brace != -1 and last_brace == -1:
            json_text = output_text[first_brace:]
            json_text = fix_incomplete_json(json_text)
        elif first_brace == -1 and last_brace == -1:
            raise ValueError("No JSON braces found")
        else:
            json_text = output_text[first_brace:last_brace+1]
        json_text = re.sub(r"'(\w+)'(\s*:\s*)", r'"\1"\2', json_text)
        json_text = re.sub(r",\s*}", "}", json_text)
        result = json.loads(json_text)
        strength = float(result.get("PERSPECTIVE STRENGTH", 0.0))
        explanation = result.get("EXPLANATION", "")
        return {"strength": strength, "explanation": explanation}
    except Exception as e:
        return {
            "strength": 0.0,
            "explanation": f"FAILED TO PARSE JSON: {str(e)}. RAW OUTPUT: {output_text[:]}"
        }


def parse_llm_output_json_summarize(output_text):
    try:
        output_text = output_text.strip()
        if output_text.startswith("```"):
            output_text = re.sub(r"^```[a-zA-Z]*\n?", "", output_text)
        if output_text.endswith("```"):
            output_text = output_text[:output_text.rfind("```")]
        first_brace = output_text.find('{')
        last_brace = output_text.rfind('}')
        if first_brace == -1 and last_brace != -1:
            json_text = '{' + output_text[:last_brace+1]
        elif first_brace != -1 and last_brace == -1:
            json_text = output_text[first_brace:]
            json_text = fix_incomplete_json(json_text)
        elif first_brace == -1 and last_brace == -1:
            raise ValueError("No JSON braces found")
        else:
            json_text = output_text[first_brace:last_brace+1]
        json_text = re.sub(r"'(\w+)'(\s*:\s*)", r'"\1"\2', json_text)
        json_text = re.sub(r",\s*}", "}", json_text)
        result = json.loads(json_text)
        summarization = result.get("summary_sentence", "no summary")
        return {"summarization": summarization}

    except Exception as e:
        return {
            "summarization": "no summary",
        }


def parse_llm_output_json_unfied(output_text):
    try:
        output_text = output_text.strip()
        if output_text.startswith("```"):
            output_text = re.sub(r"^```[a-zA-Z]*\n?", "", output_text)
        if output_text.endswith("```"):
            output_text = output_text[:output_text.rfind("```")]
        first_brace = output_text.find('{')
        last_brace = output_text.rfind('}')
        if first_brace == -1 and last_brace != -1:
            json_text = '{' + output_text[:last_brace+1]
        elif first_brace != -1 and last_brace == -1:
            json_text = output_text[first_brace:]
            json_text = fix_incomplete_json(json_text)
        elif first_brace == -1 and last_brace == -1:
            raise ValueError("No JSON braces found")
        else:
            json_text = output_text[first_brace:last_brace+1]
        json_text = re.sub(r"'(\w+)'(\s*:\s*)", r'"\1"\2', json_text)
        json_text = re.sub(r",\s*}", "}", json_text)
        result = json.loads(json_text)
        decision = result.get("decision", "no").lower()
        return decision

    except Exception as e:
        return "no"


def eval_performance(y_true, y_pred, metric_path=None):
    # Precision
    metric_dict = {}
    precision = metrics.precision_score(y_true, y_pred, labels=[0, 1], average='binary', zero_division=0)
    print("Precision:\n\t", precision)
    metric_dict['Precision'] = precision

    # Recall
    recall = metrics.recall_score(y_true, y_pred, labels=[0, 1], average='binary', zero_division=0)
    print("Recall:\n\t", recall)
    metric_dict['Recall'] = recall

    # Accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy:\n\t", accuracy)
    metric_dict['Accuracy'] = accuracy

    print("-------------------F1, Micro-F1, Macro-F1, Weighted-F1..-------------------------")

    # F1 Score
    f1 = metrics.f1_score(y_true, y_pred, labels=[0, 1], average='binary', zero_division=0)
    print("F1 Score:\n\t", f1)
    metric_dict['F1'] = f1

    # Micro-F1 Score
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)
    print("Micro-F1 Score:\n\t", micro_f1)
    metric_dict['Micro-F1'] = micro_f1

    # Macro-F1 Score
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    print("Macro-F1 Score:\n\t", macro_f1)
    metric_dict['Macro-F1'] = macro_f1

    # Weighted-F1 Score
    weighted_f1 = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print("Weighted-F1 Score:\n\t", weighted_f1)
    metric_dict['Weighted-F1'] = weighted_f1

    print("-------------------**********************************-------------------------")

    try:
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        print("ROC AUC:\n\t", roc_auc)
        metric_dict['ROC-AUC'] = roc_auc
    except:
        print('Only one class present in y_true. ROC AUC score is not defined.')
        metric_dict['ROC-AUC'] = 0

    # Confusion matrix
    print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))

    if metric_path is not None:
        with open(metric_path, 'w', encoding='utf-8') as f:
            json.dump(metric_dict, f, indent=4)

    return metric_dict

