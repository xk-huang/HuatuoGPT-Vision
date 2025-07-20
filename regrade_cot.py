"""
python regrade_cot.py outputs/cot/HuatuoGPT-Vision-7B_medical_multimodel_evaluation_data.json
"""
import click
from pathlib import Path
import json
import tqdm
import re
import collections
import pandas as pd

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
def main(input_file):
    """
    Main function to process the input file.
    
    :param input_file: Path to the input file.
    """
    input_file = Path(input_file)
    
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    output_file = input_file.with_suffix('.output.json')
    num_correct_dict = collections.defaultdict(list)
    for line in tqdm.tqdm(data):
        outputs = regrade_data(line, llava_med_rule=False)
        dataset_name = outputs["dataset_name"]
        num_correct = outputs["num_correct"]
        num_correct_dict[dataset_name].append(num_correct)
    avg_correct_dict = {
        dataset_name: sum(num_corrects) / len(num_corrects)
        for dataset_name, num_corrects in num_correct_dict.items()
    }
    with open(output_file, 'w') as file:
        json.dump(avg_correct_dict, file, indent=4)
    print(f"Regraded data saved to {output_file}")
    # save tsv
    tsv_file = input_file.with_suffix('.output.tsv')
    df = pd.DataFrame.from_dict(avg_correct_dict, orient='index', columns=['average_num_correct'])
    df.index.name = 'dataset_name'
    df.to_csv(tsv_file, sep='\t')
    print(f"Regraded data saved to {tsv_file}")

        



def _extract_answer(text: str) -> str:
    """Extract the model’s final outputs."""
    m = list(re.finditer(r"<answer>(.*?)</answer>", text, re.S))
    if m:
        text = m[-1].group(1).strip()
        first_line = re.search(r"\s*([^\n\r]+)", text)
        if first_line:
            return first_line.group(1).strip()
        else:
            return text

    m = list(re.finditer(r"answer:\s*(.*)\s*", text, re.I))
    if m:
        return m[-1].group(1).strip()

    m = list(re.finditer(r"answer is:\s*(.*)\s*", text, re.I))
    if m:
        return m[-1].group(1).strip()

    m = list(re.finditer(r"answer is\s*(.*)\s*", text, re.I))
    if m:
        return m[-1].group(1).strip()

    # m = list(re.finditer(r"answer\s*(.*)\s*", text, re.I))
    # if m:
    #     return m[-1].group(1).strip()

    m = list(re.finditer(r"is:\s*(.*)\s*", text, re.I))
    if m:
        return m[-1].group(1).strip()

    return text.strip()


def extract_answer(text):
    text = _extract_answer(text)
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    text = text.replace("<answer>", "")
    text = text.replace("</answer>", "")
    text = text.strip()
    return text


def grade_answer(prediction, answer, answer_label=None, llava_med_rule=False):
    if llava_med_rule:
        # NOTE(xk): llava med cannot follow the instruction about output format, thus we use such a loose rule
        if answer.lower() in prediction.lower():
            return True
    if answer_label is not None:
        if prediction.strip().lower() == f"{answer_label}. {answer}".strip().lower():
            return True
        elif prediction.strip().lower() == answer_label.strip().lower():
            return True

    if prediction.strip().lower() == answer.strip().lower():
        return True

    return False


FAILED_TO_CONVERT = []


def regrade_data(data, llava_med_rule=False):
    answer = data["answer"]
    answer_label = data["answer_label"]
    dataset_index = data["dataset_index"]
    options = data["options"]
    prompts = data["question"]

    model_output = data["model_output"]
    data["parsed_outputs"] = [{"output_text":model_output}]
    num_rollouts = len(data["parsed_outputs"])

    num_correct = 0
    pred_letter_list = []
    for parsed_output in data["parsed_outputs"]:
        output_text = parsed_output["output_text"]
        pred_letter = extract_answer(output_text)
        converted_pred_letter = pred_letter
        is_convert = False
        for option_answer_label, option_answer in options.items():
            if grade_answer(
                pred_letter, option_answer, option_answer_label, llava_med_rule
            ):
                converted_pred_letter = option_answer_label
                is_convert = True
                break
        if not is_convert:
            FAILED_TO_CONVERT.append(
                {
                    "pred_letter": pred_letter,
                    "answer": answer,
                    "answer_label": answer_label,
                    "dataset_index": dataset_index,
                    "output_text": output_text,
                    "prompts": prompts,
                }
            )

        parsed_output["pred_letter"] = converted_pred_letter
        if grade_answer(pred_letter, answer, answer_label, llava_med_rule):
            parsed_output["is_correct"] = True
            num_correct += 1
        else:
            parsed_output["is_correct"] = False

        pred_letter_list.append(converted_pred_letter)

    data["prev_num_correct"] = data.get("num_correct", -1)
    data["num_correct"] = num_correct
    majority_vote_pred_letter = collections.Counter(pred_letter_list).most_common(1)[0][
        0
    ]
    data["majority_vote_pred_letter"] = majority_vote_pred_letter
    data["average_num_correct"] = num_correct / num_rollouts
    data["pass_at_num_rollouts"] = num_correct > 0
    data["majority_at_num_rollouts"] = majority_vote_pred_letter == answer_label
    data["options"] = options

    return data

if __name__ == "__main__":
    main()