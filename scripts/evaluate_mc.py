"""Score JSONL multiple-choice predictions. Unparsed responses count as incorrect."""
import argparse
import json
from pathlib import Path
import re


def parse_answer(response, valid_letters):
    valid = tuple(str(x).upper() for x in valid_letters)
    if not valid or any(not re.fullmatch(r'[A-Z]', x) for x in valid):
        raise ValueError('Valid option letters must be single uppercase letters.')
    if not isinstance(response, str):
        return None
    text = response.strip().replace('**', '')
    letters = ''.join(valid)
    # Prefer a complete letter-only answer or an explicit answer phrase.
    plain = re.fullmatch(rf'[\s(\[]*([{letters}])[\s)\].:!]*', text, re.I)
    if plain:
        return plain.group(1).upper()
    explicit = re.findall(
        rf'(?:\b(?:the\s+)?(?:best\s+|correct\s+)?answer(?:\s+is)?\s*[:=]?\s*|<answer>\s*)'
        rf'(?:option\s*)?[\[(]?([{letters}])\b', text, re.I)
    if explicit:
        answers = set(x.upper() for x in explicit)
        return answers.pop() if len(answers) == 1 else None
    initial = re.match(rf'^\s*[\[(]?([{letters}])[\]).:]\s*', text)
    return initial.group(1) if initial else None


def score_records(records):
    rows, seen = [], set()
    for record in records:
        qid = str(record['question_id'])
        if qid in seen:
            raise ValueError(f'Duplicate question_id: {qid}')
        seen.add(qid)
        options = record['options']
        if isinstance(options, dict):
            letters = list(options)
        elif isinstance(options, list) and 2 <= len(options) <= 26:
            letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(options)])
        else:
            raise ValueError(f'Invalid options for {qid}')
        expected = str(record['answer']).strip().upper()
        if expected not in letters:
            raise ValueError(f'Reference must be a valid option letter for {qid}')
        predicted = parse_answer(record.get('response'), letters)
        rows.append({'question_id': qid, 'answer': expected, 'prediction': predicted,
                     'correct': predicted == expected})
    if not rows:
        raise ValueError('Prediction file contains no questions.')
    correct = sum(row['correct'] for row in rows)
    return {'questions': len(rows), 'correct': correct,
            'unparsed': sum(row['prediction'] is None for row in rows),
            'accuracy_percent': 100 * correct / len(rows), 'records': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(encoding='utf-8') as stream:
        result = score_records(json.loads(line) for line in stream if line.strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
