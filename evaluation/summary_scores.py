# Get results of evaluation

import argparse
import os
import json

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--outfile", type=str, default=None)
args = parser.parse_args()

# Load classnames

with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

# Load results

df = pd.read_json(args.filename, orient="records", lines=True)

summary_json = {}

# Measure overall success

print("Summary")
print("=======")
print(f"Total images: {len(df)}")
print(f"Total prompts: {len(df.groupby('metadata'))}")
print(f"% correct images: {df['correct'].mean():.2%}")
print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
print()
summary_json["total_images"] = int(len(df))
summary_json["total_prompts"] = int(len(df.groupby('metadata')))
summary_json["correct_images"] = df['correct'].mean()
summary_json["correct_prompts"] = df.groupby('metadata')['correct'].any().mean()

# By group

task_scores = []

print("Task breakdown")
print("==============")
for tag, task_df in df.groupby('tag', sort=False):
    task_scores.append(task_df['correct'].mean())
    print(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")
    summary_json[tag] = {
        "correct": task_df['correct'].mean(),
        "count": int(len(task_df)),
        "correct_count": int(task_df['correct'].sum())
    }
print()

print(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")
summary_json["overall_score"] = np.mean(task_scores).item()

if args.outfile is not None:
    with open(args.outfile, 'w') as outfile:
        json.dump(summary_json, outfile, indent=2)
