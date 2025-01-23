import re
import os

# Paths
data_dir = "fr-en"
output_dir = "dataset/cleaned_data"
os.makedirs(output_dir, exist_ok=True)

def clean_train_data(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
        inside_transcript = False
        for line in f:
            line = line.strip()
            if "<transcript>" in line:
                inside_transcript = True
                continue
            elif "</transcript>" in line:
                inside_transcript = False
                continue
            if inside_transcript and line:
                out.write(line + "\n")

def clean_xml_data(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            match = re.search(r"<seg id=\"\d+\">(.*?)</seg>", line)
            if match:
                out.write(match.group(1).strip() + "\n")

# Clean train data
clean_train_data(f"{data_dir}/train.tags.fr-en.fr", f"{output_dir}/train.clean.fr")
clean_train_data(f"{data_dir}/train.tags.fr-en.en", f"{output_dir}/train.clean.en")

# Clean dev data
clean_xml_data(f"{data_dir}/IWSLT13.TED.dev2010.fr-en.fr.xml", f"{output_dir}/valid.clean.fr")
clean_xml_data(f"{data_dir}/IWSLT13.TED.dev2010.fr-en.en.xml", f"{output_dir}/valid.clean.en")

# Clean test data
clean_xml_data(f"{data_dir}/IWSLT13.TED.tst2010.fr-en.fr.xml", f"{output_dir}/test.clean.fr")
clean_xml_data(f"{data_dir}/IWSLT13.TED.tst2010.fr-en.en.xml", f"{output_dir}/test.clean.en")

print(f"Data Cleaning Complete! Files saved in {output_dir} directory")
