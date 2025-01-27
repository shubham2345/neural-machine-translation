import argparse
import subprocess
import os
import xml.etree.ElementTree as ET
import re

# Path joining lambda 
path = lambda x: os.path.join(os.path.dirname(__file__), x)

# Initialize parser
parser = argparse.ArgumentParser(description='Preprocess and tokenize text data')
parser.add_argument('--bpe', action='store_true', help='Enable BPE tokenization (default: False)')
args = parser.parse_args()

# Initialize paths
if not os.path.exists(path('mosesdecoder')):
    subprocess.run(['git', 'clone', 'https://github.com/moses-smt/mosesdecoder.git'])
if not os.path.exists(path('subword-nmt')):
    subprocess.run(['git', 'clone', 'https://github.com/rsennrich/subword-nmt.git'])

MOSES = path('mosesdecoder/scripts')
TOK = f'{MOSES}/tokenizer/tokenizer.perl'
CLEAN = f'{MOSES}/training/clean-corpus-n.perl'
NORM_PUNC = f'{MOSES}/tokenizer/normalize-punctuation.perl'
REM_NON_PRINT = f'{MOSES}/tokenizer/remove-non-printing-char.perl'
BPE = path('subword-nmt/subword_nmt')
BPE_TOKS = 40000

DIRS = ['prep/temp']
for d in DIRS:
    os.makedirs(path(d), exist_ok=True)

SRC = 'fr'
TAR = 'en'

# Token count logger
TOKEN_COUNT_FILE = path('outputs/token_counts.log')

def log_token_count(stage, lang, count):
    with open(TOKEN_COUNT_FILE, 'a') as f:
        f.write(f'{stage} - {lang}: {count} tokens\n')

def extract_data_from_xml(p: str, out: str, lang: str):
    xml_path = path(p)
    xml = ET.parse(xml_path)
    root = xml.getroot()
    
    segments = []
    for child in root.iter('doc'):
        seg = '\n'.join([c.text for c in child.iter('seg')])
        segments.append(seg)
    
    d = ''.join(segments)

    out = path(f'prep/temp/{out}')    
    with open(out, 'w') as f:
        f.write(d)

    subprocess.run(
        f"cat {out} | "
        f"perl {TOK} -threads 8 -a -l {lang} > {out}.tok",
        shell=True,
        check=True
    )
    
    with open(f'{out}.tok', 'r') as f:
        num_tokens = len(f.read().split())
        log_token_count(out, lang, num_tokens)

def prep_train():
    for fi in [SRC, TAR]:
        tmp_file = f'{path(DIRS[0])}/train.{fi}.tok'
        try:
            os.remove(tmp_file)
        except:
            pass

        file = path(f'fr-en/train.tags.fr-en.{fi}')
        with open(file) as f:
            content = f.read()

        pat = re.compile(r'<(transcript)>(.*?)</\1>', re.DOTALL)
        matches = re.findall(pat, content)
        
        joined = ''.join([''.join(x) for x in matches]).replace('transcript', '').strip()

        with open(f'{path(DIRS[0])}/train.{fi}', 'w') as f:
            f.write(joined)

        subprocess.run(
            f"cat {path(DIRS[0])}/train.{fi} | "
            f"perl {NORM_PUNC} {fi} | "
            f"perl {REM_NON_PRINT} | "
            f"perl {TOK} -threads 8 -a -l {fi} > {tmp_file}",
            shell=True,
            check=True,
        )

        with open(tmp_file, 'r') as f:
            num_tokens = len(f.read().split())
            log_token_count('train', fi, num_tokens)

prep_train()

for lang in [SRC, TAR]:
    extract_data_from_xml(f'fr-en/IWSLT13.TED.dev2010.fr-en.{lang}.xml', f'dev.{lang}', lang)
    extract_data_from_xml(f'fr-en/IWSLT13.TED.tst2010.fr-en.{lang}.xml', f'tst.{lang}', lang)

# Ensure Moses-tokenized files are saved
for lang in [SRC, TAR]:
    for file in ['train', 'dev', 'tst']:
        tok_file = f'prep/temp/{file}.tok.{lang}'
        untok_file = f'prep/temp/{file}.{lang}'

        print(f'Saving Moses-tokenized file: {tok_file}')
        subprocess.run(
            f"cat {untok_file} | "
            f"perl {NORM_PUNC} {lang} | "
            f"perl {REM_NON_PRINT} | "
            f"perl {TOK} -threads 8 -a -l {lang} > {tok_file}",
            shell=True,
            check=True
        )

if args.bpe:
    BPE_CODE = path('prep/code')
    TRAIN = path('prep/tot_train.fr-en')
    try:
        os.remove(TRAIN)
    except:
        pass

    t= ""
    for lang in [SRC, TAR]:
        with open(f'{path("prep/temp")}/train.{lang}') as f:
            t += f.read()
    
    with open(TRAIN, 'w') as f:
        f.write(t)

    print('Learning BPE')
    subprocess.run(
        f"cat {TRAIN} | "
        f"python {BPE}/learn_bpe.py -s {BPE_TOKS} > {BPE_CODE}",
        shell=True,
        check=True
    )

    for lang in [SRC, TAR]:
        for file in ['train', 'dev', 'tst']:
            print(f'Applying BPE to {file}.{lang}')
            subprocess.run(
                f'cat {path("prep/temp")}/{file}.{lang}.tok | '
                f'python {BPE}/apply_bpe.py -c {BPE_CODE} > {path("prep/temp")}/bpe.{file}.{lang}',
                shell=True,
                check=True
            )
            
            with open(f'{path("prep/temp")}/bpe.{file}.{lang}', 'r') as f:
                num_tokens = len(f.read().split())
                log_token_count(f'bpe-{file}', lang, num_tokens)

    for fi in ['train', 'dev']:
        subprocess.run(
            f"perl {CLEAN} -ratio 1.5 {path('prep/temp')}/bpe.{fi} {SRC} {TAR} {path('prep')}/{fi} 1 250",
            shell=True,
            check=True
        )
else:
    print('Skipping BPE tokenization as per user choice')