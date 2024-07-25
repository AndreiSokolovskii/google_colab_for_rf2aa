from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path
DEFAULT_API_SERVER = "https://api.colabfold.com"
import numpy as np
import templates 
import logging
import requests
import time
import os
from tqdm import tqdm
import random
logger = logging.getLogger(__name__)
import tarfile
import hhsearch
import pandas
import pipeline
from numpy import ndarray
import af_common_protein
import af_res_con
import af_feature_procc as feature_processing
import af_pipe_mult as pipeline_multimer
import af_msa_pairing as msa_pairing
import af_common_protein as protein
import json
def parse_fasta(fasta_string: str) -> Tuple[List[str], List[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions
def get_queries(
    input_path: Union[str, Path], sort_queries_by: str = "length"
) -> Tuple[List[Tuple[str, str, Optional[List[str]]]], bool]:
    """Reads a directory of fasta files, a single fasta file or a csv file and returns a tuple
    of job name, sequence and the optional a3m lines"""

    input_path = Path(input_path)
    if not input_path.exists():
        raise OSError(f"{input_path} could not be found")

    if input_path.is_file():
        if input_path.suffix == ".csv" or input_path.suffix == ".tsv":
            sep = "\t" if input_path.suffix == ".tsv" else ","
            df = pandas.read_csv(input_path, sep=sep)
            assert "id" in df.columns and "sequence" in df.columns
            queries = [
                (seq_id, sequence.upper().split(":"), None)
                for seq_id, sequence in df[["id", "sequence"]].itertuples(index=False)
            ]
            for i in range(len(queries)):
                if len(queries[i][1]) == 1:
                    queries[i] = (queries[i][0], queries[i][1][0], None)
        elif input_path.suffix == ".a3m":
            (seqs, header) = parse_fasta(input_path.read_text())
            if len(seqs) == 0:
                raise ValueError(f"{input_path} is empty")
            query_sequence = seqs[0]
            # Use a list so we can easily extend this to multiple msas later
            a3m_lines = [input_path.read_text()]
            queries = [(input_path.stem, query_sequence, a3m_lines)]
        elif input_path.suffix in [".fasta", ".faa", ".fa"]:
            (sequences, headers) = parse_fasta(input_path.read_text())
            queries = []
            for sequence, header in zip(sequences, headers):
                sequence = sequence.upper()
                if sequence.count(":") == 0:
                    # Single sequence
                    queries.append((header, sequence, None))
                else:
                    # Complex mode
                    queries.append((header, sequence.upper().split(":"), None))
        else:
            raise ValueError(f"Unknown file format {input_path.suffix}")
    else:
        assert input_path.is_dir(), "Expected either an input file or a input directory"
        queries = []
        for file in sorted(input_path.iterdir()):
            if not file.is_file():
                continue
            if file.suffix.lower() not in [".a3m", ".fasta", ".faa"]:
                logger.warning(f"non-fasta/a3m file in input directory: {file}")
                continue
            (seqs, header) = parse_fasta(file.read_text())
            if len(seqs) == 0:
                logger.error(f"{file} is empty")
                continue
            query_sequence = seqs[0]
            if len(seqs) > 1 and file.suffix in [".fasta", ".faa", ".fa"]:
                logger.warning(
                    f"More than one sequence in {file}, ignoring all but the first sequence"
                )

            if file.suffix.lower() == ".a3m":
                a3m_lines = [file.read_text()]
                queries.append((file.stem, query_sequence.upper(), a3m_lines))
            else:
                if query_sequence.count(":") == 0:
                    # Single sequence
                    queries.append((file.stem, query_sequence, None))
                else:
                    # Complex mode
                    queries.append((file.stem, query_sequence.upper().split(":"), None))

    # sort by seq. len
    if sort_queries_by == "length":
        queries.sort(key=lambda t: len("".join(t[1])))

    elif sort_queries_by == "random":
        random.shuffle(queries)

    is_complex = False
    for job_number, (_, query_sequence, a3m_lines) in enumerate(queries):
        if isinstance(query_sequence, list):
            is_complex = True
            break
        if a3m_lines is not None and a3m_lines[0].startswith("#"):
            a3m_line = a3m_lines[0].splitlines()[0]
            tab_sep_entries = a3m_line[1:].split("\t")
            if len(tab_sep_entries) == 2:
                query_seq_len = tab_sep_entries[0].split(",")
                query_seq_len = list(map(int, query_seq_len))
                query_seqs_cardinality = tab_sep_entries[1].split(",")
                query_seqs_cardinality = list(map(int, query_seqs_cardinality))
                is_single_protein = (
                    True
                    if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1
                    else False
                )
                if not is_single_protein:
                    is_complex = True
                    break
    return queries, is_complex
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
def run_mmseqs2(x, prefix, use_env=True, use_filter=True,
                use_templates=False, filter=None, use_pairing=False, pairing_strategy="greedy",
                host_url="https://api.colabfold.com",
                user_agent: str = "") -> Tuple[List[str], List[str]]:
  submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

  headers = {}
  if user_agent != "":
    headers['User-Agent'] = user_agent
  else:
    logger.warning("No user agent specified. Please set a user agent (e.g., 'toolname/version contact@email') to help us debug in case of problems. This warning will become an error in the future.")

  def submit(seqs, mode, N=101):
    n, query = N, ""
    for seq in seqs:
      query += f">{n}\n{seq}\n"
      n += 1

    while True:
      error_count = 0
      try:
        # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
        # "good practice to set connect timeouts to slightly larger than a multiple of 3"
        res = requests.post(f'{host_url}/{submission_endpoint}', data={ 'q': query, 'mode': mode }, timeout=6.02, headers=headers)
      except requests.exceptions.Timeout:
        logger.warning("Timeout while submitting to MSA server. Retrying...")
        continue
      except Exception as e:
        error_count += 1
        logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
        logger.warning(f"Error: {e}")
        time.sleep(5)
        if error_count > 5:
          raise
        continue
      break

    try:
      out = res.json()
    except ValueError:
      logger.error(f"Server didn't reply with json: {res.text}")
      out = {"status":"ERROR"}
    return out

  def status(ID):
    while True:
      error_count = 0
      try:
        res = requests.get(f'{host_url}/ticket/{ID}', timeout=6.02, headers=headers)
      except requests.exceptions.Timeout:
        logger.warning("Timeout while fetching status from MSA server. Retrying...")
        continue
      except Exception as e:
        error_count += 1
        logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
        logger.warning(f"Error: {e}")
        time.sleep(5)
        if error_count > 5:
          raise
        continue
      break
    try:
      out = res.json()
    except ValueError:
      logger.error(f"Server didn't reply with json: {res.text}")
      out = {"status":"ERROR"}
    return out

  def download(ID, path):
    error_count = 0
    while True:
      try:
        res = requests.get(f'{host_url}/result/download/{ID}', timeout=6.02, headers=headers)
      except requests.exceptions.Timeout:
        logger.warning("Timeout while fetching result from MSA server. Retrying...")
        continue
      except Exception as e:
        error_count += 1
        logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
        logger.warning(f"Error: {e}")
        time.sleep(5)
        if error_count > 5:
          raise
        continue
      break
    with open(path,"wb") as out: out.write(res.content)

  # process input x
  seqs = [x] if isinstance(x, str) else x

  # compatibility to old option
  if filter is not None:
    use_filter = filter

  # setup mode
  if use_filter:
    mode = "env" if use_env else "all"
  else:
    mode = "env-nofilter" if use_env else "nofilter"

  if use_pairing:
    use_templates = False
    mode = ""
    # greedy is default, complete was the previous behavior
    if pairing_strategy == "greedy":
      mode = "pairgreedy"
    elif pairing_strategy == "complete":
      mode = "paircomplete"
    if use_env:
      mode = mode + "-env"

  # define path
  path = f"{prefix}_{mode}"
  if not os.path.isdir(path): os.mkdir(path)

  # call mmseqs2 api
  tar_gz_file = f'{path}/out.tar.gz'
  N,REDO = 101,True

  # deduplicate and keep track of order
  seqs_unique = []
  #TODO this might be slow for large sets
  [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
  Ms = [N + seqs_unique.index(seq) for seq in seqs]
  # lets do it!
  if not os.path.isfile(tar_gz_file):
    TIME_ESTIMATE = 150 * len(seqs_unique)
    # with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
    with open('tmp.txt', 'w') as pbar:
      while REDO:
        # pbar.set_description("SUBMIT")

        # Resubmit job until it goes through
        out = submit(seqs_unique, mode, N)
        while out["status"] in ["UNKNOWN", "RATELIMIT"]:
          sleep_time = 5 + random.randint(0, 5)
          logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
          # resubmit
          time.sleep(sleep_time)
          out = submit(seqs_unique, mode, N)

        if out["status"] == "ERROR":
          raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

        if out["status"] == "MAINTENANCE":
          raise Exception(f'MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')

        # wait for job to finish
        ID,TIME = out["id"],0
        # pbar.set_description(out["status"])
        while out["status"] in ["UNKNOWN","RUNNING","PENDING"]:
          t = 5 + random.randint(0,5)
          logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
          time.sleep(t)
          out = status(ID)
          # pbar.set_description(out["status"])
          if out["status"] == "RUNNING":
            TIME += t
            # pbar.update(n=t)
          #if TIME > 900 and out["status"] != "COMPLETE":
          #  # something failed on the server side, need to resubmit
          #  N += 1
          #  break

        if out["status"] == "COMPLETE":
          # if TIME < TIME_ESTIMATE:
          #   pbar.update(n=(TIME_ESTIMATE-TIME))
          REDO = False

        if out["status"] == "ERROR":
          REDO = False
          raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

      # Download results
      download(ID, tar_gz_file)

  # prep list of a3m files
  if use_pairing:
    a3m_files = [f"{path}/pair.a3m"]
  else:
    a3m_files = [f"{path}/uniref.a3m"]
    if use_env: a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

  # extract a3m files
  if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
    with tarfile.open(tar_gz_file) as tar_gz:
      tar_gz.extractall(path)

  # templates
  if use_templates:
    templates = {}
    #print("seq\tpdb\tcid\tevalue")
    for line in open(f"{path}/pdb70.m8","r"):
      p = line.rstrip().split()
      M,pdb,qid,e_value = p[0],p[1],p[2],p[10]
      M = int(M)
      if M not in templates: templates[M] = []
      templates[M].append(pdb)
      #if len(templates[M]) <= 20:
      #  print(f"{int(M)-N}\t{pdb}\t{qid}\t{e_value}")

    template_paths = {}
    for k,TMPL in templates.items():
      TMPL_PATH = f"{prefix}_{mode}/templates_{k}"
      if not os.path.isdir(TMPL_PATH):
        os.mkdir(TMPL_PATH)
        TMPL_LINE = ",".join(TMPL[:20])
        response = None
        while True:
          error_count = 0
          try:
            # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
            # "good practice to set connect timeouts to slightly larger than a multiple of 3"
            response = requests.get(f"{host_url}/template/{TMPL_LINE}", stream=True, timeout=6.02, headers=headers)
          except requests.exceptions.Timeout:
            logger.warning("Timeout while submitting to template server. Retrying...")
            continue
          except Exception as e:
            error_count += 1
            logger.warning(f"Error while fetching result from template server. Retrying... ({error_count}/5)")
            logger.warning(f"Error: {e}")
            time.sleep(5)
            if error_count > 5:
              raise
            continue
          break
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
          tar.extractall(path=TMPL_PATH)
        os.symlink("pdb70_a3m.ffindex", f"{TMPL_PATH}/pdb70_cs219.ffindex")
        with open(f"{TMPL_PATH}/pdb70_cs219.ffdata", "w") as f:
          f.write("")
      template_paths[k] = TMPL_PATH

  # gather a3m lines
  a3m_lines = {}
  for a3m_file in a3m_files:
    update_M,M = True,None
    for line in open(a3m_file,"r"):
      if len(line) > 0:
        if "\x00" in line:
          line = line.replace("\x00","")
          update_M = True
        if line.startswith(">") and update_M:
          M = int(line[1:].rstrip())
          update_M = False
          if M not in a3m_lines: a3m_lines[M] = []
        a3m_lines[M].append(line)

  # return results

  a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

  if use_templates:
    template_paths_ = []
    for n in Ms:
      if n not in template_paths:
        template_paths_.append(None)
        #print(f"{n-N}\tno_templates_found")
      else:
        template_paths_.append(template_paths[n])
    template_paths = template_paths_


  return (a3m_lines, template_paths) if use_templates else a3m_lines

def mk_mock_template(
    query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(
            output_confidence_scores[None], [num_temp, 1]
        ),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_release_date": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features

def mk_template(
    a3m_lines: str, template_path: str, query_sequence: str, jobname: str
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_path,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign2",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )
    cwd = os.getcwd() + '/' + jobname + '/A'
    if not os.path.exists(cwd):
        os.mkdir(cwd)
    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/pdb70"], path_atab_hhr=cwd
    )

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    # print("RUN of mk_template")
    return dict(templates_result.features)

def get_msa_and_templates(
    jobname: str,
    query_sequences: Union[str, List[str]],
    a3m_lines: Optional[List[str]],
    result_dir: Path,
    msa_mode: str,
    use_templates: bool,
    custom_template_path: str,
    pair_mode: str,
    pairing_strategy: str = "greedy",
    host_url: str = DEFAULT_API_SERVER,
    user_agent: str = "",
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    # from colabfold.colabfold import run_mmseqs2

    use_env = msa_mode == "mmseqs2_uniref_env" or msa_mode == "mmseqs2_uniref_env_envpair"
    use_envpair = msa_mode == "mmseqs2_uniref_env_envpair"
    if isinstance(query_sequences, str): query_sequences = [query_sequences]

    # remove duplicates before searching
    query_seqs_unique = []
    for x in query_sequences:
        if x not in query_seqs_unique:
            query_seqs_unique.append(x)

    # determine how many times is each sequence is used
    query_seqs_cardinality = [0] * len(query_seqs_unique)
    for seq in query_sequences:
        seq_idx = query_seqs_unique.index(seq)
        query_seqs_cardinality[seq_idx] += 1

    # get template features
    template_features = []
    if use_templates:
        # Skip template search when custom_template_path is provided
        if custom_template_path is not None:
            if msa_mode == "single_sequence":
                a3m_lines = []
                num = 101
                for i, seq in enumerate(query_seqs_unique):
                    a3m_lines.append(f">{num + i}\n{seq}")

            if a3m_lines is None:
                a3m_lines_mmseqs2 = run_mmseqs2(
                    query_seqs_unique,
                    str(result_dir.joinpath(jobname)),
                    use_env,
                    use_templates=False,
                    host_url=host_url,
                    user_agent=user_agent,
                )
            else:
                a3m_lines_mmseqs2 = a3m_lines
            template_paths = {}
            for index in range(0, len(query_seqs_unique)):
                template_paths[index] = custom_template_path
        else:
            a3m_lines_mmseqs2, template_paths = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_templates=True,
                host_url=host_url,
                user_agent=user_agent,
            )
        if template_paths is None:
            logger.info("No template detected")
            for index in range(0, len(query_seqs_unique)):
                template_feature = mk_mock_template(query_seqs_unique[index])
                template_features.append(template_feature)
        else:
            for index in range(0, len(query_seqs_unique)):
                if template_paths[index] is not None:
                    template_feature = mk_template(
                        a3m_lines_mmseqs2[index],
                        template_paths[index],
                        query_seqs_unique[index],
                        jobname
                    )
                    if len(template_feature["template_domain_names"]) == 0:
                        template_feature = mk_mock_template(query_seqs_unique[index])
                        logger.info(f"Sequence {index} found no templates")
                    else:
                        logger.info(
                            f"Sequence {index} found templates: {template_feature['template_domain_names'].astype(str).tolist()}"
                        )
                else:
                    template_feature = mk_mock_template(query_seqs_unique[index])
                    logger.info(f"Sequence {index} found no templates")

                template_features.append(template_feature)
    else:
        for index in range(0, len(query_seqs_unique)):
            template_feature = mk_mock_template(query_seqs_unique[index])
            template_features.append(template_feature)

    if len(query_sequences) == 1:
        pair_mode = "none"

    if pair_mode == "none" or pair_mode == "unpaired" or pair_mode == "unpaired_paired":
        if msa_mode == "single_sequence":
            a3m_lines = []
            num = 101
            for i, seq in enumerate(query_seqs_unique):
                a3m_lines.append(f">{num + i}\n{seq}")
        else:
            # find normal a3ms
            a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=False,
                host_url=host_url,
                user_agent=user_agent,
            )
    else:
        a3m_lines = None

    if msa_mode != "single_sequence" and (
        pair_mode == "paired" or pair_mode == "unpaired_paired"
    ):
        # find paired a3m if not a homooligomers
        if len(query_seqs_unique) > 1:
            paired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_envpair,
                use_pairing=True,
                pairing_strategy=pairing_strategy,
                host_url=host_url,
                user_agent=user_agent,
            )
        else:
            # homooligomers
            num = 101
            paired_a3m_lines = []
            for i in range(0, query_seqs_cardinality[0]):
                paired_a3m_lines.append(f">{num+i}\n{query_seqs_unique[0]}\n")
    else:
        paired_a3m_lines = None

    return (
        a3m_lines,
        paired_a3m_lines,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )

def pair_sequences(
    a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    a3m_line_paired = [""] * len(a3m_lines[0].splitlines())
    for n, seq in enumerate(query_sequences):
        lines = a3m_lines[n].splitlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                if n != 0:
                    line = line.replace(">", "\t", 1)
                a3m_line_paired[i] = a3m_line_paired[i] + line
            else:
                a3m_line_paired[i] = a3m_line_paired[i] + line * query_cardinality[n]
    return "\n".join(a3m_line_paired)

def pad_sequences(
    a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    _blank_seq = [
        ("-" * len(seq))
        for n, seq in enumerate(query_sequences)
        for _ in range(query_cardinality[n])
    ]
    a3m_lines_combined = []
    pos = 0
    for n, seq in enumerate(query_sequences):
        for j in range(0, query_cardinality[n]):
            lines = a3m_lines[n].split("\n")
            for a3m_line in lines:
                if len(a3m_line) == 0:
                    continue
                if a3m_line.startswith(">"):
                    a3m_lines_combined.append(a3m_line)
                else:
                    a3m_lines_combined.append(
                        "".join(_blank_seq[:pos] + [a3m_line] + _blank_seq[pos + 1 :])
                    )
            pos += 1
    return "\n".join(a3m_lines_combined)
def pair_msa(
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
    paired_msa: Optional[List[str]],
    unpaired_msa: Optional[List[str]],
) -> str:
    if paired_msa is None and unpaired_msa is not None:
        a3m_lines = pad_sequences(
            unpaired_msa, query_seqs_unique, query_seqs_cardinality
        )
    elif paired_msa is not None and unpaired_msa is not None:
        a3m_lines = (
            pair_sequences(paired_msa, query_seqs_unique, query_seqs_cardinality)
            + "\n"
            + pad_sequences(unpaired_msa, query_seqs_unique, query_seqs_cardinality)
        )
    elif paired_msa is not None and unpaired_msa is None:
        a3m_lines = pair_sequences(
            paired_msa, query_seqs_unique, query_seqs_cardinality
        )
    else:
        raise ValueError(f"Invalid pairing")
    return a3m_lines
def msa_to_str(
    unpaired_msa: List[str],
    paired_msa: List[str],
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
) -> str:
    msa = "#" + ",".join(map(str, map(len, query_seqs_unique))) + "\t"
    msa += ",".join(map(str, query_seqs_cardinality)) + "\n"
    # build msa with cardinality of 1, it makes it easier to parse and manipulate
    query_seqs_cardinality = [1 for _ in query_seqs_cardinality]
    msa += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)
    return msa


def build_monomer_feature(
    sequence: str, unpaired_msa: str, template_features: Dict[str, Any]
):
    msa = pipeline.parsers.parse_a3m(unpaired_msa)
    # gather features
    return {
        **pipeline.make_sequence_features(
            sequence=sequence, description="none", num_res=len(sequence)
        ),
        **pipeline.make_msa_features([msa]),
        **template_features,
    }

def build_multimer_feature(paired_msa: str) -> Dict[str, ndarray]:
    parsed_paired_msa = pipeline.parsers.parse_a3m(paired_msa)
    return {
        f"{k}_all_seq": v
        for k, v in pipeline.make_msa_features([parsed_paired_msa]).items()
    }

def process_multimer_features(
    features_for_chain: Dict[str, Dict[str, ndarray]],
    min_num_seq: int = 512,
) -> Dict[str, ndarray]:
    all_chain_features = {}
    for chain_id, chain_features in features_for_chain.items():
        all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
            chain_features, chain_id
        )

    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    # np_example = feature_processing.pair_and_merge(
    #    all_chain_features=all_chain_features, is_prokaryote=is_prokaryote)
    feature_processing.process_unmerged_features(all_chain_features)
    np_chains_list = list(all_chain_features.values())
    # noinspection PyProtectedMember
    pair_msa_sequences = not feature_processing._is_homomer_or_monomer(np_chains_list)
    chains = list(np_chains_list)
    chain_keys = chains[0].keys()
    updated_chains = []
    for chain_num, chain in enumerate(chains):
        new_chain = {k: v for k, v in chain.items() if "_all_seq" not in k}
        for feature_name in chain_keys:
            if feature_name.endswith("_all_seq"):
                feats_padded = msa_pairing.pad_features(
                    chain[feature_name], feature_name
                )
                new_chain[feature_name] = feats_padded
        new_chain["num_alignments_all_seq"] = np.asarray(
            len(np_chains_list[chain_num]["msa_all_seq"])
        )
        updated_chains.append(new_chain)
    np_chains_list = updated_chains
    np_chains_list = feature_processing.crop_chains(
        np_chains_list,
        msa_crop_size=feature_processing.MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    # merge_chain_features crashes if there are additional features only present in one chain
    # remove all features that are not present in all chains
    common_features = set([*np_chains_list[0]]).intersection(*np_chains_list)
    np_chains_list = [
        {key: value for (key, value) in chain.items() if key in common_features}
        for chain in np_chains_list
    ]
    np_example = feature_processing.msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    np_example = feature_processing.process_final(np_example)

    # Pad MSA to avoid zero-sized extra_msa.
    np_example = pipeline_multimer.pad_msa(np_example, min_num_seq=min_num_seq)
    return np_example

def set_model_type(is_complex: bool, model_type: str) -> str:
    # backward-compatibility with old options
    old_names = {
        "AlphaFold2-multimer-v1":"alphafold2_multimer_v1",
        "AlphaFold2-multimer-v2":"alphafold2_multimer_v2",
        "AlphaFold2-multimer-v3":"alphafold2_multimer_v3",
        "AlphaFold2-ptm":        "alphafold2_ptm",
        "AlphaFold2":            "alphafold2",
        "DeepFold":              "deepfold_v1",
    }
    model_type = old_names.get(model_type, model_type)
    if model_type == "auto":
        if is_complex:
            model_type = "alphafold2_multimer_v3"
        else:
            model_type = "alphafold2_ptm"
    return model_type

def generate_input_feature(
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
    unpaired_msa: List[str],
    paired_msa: List[str],
    template_features: List[Dict[str, Any]],
    is_complex: bool,
    model_type: str,
    max_seq: int,
) -> Tuple[Dict[str, Any], Dict[str, str]]:

    input_feature = {}
    domain_names = {}
    if is_complex and "multimer" not in model_type:

        full_sequence = ""
        Ls = []
        for sequence_index, sequence in enumerate(query_seqs_unique):
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                full_sequence += sequence
                Ls.append(len(sequence))

        # bugfix
        a3m_lines = f">0\n{full_sequence}\n"
        a3m_lines += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)

        input_feature = build_monomer_feature(full_sequence, a3m_lines, mk_mock_template(full_sequence))
        input_feature["residue_index"] = np.concatenate([np.arange(L) for L in Ls])
        input_feature["asym_id"] = np.concatenate([np.full(L,n) for n,L in enumerate(Ls)])
        if any(
            [
                template != b"none"
                for i in template_features
                for template in i["template_domain_names"]
            ]
        ):
            logger.warning(
                f"{model_type} complex does not consider templates. Chose multimer model-type for template support."
            )

    else:
        features_for_chain = {}
        chain_cnt = 0
        # for each unique sequence
        for sequence_index, sequence in enumerate(query_seqs_unique):

            # get unpaired msa
            if unpaired_msa is None:
                input_msa = f">{101 + sequence_index}\n{sequence}"
            else:
                input_msa = unpaired_msa[sequence_index]

            feature_dict = build_monomer_feature(
                sequence, input_msa, template_features[sequence_index])

            if "multimer" in model_type:
                # get paired msa
                if paired_msa is None:
                    input_msa = f">{101 + sequence_index}\n{sequence}"
                else:
                    input_msa = paired_msa[sequence_index]
                feature_dict.update(build_multimer_feature(input_msa))

            # for each copy
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                features_for_chain[protein.PDB_CHAIN_IDS[chain_cnt]] = feature_dict
                chain_cnt += 1

        if "multimer" in model_type:
            # combine features across all chains
            input_feature = process_multimer_features(features_for_chain, min_num_seq=max_seq + 4)
            domain_names = {
                chain: [
                    name.decode("UTF-8")
                    for name in feature["template_domain_names"]
                    if name != b"none"
                ]
                for (chain, feature) in features_for_chain.items()
            }
        else:
            input_feature = features_for_chain[protein.PDB_CHAIN_IDS[0]]
            input_feature["asym_id"] = np.zeros(input_feature["aatype"].shape[0],dtype=int)
            domain_names = {
                protein.PDB_CHAIN_IDS[0]: [
                    name.decode("UTF-8")
                    for name in input_feature["template_domain_names"]
                    if name != b"none"
                ]
            }
    return (input_feature, domain_names)
def main(sequence, name_of_job, USE_TEMPLATES, MSA_MODE):

  import re
  import hashlib
  def add_hash(x,y):
    return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]
  query_sequence = sequence #'GPHMATGQDRVVALVDMDCFFVQVEQRQNPHLRNKPCAVVQYKSWKGGGIIAVSYEARAFGVTRSMWADDAKKLCPDLLLAQVRESRGKANLTKYREASVEVMEIMSRFAVIERASIDEAYVDLTSAVQERLQKLQGQPISADLLPSTYIEGLPQGPTTAEETVQKEGMRKQGLFQWLDSLQIDNLTSPDLQLTVGAVIVEEMRAAIERETGFQCSAGISHNKVLAKLACGLNKPNRQTLVSHGSVPQLFSQMPIRKIRSLGGKLGASVIEILGIEYMGELTQFTESQLQSHFGEKNGSWLYAMCRGIEHDPVKPRQLPKTIGCSKNFPGKTALATREQVQWWLLQLAQELEERLTKDRNDNDRVATQLVVSIRVQGDKRLSSLRRCCALTRYDAHKMSHDAFTVIKNCNTSGIQTEWSPPLTMLFLCATKFSAS'
  query_sequence = "".join(query_sequence.split())
  jobname = name_of_job# "7U7W_1_test_2_0"
  basejobname = "".join(jobname.split())
  basejobname = re.sub(r'\W+', '', basejobname)
  jobname = add_hash(basejobname, query_sequence)
  # query_sequence =
  a3m_lines = None
  result_dir = "res"
  msa_mode = MSA_MODE #"mmseqs2_uniref_env"
  if msa_mode != 'single_sequence':
    msa_mode = "mmseqs2_uniref_env"

  custom_template_path = None
  use_templates = USE_TEMPLATES #True
  pair_mode = "unpaired_paired"
  host_url = DEFAULT_API_SERVER
  user_agent = "rf2aa/google-colab"
  pairing_strategy = "greedy"

  os.makedirs(jobname, exist_ok=True)

  # save queries
  queries_path = os.path.join(jobname, f"{jobname}.csv")
  with open(queries_path, "w") as text_file:
    text_file.write(f"id,sequence\n{jobname},{query_sequence}")

  queries, is_complex = get_queries(queries_path)
  model_type = 'auto'
  model_type = set_model_type(is_complex, model_type)
  # set_if = lambda x,y: y if x is None else x
  (max_seq, max_extra_seq) = (512, 5120)

  (raw_jobname, query_sequence, a3m_lines) = queries[0]
  result_dir = Path(jobname)

  (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) = get_msa_and_templates(jobname, query_sequence, a3m_lines, result_dir, msa_mode, use_templates, custom_template_path, pair_mode, pairing_strategy, host_url, user_agent)
  msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
  result_dir.joinpath(f"{jobname}.a3m").write_text(msa)

  (feature_dict, domain_names) \
            = generate_input_feature(query_seqs_unique, query_seqs_cardinality, unpaired_msa, paired_msa,
                                     template_features, is_complex, model_type, max_seq=max_seq)
  result_files = []
  if use_templates:
    templates_file = result_dir.joinpath(f"{jobname}_template_domain_names.json")
    templates_file.write_text(json.dumps(domain_names))
    result_files.append(templates_file)

  result_files.append(result_dir.joinpath(jobname + ".a3m"))
  
  return jobname
# print(unpaired_msa)
# print(paired_msa)
