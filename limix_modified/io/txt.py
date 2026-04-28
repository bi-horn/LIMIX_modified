import pandas as pd
import os
from re import sub

def read_txt(filename, sep=None, header=True, verbose=True, lineterminator=None):
    """
    Read a text file.

    Parameters
    ----------
    filename : str
        Path to a text file.
    sep : str
        Separator. ``None`` triggers auto-detection. Defaults to ``None``.
    header : bool
        ``True`` for file with a header; ``False`` otherwise. Defaults
        to ``True``.
    verbose : bool
        ``True`` for progress information; ``False`` otherwise.
    lineterminator : str
        Character used to terminate lines. Defaults to ``None``.

    Returns
    -------
    data : pandas dataframe
    """
    if sep is None:
        sep = _infer_separator(filename)

    header = 0 if header else None

    if verbose:
        print("Reading {}...".format(filename))

    with open(filename, 'r') as file:
        content = file.read().strip()  # Remove any trailing newlines or spaces

    if lineterminator:
        lines = content.split(lineterminator)
    else:
        lines = content.splitlines()

    if header is not None:
        columns = lines[0].split(sep)
        data = [line.split(sep) for line in lines[1:] if line]  # Ensure no empty lines are included
    else:
        columns = None
        data = [line.split(sep) for line in lines if line]  # Ensure no empty lines are included

    df = pd.DataFrame(data, columns=columns)

    return df

def _see(filepath, header, verbose=True, lineterminator=None):
    """
    Shows a human-friendly representation of a text file.

    Parameters
    ----------
    filepath : str
        Text file path.
    header : bool
        ``True`` for parsing the header; ``False`` otherwise.
    verbose : bool
        ``True`` for verbose; ``False`` otherwise.
    lineterminator : str
        Character used to terminate lines. Defaults to ``None``.

    Returns
    -------
    str
        Text file representation.
    """
    header = 0 if header else None

    if verbose:
        print("Reading {}...".format(filepath))

    sep = _infer_separator(filepath)
    df = read_txt(filepath, sep=sep, header=header, lineterminator=lineterminator)
    print(df.head())

def _count(candidates, line):
    counter = {c: 0 for c in candidates}
    for i in line:
        if i in candidates:
            counter[i] += 1
    return counter

def _update(counter, c):
    for k, v in c.items():
        if counter[k] != v:
            del counter[k]

def _infer_separator(fn):
    nmax = 9

    with open(fn, "r") as f:
        line = _remove_repeat(f.readline())
        counter = _count(set(line), line)

        for _ in range(nmax - 1):
            line = _remove_repeat(f.readline())
            if len(line) == 0:
                break
            c = _count(set(counter.keys()), line)
            _update(counter, c)
            if len(counter) == 1:
                return next(iter(counter.keys()))

    for c in set([",", "\t", " "]):
        if c in counter:
            return c

    counter = list(counter.items())
    if len(counter) == 0:
        return None

    counter = sorted(counter, key=lambda kv: kv[1])
    return counter[-1][0]

def _remove_repeat(s):
    return sub(r"(.)\1+", r"\1", s)

def _is_large_file(filepath):
    large = 1024 * 1024 * 100
    return os.path.getsize(filepath) >= large

# Example usage
directory = 'path/to/your/directory'
txt_filename = 'data.txt'
csv_filename = 'data.csv'

# Convert TXT to CSV with specific line terminator
def convert_txt_to_csv(directory, txt_filename, csv_filename, sep=None, lineterminator=None):
    txt_filepath = os.path.join(directory, txt_filename)
    csv_filepath = os.path.join(directory, csv_filename)

    if sep is None:
        sep = _infer_separator(txt_filepath)

    # Read the text file into a DataFrame
    df = read_txt(txt_filepath, sep=sep, lineterminator=lineterminator)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_filepath, index=False)

# Call the function
# convert_txt_to_csv(directory, txt_filename, csv_filename, lineterminator='\n')
