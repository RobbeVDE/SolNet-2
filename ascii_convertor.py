import re

# Path to your input and output BibTeX files
input_file_path = 'Thesis.bib'
output_file_path = 'Thesis_converted.bib'

# Dictionary to map non-ASCII characters to their LaTeX equivalents
latex_replacements = {
    'á': "\\'a", 'é': "\\'e", 'í': "\\'i", 'ó': "\\'o", 'ú': "\\'u",
    'Á': "\\'A", 'É': "\\'E", 'Í': "\\'I", 'Ó': "\\'O", 'Ú': "\\'U",
    'à': "\\`a", 'è': "\\`e", 'ì': "\\`i", 'ò': "\\`o", 'ù': "\\`u",
    'À': "\\`A", 'È': "\\`E',  'Ò': "\\`O", 'Ù': r" \\`U ",
    'ä': '\\"a', 'ë': '\\"e', 'ï': '\\"i', 'ö': '\\"o', 'ü': '\\"u',
    'Ä': '\\"A', 'Ë': '\\"E', 'Ï': '\\"I', 'Ö': '\\"O', 'Ü': '\\"U',
    'â': "\\^a", 'ê': "\\^e", 'î': "\\^i", 'ô': "\\^o', 'û': "\\^u",
    'Â': "\\^A', 'Ê': "\\^E', 'Î': "\\^I', 'Ô': "\\^O', 'Û': "\\^U',
    'ã': "\\~a', 'ñ': "\\~n', 'õ': "\\~o",
    'Ã': "\\~A', 'Ñ': "\\~N', 'Õ': "\\~O",
    'å': "\\aa", 'Å': "\\AA",
    'æ': "\\ae", 'Æ': "\\AE",
    'ç': "\\c{c}', 'Ç': r"\\c{C}',
    'œ': "\\oe", 'Œ': "\\OE",
    'ø': "\\o", 'Ø': "\\O",
    'ß': "\\ss",
    'ÿ': '\\"y', 'Ÿ': '\\"Y'
}

# Function to replace non-ASCII characters with LaTeX equivalents
def replace_non_ascii_chars(text, replacements):
    def replace(match):
        char = match.group(0)
        return replacements.get(char, char)  # Replace if found in dictionary, else keep original

    # Regular expression to find all non-ASCII characters
    non_ascii_re = re.compile(r'[^\x00-\x7F]')
    return non_ascii_re.sub(replace, text)

# Read the input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    bibtex_content = file.read()

# Replace non-ASCII characters
converted_bibtex_content = replace_non_ascii_chars(bibtex_content, latex_replacements)

# Write the converted content to the output file
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(converted_bibtex_content)

print(f"Converted BibTeX content has been saved to {output_file_path}")
