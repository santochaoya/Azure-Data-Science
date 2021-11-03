# Authors : Xiao Meng <santochaoya2@gmail.com>


def starter(title):
    """
    The title template

    Parameters
    ----------
    title : String
        Title of the module.
    """

    if not type(title) == str:
        raise TypeError(
            "Please use a string as title."
        )

    beginning = '\n'
    first_line = '#' * (len(title) + 12) + '\n'
    second_line = '#' + ' ' * (len(title) + 10) + '#' + '\n'
    title_line = '#' + ' ' * 5 + title + ' ' * 5 + '#' + '\n'
    forth_line = '#' + ' ' * (len(title) + 10) + '#' + '\n'
    final_line = '#' * (len(title) + 12) + '\n'

    return print(beginning, first_line, second_line, title_line, forth_line, final_line)

def section_label(label):
    """
    The label for sections.

    Parameters
    ----------
    label : sTRING
        The label of each section
    """

    if not type(label) == str:
        raise TypeError(
            "Please use a string as label."
        )

    beginning = '\n'
    label_line = label + ':\n'
    delimeter_line = '-' * (len(label) + 2)
    
    return print(beginning, label_line, delimeter_line)

def section_ending():
    """
    The ending line of each sections.
    """

    return print('\n' + ' ' + '-' * 125)

