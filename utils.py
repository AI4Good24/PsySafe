import json
import yaml
import os


def write_chat_history(sender, message, output_file):
    """
    Write the chat history into a text file.
    
    Args:
        sender (object): The sender of the message, which is agent class in autogen.
        message (str): The message content.
        output_file (str): The path to the output file.
    """
    if sender == None:
        pass
    # TODO reconstruct this part
    elif sender.name == "other_agent1" or sender.name == "other_agent2" or sender.name == "chat_manager":
        pass
    else:
        lines_to_write = [sender.name, "____________________", message, "____________________", "<eoa>"] # make it easy to seperate the each round of chat.
        with open(output_file, 'a') as file:
            for line in lines_to_write:
                file.write(line + '\n')


# for psy test

def convert_results(result, column_header):
    """
    from https://github.com/CUHK-ARISE/PsychoBench/tree/main
    extract the psy result.
    """
    result = result.strip()  # Remove leading and trailing whitespace
    try:
        result_list = [int(element.strip()[-1]) for element in result.split('\n') if element.strip()]
    except:
        result_list = ["" for element in result.split('\n')]
        print(f"Unable to capture the responses on {column_header}.")
        
    return result_list


def get_questionnaire(questionnaire_name, path):
    """
    Get a questionnaire from a JSON file.

    Args:
        questionnaire_name (str): The name of the questionnaire to retrieve.
        path (str): The path to the JSON file.

    Returns:
        dict: The questionnaire data.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the questionnaire is not found in the JSON file.
    """
    try:
        with open(path) as dataset:
            data = json.load(dataset)
    except FileNotFoundError:
        raise FileNotFoundError("The 'questionnaires.json' file does not exist.")

    # Matching by questionnaire_name in dataset
    questionnaire = None
    for item in data:
        if item["name"] == questionnaire_name:
            questionnaire = item

    if questionnaire is None:
        raise ValueError("Questionnaire not found.")

    return questionnaire

def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created successfully.")
    else:
        print(f"Folder '{path}' already exists.")

def load_json(path):
    """
    load json file
    """
    with open(path, 'r') as file: 
        data = json.load(file)
    
    return data


def load_config_yaml(path):
    """
    Load config file
    """
    with open(path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def get_prompt(prompt_file):
    """
    this function is used to generate the prompt for specific prompt.

    input: prompt_file(the path of prompt_file)
    output: content(the str prompt)
    """
    with open(prompt_file, 'r', encoding='utf-8') as file:
        content = file.read()

    return content














