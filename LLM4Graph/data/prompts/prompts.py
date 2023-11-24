from abc import ABC, abstractmethod


"""
Utility functions for prompts
We support two types of prompts:
* Plain prompts: A plain prompt is a string that is returned as is.
* Template prompts: A template prompt with the following placeholders:
    * {background}: The background of the prompt.
    * {text}: The text content.
    * {instruction}: The instruction of the prompt.
    * {question}: The question of the prompt.
    * {answer}: The answer of the prompt.
"""


class Prompt(ABC):
    """
    Abstract class for prompts
    """
    @abstractmethod
    def generate_prompt(self, **kwargs):
        """
        Abstract method to generate a prompt.

        This method should be overridden in subclasses to generate a prompt based on the provided keyword arguments.

        Parameters:
        **kwargs: Arbitrary keyword arguments.
        """
        pass


class PromptUnit(Prompt):
    """
    Class for prompt units
    """
    def __init__(self, background: str = None, prompt: str = None, instruction: str = None, question: str = None, answer: str = None, **kwargs):
        """
        Constructor method for the PromptUnit class.

        Parameters:
        prompt (str): The prompt to use.
        """
        self.background = background
        self.prompt = prompt
        self.instruction = instruction
        self.question = question
        self.answer = answer


    def generate_prompt(self):
        """
        Method to generate a prompt.

        This method generates a prompt by returning the prompt provided in the constructor.

        Parameters:
        **kwargs: Arbitrary keyword arguments.

        Returns:
        str: The prompt.
        """
        contents = []
        if self.background is not None:
            contents.append(self.background)
        if self.prompt is not None:
            contents.append(self.prompt)
        if self.instruction is not None:
            contents.append(self.instruction)
        if self.question is not None:
            contents.append(self.question)
        if self.answer is not None:
            contents.append(self.answer)
        return '\n'.join(contents)
    






class PlainPrompt(Prompt):
    """
    Class for plain prompts
    """
    def __init__(self, prompt: str):
        """
        Constructor method for the PlainPrompt class.

        Parameters:
        prompt (str): The prompt to use.
        """
        self.prompt = prompt

    def generate_prompt(self, **kwargs):
        """
        Method to generate a prompt.

        This method generates a prompt by returning the prompt provided in the constructor.

        Parameters:
        **kwargs: Arbitrary keyword arguments.

        Returns:
        str: The prompt.
        """
        return self.prompt
    


class TemplatePrompt(Prompt):
    """
    Class for template prompts
    """
    def __init__(self, text, background = None, instruction = None, question = None, answers = [], selected_demos = []):
        """
        Constructor method for the TemplatePrompt class.

        Parameters:
        text (str): The text content of the main prompt.
        background (str): The background of the prompt.
        instruction (str): The instruction of the prompt.
        question (str): The question of the prompt.
        answers (list): The answers of the selected demons.
        selected_demos (list): The selected demos of the prompt.
        """
        assert len(answers) == len(selected_demos), "The number of answers must match the number of selected demos."
        self.text = text
        self.background = background
        self.instruction = instruction
        self.question = question
        self.answers = answers
        self.selected_demos = selected_demos
    
    def generate_prompt(self):
        """
        Method to generate a prompt.

        This method generates a prompt by returning the template provided in the constructor.


        Returns:
        str: The prompt.
        """
        demos = [PromptUnit(self.background, 
                            example,
                            self.instruction,
                            self.question,
                            answer) for example, answer in zip(self.selected_demos,  self.answers)]

        demos = [demo.generate_prompt() for demo in demos]

        this_prompt = PromptUnit(self.background, self.text, 
                                 self.instruction, self.question)
        
        this_prompt_text = this_prompt.generate_prompt()

        return this_prompt_text + '\n' + '\n'.join(demos)
    



        
