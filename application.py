from __future__ import print_function, unicode_literals
import regex
from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt, Separator
from PyInquirer import Validator, ValidationError
from google_trans_new import google_translator
import pyfiglet
import pickle 
import pickle_dumper

with open('trained_model', 'rb') as f:
    clf = pickle.load(f)

def predict_class(user_input_from_function):
    translator = google_translator()
    if(translator.detect(user_input_from_function)[0]!='de'):
        user_input_from_function = translator.translate(user_input_from_function, lang_tgt='de')
    return (clf.predict([user_input_from_function])[0])

print(pyfiglet.figlet_format("WELCOME"))

style = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})

class EmptyValidator(Validator):
    def validate(self, value):
        if len(value.text):
            return True
        else:
            raise ValidationError(message="You cannot leave this blank!", cursor_position=len(value.text))

print("Hello! Welcome to my application!")

question1 = [
    {
        'type': 'input',
        'name': 'input_data',
        'message': 'Enter the skill you want to classify as soft, tech or none ->',
        'validate': EmptyValidator
    }
]
question2 = [
    {
        'type': 'checkbox',
        'message': 'Please select one option:',
        'name': 'exit_choice',
        'choices': [
            Separator('= Do you want to: ='),
            {
                'name': 'test other skills'
            },
            {
                'name': 'exit'
            }
        ],
        'validate': lambda answer: 'You must choose atleast one option.' \
            if len(answer2) == 0 else True
    }
]

while(1):
    answer1 = prompt(question1, style=style)
    print("Input Text:", end = '')
    pprint(answer1['input_data'])
    temp = predict_class(answer1['input_data'])
    print("Predicted Class:", end = '')
    pprint(temp)

    print("\n")

    answer2 = prompt(question2, style=style)
    # print(answer2)
    if (answer2['exit_choice'] == ['exit']):
        print("Thank you!")
        print(pyfiglet.figlet_format("The End"))
        break
    elif (answer2['exit_choice'] == []):
        print("Please press spacebar to select option before pressing enter! Program will be continued.")
    elif (answer2['exit_choice']==['test other skills', 'exit']):
        print("Please select only one option! Program will be continued.")
    else:
        pass
    print("\n")