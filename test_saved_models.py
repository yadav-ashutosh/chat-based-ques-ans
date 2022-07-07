from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
print("started")


new_pipeline = pipeline("question-answering" ,r".\Answer-from-context")

print(new_pipeline(question = 'What is a credit limit?' , context = 'A credit limit is the maximum amount you can charge on a revolving credit account, such as a credit card.')) 
print(new_pipeline(question = 'What is a credit limit?' , context = 'A credit limit is the maximum amount you can charge on a revolving credit account, such as a credit card.')) 