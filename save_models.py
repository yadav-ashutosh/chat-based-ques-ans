from transformers import pipeline

question_identifier_pipeline = pipeline("text-classification" , model = "shahrukhx01/question-vs-statement-classifier" , tokenizer = "shahrukhx01/question-vs-statement-classifier")
question_identifier_pipeline.save_pretrained(r".\Question-identifier")  

print("Completed saving")

new_pipeline = pipeline("text-classification" ,r".\Question-identifier")
 
print(new_pipeline("What is a car?")) 
print(new_pipeline("A credit limit is the maximum amount you can charge on a revolving credit account, such as a credit card.")) 

# model_name = "deepset/roberta-base-squad2"

# ans_from_context_pipeline = pipeline("question-answering" , model = model_name, tokenizer= model_name)
# ans_from_context_pipeline.save_pretrained(r".\Answer-from-context") 

# print("Completed saving")

# new_pipeline = pipeline("question-answering" ,r".\Answer-from-context")

# print(new_pipeline(question = 'What is a credit limit?' , context = 'A credit limit is the maximum amount you can charge on a revolving credit account, such as a credit card.')) 
# print("Completed saving")