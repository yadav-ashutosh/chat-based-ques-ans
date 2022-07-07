from models import nlp_models 

sample_chat =       [ 
                     "What should meg do after she comes back?" ,
                     "What is nature writing" ,
                     "What is a credit limit?" ,
                     "Nature writing is nonfiction or fiction prose or poetry about the natural environment" ,
                     "Does anyone know about gilbert white?" ,
                     "An important early figure was the parson-naturalist Gilbert White (1720 – 1793), a pioneering English naturalist and ornithologist.",
                     "You never and when meg comes back, she must go out again for a bit of picture – cord and Tom you come here I shall want you to hand me up the picture.",
                     "A credit limit is the maximum amount you can charge on a revolving credit account, such as a credit card.",
                     "Where can I raise access ?" ,
                     "go to 'www.google.com' to raise access " ]

nlp_model = nlp_models() 

def pretty(d, indent=0):
   x =1 
   for item in sample_chat:
    print("msg "+str(x)+": "+item)
    x = x+1
   print("========================================================")
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def process_chat_data():
    ques_list = []
    context_list = []
    for x in sample_chat:
        label = nlp_model.if_question_model(x)
        if label[0].get('label') == 'LABEL_1':
            ques_list.append(x)
        else:
            context_list.append(x)
    ans_dict = {}
    index , model = nlp_model.faiss(context_list)
    for x in ques_list:
        contexts = nlp_model.search(x, index , model , context_list)
        # print(x)
        # print(contexts)
        final_context = contexts[0]
        
        ans = nlp_model.answer_from_context_model(x, final_context)
        ans_dict[x] = ans.get('answer')
    pretty(ans_dict)  

process_chat_data()

